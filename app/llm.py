from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import psutil

from .config import LLM_MODEL_NAME, LLM_CONFIG
from .exceptions import ModelLoadException
from .logger import setup_logger

logger = setup_logger(__name__)

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_pipe = None


def check_system_resources():
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.2f}GB)")
    else:
        logger.info("No GPU detected, will use CPU")
    
    # Check RAM
    mem = psutil.virtual_memory()
    total_ram = mem.total / (1024**3)
    available_ram = mem.available / (1024**3)
    logger.info(f"System RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available")


def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def init_llm():
    global _tokenizer, _model, _pipe
    
    if _pipe is not None:
        return _pipe
    
    try:
        logger.info(f"Loading LLM model: {LLM_MODEL_NAME}")
        check_system_resources()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        if _tokenizer is None:
            logger.info("Loading tokenizer...")
            _tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_NAME,
                trust_remote_code=True
            )
            logger.info("Tokenizer loaded")
        
        # Load model
        if _model is None:
            if device == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # Strategy based on GPU memory
                if gpu_memory >= 6:
                    logger.info("Loading with 4-bit quantization")
                    
                    quantization_config = get_quantization_config()
                    
                    _model = AutoModelForCausalLM.from_pretrained(
                        LLM_MODEL_NAME,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                    )
                    logger.info("Model loaded with 4-bit quantization")
                    
                elif gpu_memory >= 4:
                    logger.info("Loading with 8-bit quantization")
                    
                    _model = AutoModelForCausalLM.from_pretrained(
                        LLM_MODEL_NAME,
                        load_in_8bit=True,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    logger.info("Model loaded with 8-bit quantization")
                    
                else:
                    logger.info("Loading with float16")
                    
                    _model = AutoModelForCausalLM.from_pretrained(
                        LLM_MODEL_NAME,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    )
                    logger.info("Model loaded with float16")
                    
            else:
                # CPU loading
                logger.info("Loading on CPU")
                
                _model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL_NAME,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                logger.info("Model loaded on CPU")
        
        # Create pipeline
        if _pipe is None:
            logger.info("Creating inference pipeline...")
            _pipe = pipeline(
                "text-generation",
                model=_model,
                tokenizer=_tokenizer,
                **LLM_CONFIG
            )
            logger.info("Pipeline ready")
            
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                logger.info(f"GPU Memory - Free: {(gpu_memory - allocated):.2f}GB")
        
        return _pipe
    
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check internet connection for model download")
        logger.error("2. Verify you have at least 4GB GPU VRAM available")
        logger.error("3. Try closing other GPU applications")
        logger.error("4. Check CUDA/PyTorch installation")
        raise ModelLoadException(LLM_MODEL_NAME, e)


def build_prompt(
    system_instruction: str,
    context: str,
    history: List[Dict[str, str]],
    user_query: str
) -> str:
    prompt_parts = [
        "<|system|>",
        system_instruction.strip(),
        "</|system|>\n"
    ]
    
    if history:
        prompt_parts.append("<|history|>")
        recent_history = history[-10:] if len(history) > 10 else history
        
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")
        
        prompt_parts.append("</|history|>\n")
    
    if context:
        prompt_parts.append("<|context|>")
        prompt_parts.append("Info From Database:")
        prompt_parts.append(context)
        prompt_parts.append("</|context|>\n")
    
    prompt_parts.append("<|query|>")
    prompt_parts.append(f"User: {user_query}")
    prompt_parts.append("</|query|>\n")
    
    prompt_parts.append("Assistant:")
    
    return "\n".join(prompt_parts)


def generate_answer(prompt: str, max_retries: int = 2) -> str:
    for attempt in range(max_retries + 1):
        try:
            pipe = init_llm()
            
            logger.info(f"Generating response (attempt {attempt + 1}/{max_retries + 1})")
            
            # Clear CUDA cache before generation if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import time
            start_time = time.time()
            
            outputs = pipe(prompt)
            
            elapsed = time.time() - start_time
            logger.info(f"Generation completed in {elapsed:.1f} seconds")
            
            if not outputs or len(outputs) == 0:
                raise ValueError("Empty response from model")
            
            generated_text = outputs[0]["generated_text"]
            
            if "Assistant:" in generated_text:
                answer = generated_text.split("Assistant:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            answer = answer.strip()
            
            if "</s>" in answer:
                answer = answer.split("</s>")[0].strip()
            
            logger.info(f"Response: {len(answer)} characters generated")
            
            return answer
        
        except Exception as e:
            logger.error(f"Generation attempt {attempt + 1} failed: {e}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if attempt == max_retries:
                logger.error("All generation attempts failed")
                return "Error in generating answer."
    
    return "System Error."


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def clear_model_cache():
    global _tokenizer, _model, _pipe
    
    _tokenizer = None
    _model = None
    _pipe = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Model cache cleared")
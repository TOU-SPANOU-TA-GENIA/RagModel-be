# app/llm/fast_providers.py
"""
Ultra-fast LLM provider optimized for first-response speed.
"""

import time
import torch
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.core.interfaces import LLMProvider
from app.logger import setup_logger
from app.config import LLMConfig, FAST_LLM_CONFIG

logger = setup_logger(__name__)

class FastLocalModelProvider(LLMProvider):
    """
    Ultra-fast LLM provider with aggressive optimizations.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialized = False
    
    def get_model_name(self) -> str:
        return self.config.model_name
    
    def _ensure_initialized(self):
        if not self._initialized:
            self._load_model_fast()
    
    def _load_model_fast(self):
        """Aggressively optimized model loading."""
        logger.info(f"🚀 FAST loading LLM: {self.config.model_name}")
        
        start_time = time.time()
        
        try:
            # Force CUDA and aggressive settings
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            # Load tokenizer first (fast)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left"  # For batch processing
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Aggressive model loading with 4-bit quantization
            if device == "cuda":
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
            
            # Create optimized pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **FAST_LLM_CONFIG
            )
            
            self._initialized = True
            load_time = time.time() - start_time
            logger.info(f"✅ FAST LLM loaded in {load_time:.2f}s on {device}")
            
        except Exception as e:
            logger.error(f"❌ Fast LLM loading failed: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Ultra-fast generation with aggressive optimizations."""
        self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            # Clear cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Use shorter parameters for first response
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", 128),  # Even shorter for first response
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": True,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            
            # Generate
            outputs = self.pipeline(
                prompt,
                **gen_kwargs
            )
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
                
                # Extract response
                if generated_text.startswith(prompt):
                    response = generated_text[len(prompt):].strip()
                else:
                    response = generated_text.strip()
                
                # Truncate if too long (safety)
                if len(response) > 500:
                    response = response[:500] + "..."
                
                elapsed = time.time() - start_time
                logger.info(f"⚡ Generated {len(response)} chars in {elapsed:.2f}s")
                
                return response
            
            return "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I encountered an error generating a response."
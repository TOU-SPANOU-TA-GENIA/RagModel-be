# app/llm/fast_providers.py
"""
Ultra-fast LLM provider optimized for first-response speed.
FIXED: max_new_tokens properly passed from config.
"""

import time
import torch
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.core.interfaces import LLMProvider
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class FastLocalModelProvider(LLMProvider):
    """
    Ultra-fast LLM provider with aggressive optimizations.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._pipeline = None
        self._initialized = False
    
    def get_model_name(self) -> str:
        return self.config.model_name
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
    
    def _ensure_initialized(self):
        if not self._initialized:
            self._load_model_fast()
    
    def _load_model_fast(self):
        """Optimized model loading with GPU lock."""
        from app.rag.embedding_providers import get_model_loading_lock
        
        with get_model_loading_lock():
            logger.info(f"🚀 FAST loading LLM: {self.config.model_name}")
            
            start_time = time.time()
            
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    padding_side="left"
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
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
                
                # DON'T create pipeline with fixed max_new_tokens here!
                # We'll create it dynamically in generate()
                
                self._initialized = True
                load_time = time.time() - start_time
                logger.info(f"✅ FAST LLM loaded in {load_time:.2f}s on {device}")
                
            except Exception as e:
                logger.error(f"❌ Fast LLM loading failed: {e}")
                raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with proper max_new_tokens from kwargs or config."""
        self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            # Get max_tokens from kwargs, trying multiple names
            max_tokens = kwargs.get("max_tokens") or kwargs.get("max_new_tokens") or self.config.max_tokens
            temperature = kwargs.get("temperature", self.config.temperature)
            
            logger.info(f"Generating with max_new_tokens={max_tokens}, temp={temperature}")
            
            # Create pipeline with DYNAMIC max_new_tokens each time
            gen_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=max_tokens,  # Use the passed value!
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
            )
            
            outputs = gen_pipeline(prompt)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].get("generated_text", "")
                
                elapsed = time.time() - start_time
                logger.info(f"⚡ Generated {len(generated_text)} chars in {elapsed:.2f}s")
                
                return generated_text.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return "Συγγνώμη, παρουσιάστηκε σφάλμα κατά τη δημιουργία απάντησης."


class CachedFastProvider(FastLocalModelProvider):
    """Fast provider with response caching."""
    
    def __init__(self, config, cache_size: int = 100):
        super().__init__(config)
        self._cache: Dict[str, str] = {}
        self._cache_size = cache_size
    
    def generate(self, prompt: str, **kwargs) -> str:
        cache_key = f"{prompt[:100]}:{kwargs.get('max_tokens', '')}:{kwargs.get('temperature', '')}"
        
        if cache_key in self._cache:
            logger.debug("Cache hit for prompt")
            return self._cache[cache_key]
        
        response = super().generate(prompt, **kwargs)
        
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = response
        return response
    
    def clear_cache(self):
        self._cache.clear()
        logger.info("Response cache cleared")
# app/llm/fast_providers.py
"""
Ultra-fast LLM provider optimized for first-response speed.
Includes thread-safe GPU loading to prevent contention.
"""

import time
import torch
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from app.core.interfaces import LLMProvider
from app.utils.logger import setup_logger
from app.config import LLMConfig, FAST_LLM_CONFIG

logger = setup_logger(__name__)


class FastLocalModelProvider(LLMProvider):
    """
    Ultra-fast LLM provider with aggressive optimizations.
    Uses global lock to prevent GPU contention during loading.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
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
        """
        Aggressively optimized model loading with GPU lock.
        
        Uses global lock from embedding_providers to prevent
        concurrent GPU model loading (causes meta tensor errors).
        """
        # Import the global lock
        from app.rag.embedding_providers import get_model_loading_lock
        
        # Acquire global lock - only one model loads at a time
        with get_model_loading_lock():
            logger.info(f"🚀 FAST loading LLM: {self.config.model_name}")
            
            start_time = time.time()
            
            try:
                # Force CUDA and aggressive settings
                device = "cuda" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
                
                # Load tokenizer first (fast, no GPU needed)
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
            # Override config with kwargs if provided
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
            temperature = kwargs.get("temperature", self.config.temperature)
            
            # Generate with pipeline
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
            )
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].get("generated_text", "")
                
                elapsed = time.time() - start_time
                logger.info(f"⚡ Generated {len(generated_text)} chars in {elapsed:.2f}s")
                
                return generated_text.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I apologize, but I encountered an error generating a response."


class CachedFastProvider(FastLocalModelProvider):
    """
    Fast provider with response caching for repeated queries.
    """
    
    def __init__(self, config: LLMConfig, cache_size: int = 100):
        super().__init__(config)
        self._cache: Dict[str, str] = {}
        self._cache_size = cache_size
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with caching."""
        # Create cache key from prompt and relevant kwargs
        cache_key = f"{prompt}:{kwargs.get('max_tokens', '')}:{kwargs.get('temperature', '')}"
        
        # Check cache
        if cache_key in self._cache:
            logger.debug("Cache hit for prompt")
            return self._cache[cache_key]
        
        # Generate
        response = super().generate(prompt, **kwargs)
        
        # Cache response (with simple LRU eviction)
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = response
        return response
    
    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
        logger.info("Response cache cleared")
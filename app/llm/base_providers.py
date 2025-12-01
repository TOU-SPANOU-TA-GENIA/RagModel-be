# app/llm/base_providers.py
"""
Base LLM provider implementations.
"""

import time
import torch
from typing import Dict, Any, Optional
from abc import abstractmethod

from app.core.interfaces import LLMProvider
from app.llm.config import LLMConfig
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseLLMProvider(LLMProvider):
    """Base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def get_model_name(self) -> str:
        return self.config.model_name
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.to_dict()
    
    @abstractmethod
    def _load_model(self):
        """Load the model - implement in subclasses."""
        pass
    
    def _ensure_initialized(self):
        if not self._initialized:
            self._load_model()
            self._initialized = True
    
    def generate(self, prompt: str, **kwargs) -> str:
        self._ensure_initialized()
        
        start_time = time.time()
        try:
            response = self._generate_impl(prompt, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Generated {len(response)} chars in {elapsed:.2f}s")
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I apologize, but I encountered an error generating a response."
    
    @abstractmethod
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        """Generation implementation - override in subclasses."""
        pass

class LocalModelProvider(BaseLLMProvider):
    """Provider for local Hugging Face models."""
    
    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from app.core.memory_store import model_manager
        
        logger.info(f"Loading local model: {self.config.model_name}")
        
        # Check cache
        cached = model_manager.db.get_cached_model(self.config.model_name)
        if cached:
            self.model = cached["model"]
            self.tokenizer = cached["tokenizer"]
            logger.info("Using cached model")
            return
        
        device = self._get_device()
        
        # CRITICAL: This works with BOTH HuggingFace paths and local directories
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,  # ← "./offline_models/qwen3-4b" works here!
            trust_remote_code=True
        )
        
        model_kwargs = self._get_model_kwargs(device)
        
        # CRITICAL: This also works with BOTH remote and local paths
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,  # ← "./offline_models/qwen3-4b" works here!
            **model_kwargs
        )
        
        # Cache for reuse
        model_manager.db.cache_model(
            self.config.model_name,
            {"model": self.model, "tokenizer": self.tokenizer}
        )
        
        logger.info(f"Model loaded on {device}")
    
    def _get_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _get_model_kwargs(self, device: str) -> Dict[str, Any]:
        kwargs = {"trust_remote_code": True}
        
        if device == "cuda":
            kwargs["device_map"] = "auto"
            
            if self.config.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.config.quantization == "8bit":
                kwargs["load_in_8bit"] = True
            else:
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["device_map"] = "cpu"
            kwargs["torch_dtype"] = torch.float32
            kwargs["low_cpu_mem_usage"] = True
        
        return kwargs
    
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        from transformers import pipeline
        
        if not hasattr(self, 'pipeline'):
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                do_sample=True,
                repetition_penalty=1.1,
            )
        
        outputs = self.pipeline(prompt)
        
        if outputs and len(outputs) > 0:
            generated_text = outputs[0]["generated_text"]
            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip()
            return generated_text.strip()
        
        return ""

class MockLLMProvider(BaseLLMProvider):
    """Mock provider for testing."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        config = config or LLMConfig(model_name="mock-model")
        super().__init__(config)
        self.responses = {
            "default": "This is a mock response.",
            "question": "Based on the context: [mock answer]",
            "action": "I'll help with that action. [mock response]"
        }
    
    def _load_model(self):
        logger.info("Mock LLM provider initialized")
    
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        time.sleep(0.1)
        
        if "question" in prompt.lower():
            return self.responses["question"]
        elif "action" in prompt.lower() or "tool" in prompt.lower():
            return self.responses["action"]
        return self.responses["default"]


class CachedLLMProvider(BaseLLMProvider):
    """Wrapper adding caching to any LLM provider."""
    
    def __init__(self, base_provider: LLMProvider, cache_size: int = 100):
        self.base_provider = base_provider
        self.cache: Dict[str, str] = {}
        self.cache_size = cache_size
    
    def get_model_name(self) -> str:
        return f"cached_{self.base_provider.get_model_name()}"
    
    def get_config(self) -> Dict[str, Any]:
        return self.base_provider.get_config()
    
    def _load_model(self):
        pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        cache_key = self._get_cache_key(prompt, kwargs)
        
        if cache_key in self.cache:
            logger.info("Cache hit")
            return self.cache[cache_key]
        
        response = self.base_provider.generate(prompt, **kwargs)
        
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[cache_key] = response
        return response
    
    def _get_cache_key(self, prompt: str, kwargs: Dict) -> str:
        import hashlib
        return hashlib.md5(f"{prompt}_{kwargs}".encode()).hexdigest()
    
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        pass
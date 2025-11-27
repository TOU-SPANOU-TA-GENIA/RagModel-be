# app/llm/prewarmed_provider.py
"""
LLM provider that maintains a pre-warmed state for instant responses.
"""

import threading
import time
import queue
from typing import Optional

from app.llm.fast_providers import FastLocalModelProvider
from app.config import LLMConfig, LLM_MODEL_NAME
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class PreWarmedLLMProvider:
    """
    Maintains the LLM in a pre-warmed state for instant responses.
    """
    
    def __init__(self):
        self.config = LLMConfig(
            model_name=LLM_MODEL_NAME,
            max_tokens=128,  # Short for pre-warming
            quantization="4bit"
        )
        self.provider = None
        self._pre_warm_thread = None
        self._request_queue = queue.Queue()
        self._is_ready = False
    
    def initialize(self):
        """Initialize in background thread."""
        logger.info("🔥 Starting pre-warmed LLM provider...")
        
        self._pre_warm_thread = threading.Thread(target=self._pre_warm_llm)
        self._pre_warm_thread.daemon = True
        self._pre_warm_thread.start()
    
    def _pre_warm_llm(self):
        """Pre-warm the LLM with test generations."""
        try:
            # Load the model
            self.provider = FastLocalModelProvider(self.config)
            
            # Pre-warm with tiny generations
            warmup_prompts = [
                "Hello",
                "Hi",
                "Hey",
            ]
            
            for prompt in warmup_prompts:
                try:
                    start_time = time.time()
                    response = self.provider.generate(prompt)
                    warm_time = time.time() - start_time
                    logger.info(f"🔥 Pre-warm: '{prompt}' -> {len(response)} chars in {warm_time:.2f}s")
                    
                    # Small delay between warm-ups
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Pre-warm failed for '{prompt}': {e}")
            
            self._is_ready = True
            logger.info("✅ Pre-warmed LLM ready for instant responses!")
            
        except Exception as e:
            logger.error(f"❌ Pre-warming failed: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using pre-warmed provider."""
        if not self._is_ready or not self.provider:
            logger.warning("LLM not pre-warmed, falling back to direct generation")
            # Fallback to direct generation
            provider = FastLocalModelProvider(self.config)
            return provider.generate(prompt, **kwargs)
        
        return self.provider.generate(prompt, **kwargs)
    
    def is_ready(self) -> bool:
        return self._is_ready

# Global pre-warmed provider
prewarmed_llm = PreWarmedLLMProvider()
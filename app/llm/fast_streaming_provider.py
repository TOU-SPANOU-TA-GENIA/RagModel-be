# app/llm/fast_streaming_provider.py
"""
Fast Streaming LLM Provider - Combines speed optimizations with real streaming.

Key features:
1. Real token-by-token streaming (no buffering)
2. Can REUSE existing model from agent (no GPU memory duplication)
3. Reasonable max_tokens (256-512) for faster responses
4. Progress logging for debugging
"""

import time
import torch
from typing import Dict, Any, Generator, Optional
from threading import Thread, Lock
from dataclasses import dataclass

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class StreamConfig:
    """Streaming configuration."""
    model_name: str = "./offline_models/qwen3-4b"
    max_tokens: int = 256  # Reduced for faster response
    temperature: float = 0.7
    device: str = "auto"


class FastStreamingProvider:
    """
    LLM provider with TRUE real-time streaming.
    
    Can either:
    1. Reuse existing model/tokenizer (pass to constructor)
    2. Get from agent's LLM provider
    
    This avoids loading duplicate models and GPU memory issues.
    """
    
    def __init__(self, config: StreamConfig = None, model=None, tokenizer=None):
        self.config = config or StreamConfig()
        self._model = model
        self._tokenizer = tokenizer
        
    def _ensure_model_loaded(self):
        """Get model from agent or use provided model."""
        if self._model is not None and self._tokenizer is not None:
            return  # Already have model
        
        # Try to get from agent's LLM provider
        try:
            from app.agent.integration import get_agent
            agent = get_agent()
            llm = agent.llm_provider
            
            # Handle different provider types
            if hasattr(llm, 'provider') and llm.provider is not None:
                # PreWarmedLLMProvider wraps FastLocalModelProvider
                inner = llm.provider
                inner._ensure_initialized()
                self._model = inner.model
                self._tokenizer = inner.tokenizer
                logger.info("✅ Reusing model from PreWarmedLLMProvider")
            elif hasattr(llm, '_ensure_initialized'):
                llm._ensure_initialized()
                self._model = llm.model
                self._tokenizer = llm.tokenizer
                logger.info("✅ Reusing model from direct provider")
            elif hasattr(llm, 'model') and llm.model is not None:
                self._model = llm.model
                self._tokenizer = llm.tokenizer
                logger.info("✅ Reusing already-initialized model")
            else:
                raise RuntimeError(f"Unknown provider type: {type(llm).__name__}")
            
            return
        except Exception as e:
            logger.warning(f"Could not get model from agent: {e}")
        
        # Fallback: This should not happen in normal operation
        raise RuntimeError(
            "No model available. FastStreamingProvider requires either:\n"
            "1. Pass model/tokenizer to constructor\n"
            "2. Have agent initialized first"
        )
    
    @property
    def model(self):
        self._ensure_model_loaded()
        return self._model
    
    @property
    def tokenizer(self):
        self._ensure_model_loaded()
        return self._tokenizer
    
    def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = None,
        temperature: float = None
    ) -> Generator[str, None, None]:
        """
        Generate tokens one at a time.
        
        Yields:
            str: Each token as it's generated
        """
        from transformers import TextIteratorStreamer
        
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        logger.info(f"🚀 Starting generation: max_tokens={max_tokens}, temp={temperature}")
        logger.info(f"   Prompt length: {len(prompt)} chars")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_length = inputs["input_ids"].shape[1]
        logger.info(f"   Input tokens: {input_length}")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60  # Timeout per token
        )
        
        # Generation config
        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 0.1,
            "do_sample": temperature > 0,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Start generation thread
        gen_thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        gen_thread.start()
        
        # Yield tokens as they come
        start_time = time.time()
        token_count = 0
        
        try:
            for token in streamer:
                token_count += 1
                yield token
                
                # Log progress every 50 tokens
                if token_count % 50 == 0:
                    elapsed = time.time() - start_time
                    tps = token_count / elapsed if elapsed > 0 else 0
                    logger.debug(f"   Progress: {token_count} tokens, {tps:.1f} tok/s")
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n[Error: {e}]"
        
        finally:
            gen_thread.join(timeout=5)
            
            elapsed = time.time() - start_time
            tps = token_count / elapsed if elapsed > 0 else 0
            logger.info(f"✅ Generated {token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Non-streaming generation (collects all tokens)."""
        tokens = list(self.generate_stream(prompt, **kwargs))
        return "".join(tokens)


# Factory function
def create_fast_streaming_provider(config: StreamConfig = None) -> FastStreamingProvider:
    """Create a fast streaming provider instance."""
    return FastStreamingProvider(config)


# Test function
if __name__ == "__main__":
    provider = FastStreamingProvider()
    
    print("Testing streaming...")
    for token in provider.generate_stream("Πες μου μια σύντομη ιστορία."):
        print(token, end="", flush=True)
    print("\nDone!")
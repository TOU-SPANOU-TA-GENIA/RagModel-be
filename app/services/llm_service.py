# app/services/llm_service.py
"""
LLM Service for workflow handlers and other components.
Provides a simple interface to generate text using the configured LLM.
"""

import asyncio
from typing import Optional
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMService:
    """
    Simple LLM service wrapper for non-streaming generation.
    Used by workflow handlers and other components that need LLM output.
    """
    
    def __init__(self):
        self._provider = None
    
    def _get_provider(self):
        """Get or create provider (lazy initialization)."""
        if self._provider is None:
            try:
                # Try to use pre-warmed provider first
                from app.llm.prewarmed_provider import prewarmed_llm
                if prewarmed_llm.is_ready():
                    self._provider = prewarmed_llm
                    logger.info("Using pre-warmed LLM provider")
                    return self._provider
            except ImportError:
                pass
            
            try:
                # Fall back to fast local provider
                from app.llm.fast_providers import FastLocalModelProvider
                from app.config import LLMConfig, LLM_MODEL_NAME
                
                config = LLMConfig(
                    model_name=LLM_MODEL_NAME,
                    max_tokens=512,
                    temperature=0.7,
                    quantization="4bit"
                )
                self._provider = FastLocalModelProvider(config)
                logger.info("Created FastLocalModelProvider for LLM service")
            except Exception as e:
                logger.error(f"Failed to create LLM provider: {e}")
                raise
        
        return self._provider
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text response
        """
        provider = self._get_provider()
        
        # Build full prompt with system instruction
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Limit prompt size for 6GB VRAM
        max_prompt_chars = 5000
        if len(full_prompt) > max_prompt_chars:
            full_prompt = full_prompt[:max_prompt_chars] + "\n...[περικοπή]"
            logger.warning(f"Prompt truncated to {max_prompt_chars} chars for GPU memory")
        
        # Limit max_tokens for 6GB VRAM
        max_tokens = min(max_tokens, 800)
        
        logger.info(f"🤖 LLM generating response for prompt: {prompt[:80]}...")
        
        # Clear GPU memory before generation
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        except ImportError:
            pass
        
        # Run generation in thread pool to avoid blocking
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: provider.generate(full_prompt, max_tokens=max_tokens, temperature=temperature)
            )
            logger.info(f"🤖 LLM response: {response[:100]}...")
            return response
        except Exception as e:
            error_msg = str(e)
            if 'out of memory' in error_msg.lower() or 'cuda' in error_msg.lower():
                logger.error(f"GPU memory error: {e}")
                # Return a fallback response instead of crashing
                return "Συγγνώμη, παρουσιάστηκε σφάλμα κατά τη δημιουργία απάντησης."
            logger.error(f"LLM generation error: {e}")
            raise
    
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Synchronous version of generate."""
        provider = self._get_provider()
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        return provider.generate(full_prompt, max_tokens=max_tokens, temperature=temperature)


# Global singleton instance
llm_service = LLMService()
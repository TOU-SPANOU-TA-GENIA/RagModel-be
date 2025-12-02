# app/llm/providers.py
"""
LLM providers - re-exports from split modules.
"""

from app.llm.config import LLMConfig
from app.llm.base_providers import (
    BaseLLMProvider,
    LocalModelProvider,
    MockLLMProvider,
    CachedLLMProvider
)
from app.llm.prompt_builders import (
    SimplePromptBuilder,
    ToolAwarePromptBuilder
)
from app.llm.enhanced_response_cleaner import clean_response
from app.llm.streaming_provider import StreamingLLMProvider
from app.core.interfaces import LLMProvider, PromptBuilder
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

__all__ = [
    "LLMConfig",
    "BaseLLMProvider",
    "LocalModelProvider",
    "MockLLMProvider",
    "CachedLLMProvider",
    "SimplePromptBuilder",
    "ToolAwarePromptBuilder",
    "create_llm_provider",
    "create_prompt_builder"
]


def create_llm_provider(
    model_name: str = None,
    provider_type: str = "local",
    **kwargs
) -> LLMProvider:
    """Factory function to create LLM providers."""
    from app.config import LLM_MODEL_NAME
    
    config = LLMConfig(
        model_name=model_name or LLM_MODEL_NAME,
        max_tokens=kwargs.get("max_tokens", 256),
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", 0.9),
        device=kwargs.get("device", "auto"),
        quantization=kwargs.get("quantization", "4bit")
    )
    
    if provider_type == "mock":
        provider = MockLLMProvider(config)
    elif provider_type == "local":
        from app.llm.fast_providers import FastLocalModelProvider
        provider = FastLocalModelProvider(config)
    elif provider_type == "prewarmed":
        from app.llm.prewarmed_provider import prewarmed_llm
        return prewarmed_llm
    elif provider_type == "streaming":
        from app.llm.streaming_provider import create_streaming_provider
        return create_streaming_provider(config)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    logger.info(f"Created {provider_type} LLM provider")
    return provider


def create_prompt_builder(
    system_instruction: str = None,
    tools=None
) -> PromptBuilder:
    """Factory function to create prompt builders."""
    system_instruction = system_instruction or "You are a helpful AI assistant."
    
    if tools:
        return ToolAwarePromptBuilder(system_instruction, tools)
    return SimplePromptBuilder(system_instruction)
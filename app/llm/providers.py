# app/llm/providers.py
"""
Flexible LLM Provider implementations.
This replaces the monolithic llm.py with modular, swappable providers.
Easy to add new models or providers (OpenAI, Anthropic, local models, etc.)
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import time
import torch
from dataclasses import dataclass

from app.core.interfaces import LLMProvider, PromptBuilder, Context
from app.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# LLM Configuration
# ============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    model_name: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"  # auto, cuda, cpu
    quantization: Optional[str] = None  # None, "4bit", "8bit"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "device": self.device,
            "quantization": self.quantization
        }


# ============================================================================
# Base LLM Provider
# ============================================================================

class BaseLLMProvider(LLMProvider):
    """
    Base class for LLM providers with common functionality.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.config.model_name
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.to_dict()
    
    @abstractmethod
    def _load_model(self):
        """Load the model (implemented by subclasses)."""
        pass
    
    def _ensure_initialized(self):
        """Ensure model is loaded."""
        if not self._initialized:
            self._load_model()
            self._initialized = True
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
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
        """Actual generation implementation."""
        pass


# ============================================================================
# Local Model Provider (Hugging Face)
# ============================================================================

class LocalModelProvider(BaseLLMProvider):
    """
    Provider for local Hugging Face models.
    Supports your Llama model and others.
    """
    
    def _load_model(self):
        """Load the local model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from app.core.memory_store import model_manager
        
        logger.info(f"Loading local model: {self.config.model_name}")
        
        # Try to get from cache first
        cached = model_manager.db.get_cached_model(self.config.model_name)
        if cached:
            self.model = cached["model"]
            self.tokenizer = cached["tokenizer"]
            logger.info("Using cached model from memory")
            return
        
        # Determine device
        device = self._get_device()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Load model with appropriate settings
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            
            # Apply quantization if specified
            if self.config.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.config.quantization == "8bit":
                model_kwargs["load_in_8bit"] = True
            else:
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["device_map"] = "cpu"
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["low_cpu_mem_usage"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Cache the model
        model_manager.db.cache_model(
            self.config.model_name,
            {"model": self.model, "tokenizer": self.tokenizer}
        )
        
        logger.info(f"Model loaded and cached on {device}")
    
    def _get_device(self) -> str:
        """Determine which device to use."""
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU available with {gpu_memory:.2f}GB memory")
            return "cuda"
        else:
            logger.info("No GPU available, using CPU")
            return "cpu"
    
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        """Generate using local model."""
        from transformers import pipeline
        
        # Create pipeline if not exists
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
        
        # Generate
        outputs = self.pipeline(prompt)
        
        if outputs and len(outputs) > 0:
            generated_text = outputs[0]["generated_text"]
            
            # Remove the prompt from response
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            return response
        
        return ""


# ============================================================================
# Mock/Test Provider
# ============================================================================

class MockLLMProvider(BaseLLMProvider):
    """
    Mock provider for testing without loading actual models.
    Useful for development and testing.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        config = config or LLMConfig(model_name="mock-model")
        super().__init__(config)
        self.responses = {
            "default": "This is a mock response from the test LLM provider.",
            "question": "Based on the context, the answer to your question is: [mock answer]",
            "action": "I'll help you with that action. [mock action response]"
        }
    
    def _load_model(self):
        """No model to load for mock."""
        logger.info("Mock LLM provider initialized")
    
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        """Generate mock response."""
        # Simulate some processing time
        time.sleep(0.1)
        
        # Return different responses based on prompt content
        if "question" in prompt.lower():
            return self.responses["question"]
        elif "action" in prompt.lower() or "tool" in prompt.lower():
            return self.responses["action"]
        else:
            return self.responses["default"]


# ============================================================================
# Cached Provider (Wrapper)
# ============================================================================

class CachedLLMProvider(BaseLLMProvider):
    """
    Wrapper that adds caching to any LLM provider.
    Useful for development to avoid repeated API calls.
    """
    
    def __init__(self, base_provider: LLMProvider, cache_size: int = 100):
        self.base_provider = base_provider
        self.cache: Dict[str, str] = {}
        self.cache_size = cache_size
    
    def get_model_name(self) -> str:
        return f"cached_{self.base_provider.get_model_name()}"
    
    def get_config(self) -> Dict[str, Any]:
        return self.base_provider.get_config()
    
    def _load_model(self):
        """Delegate to base provider."""
        pass  # Base provider handles its own loading
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with caching."""
        # Create cache key
        cache_key = self._get_cache_key(prompt, kwargs)
        
        # Check cache
        if cache_key in self.cache:
            logger.info("Cache hit for prompt")
            return self.cache[cache_key]
        
        # Generate using base provider
        response = self.base_provider.generate(prompt, **kwargs)
        
        # Add to cache
        if len(self.cache) >= self.cache_size:
            # Simple FIFO eviction
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[cache_key] = response
        
        return response
    
    def _get_cache_key(self, prompt: str, kwargs: Dict) -> str:
        """Create cache key from prompt and parameters."""
        import hashlib
        key_str = f"{prompt}_{kwargs}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        """Not used, generate() is overridden."""
        pass


# ============================================================================
# Prompt Builders
# ============================================================================

class SimplePromptBuilder(PromptBuilder):
    """
    Simple prompt builder that creates clean, structured prompts.
    Much cleaner than the original build_prompt function!
    """
    
    def __init__(self, system_instruction: str = "You are a helpful AI assistant."):
        self.system_instruction = system_instruction
    
    def build(self, context: Context, **kwargs) -> str:
        """Build a clean, structured prompt."""
        sections = []
        
        # System instruction
        sections.append(f"<system>\n{self.system_instruction}\n</system>")
        
        # Add chat history if present
        if context.chat_history:
            history_text = self._format_history(context.chat_history)
            sections.append(f"<history>\n{history_text}\n</history>")
        
        # Add RAG context if present
        rag_context = kwargs.get("rag_context", "")
        if rag_context:
            sections.append(f"<context>\n{rag_context}\n</context>")
        
        # Add tool result if present
        tool_result = kwargs.get("tool_result")
        if tool_result:
            tool_text = self._format_tool_result(tool_result)
            sections.append(f"<tool_result>\n{tool_text}\n</tool_result>")
        
        # Add user query
        sections.append(f"<query>\n{context.query}\n</query>")
        
        # Add response instruction
        sections.append("\nAssistant:")
        
        return "\n\n".join(sections)
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format chat history."""
        # Keep only recent messages
        recent = history[-10:] if len(history) > 10 else history
        
        lines = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def _format_tool_result(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result."""
        if tool_result.get("success"):
            data = tool_result.get("data", {})
            if isinstance(data, dict) and "content" in data:
                return f"Tool execution successful:\n{data['content']}"
            else:
                return f"Tool execution successful:\n{data}"
        else:
            return f"Tool execution failed: {tool_result.get('error', 'Unknown error')}"


class ToolAwarePromptBuilder(SimplePromptBuilder):
    """
    Extended prompt builder that includes tool descriptions.
    """
    
    def __init__(self, system_instruction: str, tools: Dict[str, Any]):
        super().__init__(system_instruction)
        self.tools = tools
    
    def build(self, context: Context, **kwargs) -> str:
        """Build prompt with tool awareness."""
        # Get base prompt
        base_prompt = super().build(context, **kwargs)
        
        # Add tool descriptions if action intent
        intent = context.metadata.get("intent")
        if intent and intent.value == "action" and self.tools:
            tool_section = self._build_tool_section()
            base_prompt = base_prompt.replace(
                "<system>",
                f"<system>\n{self.system_instruction}\n\n{tool_section}"
            )
        
        return base_prompt
    
    def _build_tool_section(self) -> str:
        """Build tool description section."""
        lines = ["Available tools:"]
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)


# ============================================================================
# Factory Functions
# ============================================================================

# app/llm/providers.py (UPDATE factory function)
def create_llm_provider(
    model_name: str = None,
    provider_type: str = "local",
    **kwargs
) -> LLMProvider:
    """Factory function to create LLM providers."""
    
    # Create config
    from app.config import LLMConfig
    config = LLMConfig(
        model_name=model_name or LLM_MODEL_NAME,
        max_tokens=kwargs.get("max_tokens", 256),  # Shorter default
        temperature=kwargs.get("temperature", 0.7),
        top_p=kwargs.get("top_p", 0.9),
        device=kwargs.get("device", "auto"),
        quantization=kwargs.get("quantization", "4bit")  # Force 4-bit for speed
    )
    
    # Create provider based on type
    if provider_type == "mock":
        from .providers import MockLLMProvider
        provider = MockLLMProvider(config)
    elif provider_type == "local":
        from .fast_providers import FastLocalModelProvider
        provider = FastLocalModelProvider(config)
    elif provider_type == "prewarmed":
        from .prewarmed_provider import prewarmed_llm
        return prewarmed_llm  # Return the singleton
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    logger.info(f"Created {provider_type} LLM provider for {model_name}")
    return provider

def create_prompt_builder(
    system_instruction: str = None,
    tools: Optional[Dict[str, Any]] = None
) -> PromptBuilder:
    """
    Factory function to create prompt builders.
    """
    system_instruction = system_instruction or "You are a helpful AI assistant."
    
    if tools:
        return ToolAwarePromptBuilder(system_instruction, tools)
    else:
        return SimplePromptBuilder(system_instruction)
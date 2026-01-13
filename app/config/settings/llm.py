from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from .base import ConfigField, ConfigCategory

@dataclass
class LLMSettings:
    """General LLM parameters."""
    model_name: str = "./offline_models/qwen3-4b"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    quantization: str = "4bit"
    device: str = "auto"
    trust_remote_code: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EmbeddingSettings:
    """Embedding model parameters."""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384
    batch_size: int = 32
    cache_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class StreamingSettings:
    """Streaming response parameters."""
    enabled: bool = True
    chunk_delay_ms: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ModelsSettings:
    """
    Registry of available models. 
    Allows switching models dynamically via config without code changes.
    """
    active: str = "qwen3-4b"
    registry: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "qwen3-4b": {
            "family": "qwen",
            "path": "./offline_models/qwen3-4b",
            "prompt_template": "qwen_chat",
            "supports_thinking": True,
            "thinking_tags": {"start": ["<think>"], "end": ["</think>"]}
        }
    })
    
    def get_active_config(self) -> Dict[str, Any]:
        return self.registry.get(self.active, {})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
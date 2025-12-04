# app/config/schema_models.py
"""
Additional schema definitions for model registry and streaming.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

from app.config.schema import ConfigCategory, ConfigField


# =============================================================================
# Model Registry Configuration
# =============================================================================

@dataclass
class ModelDefinitionConfig:
    """Configuration for a single model in the registry."""
    family: str = "unknown"
    path: str = ""
    thinking_tags: Optional[Dict[str, List[str]]] = None
    response_tags: Optional[Dict[str, List[str]]] = None
    stop_tokens: List[str] = field(default_factory=list)
    prompt_template: str = "default"
    supports_thinking: bool = False
    trust_remote_code: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelDefinitionConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelsSettings:
    """Model registry settings."""
    registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active: str = "qwen3-4b"
    
    def __post_init__(self):
        """Initialize with default model if empty."""
        if not self.registry:
            self.registry = {
                "qwen3-4b": {
                    "family": "qwen",
                    "path": "./offline_models/qwen3-4b",
                    "thinking_tags": {
                        "start": ["<think>", "<thinking>", "<σκέψη>"],
                        "end": ["</think>", "</thinking>", "</σκέψη>"]
                    },
                    "response_tags": {
                        "start": ["<response>", "<απάντηση>"],
                        "end": ["</response>", "</απάντηση>"]
                    },
                    "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
                    "prompt_template": "qwen_chat",
                    "supports_thinking": True,
                    "trust_remote_code": True
                }
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry": self.registry,
            "active": self.active
        }
    
    def get_active_model_config(self) -> Optional[Dict[str, Any]]:
        """Get configuration for active model."""
        return self.registry.get(self.active)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("active", ConfigCategory.LLM,
                       "Active model ID", "str", "qwen3-4b",
                       requires_restart=True),
        ]


# =============================================================================
# Streaming Configuration
# =============================================================================

@dataclass
class StreamingSettings:
    """Streaming generation settings."""
    enabled: bool = True
    token_timeout: int = 60
    skip_prompt: bool = True
    skip_special_tokens: bool = True
    chunk_delay_ms: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("enabled", ConfigCategory.LLM,
                       "Enable streaming responses", "bool", True),
            ConfigField("token_timeout", ConfigCategory.LLM,
                       "Timeout per token (seconds)", "int", 60, 10, 300),
            ConfigField("chunk_delay_ms", ConfigCategory.LLM,
                       "Delay between chunks (ms)", "int", 20, 0, 100),
        ]


# =============================================================================
# Response Cleaning Configuration
# =============================================================================

@dataclass
class ResponseCleaningSettings:
    """Response cleaning settings."""
    enabled: bool = True
    clean_thinking: bool = True
    clean_xml_tags: bool = True
    clean_meta_commentary: bool = True
    normalize_whitespace: bool = True
    custom_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("enabled", ConfigCategory.LLM,
                       "Enable response cleaning", "bool", True),
            ConfigField("clean_thinking", ConfigCategory.LLM,
                       "Remove thinking blocks", "bool", True),
            ConfigField("clean_xml_tags", ConfigCategory.LLM,
                       "Remove XML/HTML tags", "bool", True),
            ConfigField("clean_meta_commentary", ConfigCategory.LLM,
                       "Remove meta-commentary", "bool", True),
        ]


# =============================================================================
# Prompt Template Configuration
# =============================================================================

@dataclass
class PromptTemplateConfig:
    """Configuration for a prompt template."""
    name: str = "default"
    system_format: str = "<|im_start|>system\n{system}<|im_end|>\n"
    user_format: str = "<|im_start|>user\n{user}<|im_end|>\n"
    assistant_format: str = "<|im_start|>assistant\n"
    supports_multi_turn: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromptTemplatesSettings:
    """Prompt template registry settings."""
    templates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default templates."""
        if not self.templates:
            self.templates = {
                "qwen_chat": {
                    "system_format": "<|im_start|>system\n{system}<|im_end|>\n",
                    "user_format": "<|im_start|>user\n{user}<|im_end|>\n",
                    "assistant_format": "<|im_start|>assistant\n",
                    "supports_multi_turn": True
                },
                "llama3_instruct": {
                    "system_format": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
                    "user_format": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
                    "assistant_format": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                    "supports_multi_turn": True
                },
                "default": {
                    "system_format": "System: {system}\n\n",
                    "user_format": "User: {user}\n\n",
                    "assistant_format": "Assistant: ",
                    "supports_multi_turn": True
                }
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {"templates": self.templates}
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get template by name."""
        return self.templates.get(name, self.templates.get("default"))
# app/llm/model_registry.py
"""
Model Registry - Config-driven model definitions.

Allows switching models by changing config.json without code changes.
Each model family can have different:
- Thinking tag patterns
- Response tag patterns  
- Stop tokens
- Prompt templates
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import re

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ModelDefinition:
    """Definition for a single model loaded from config."""
    id: str
    family: str
    path: str
    thinking_tags: Optional[Dict[str, List[str]]] = None
    response_tags: Optional[Dict[str, List[str]]] = None
    stop_tokens: List[str] = field(default_factory=list)
    prompt_template: str = "default"
    supports_thinking: bool = False
    trust_remote_code: bool = True
    
    @property
    def thinking_start_tags(self) -> List[str]:
        """Get thinking start tags."""
        if not self.thinking_tags:
            return []
        return self.thinking_tags.get("start", [])
    
    @property
    def thinking_end_tags(self) -> List[str]:
        """Get thinking end tags."""
        if not self.thinking_tags:
            return []
        return self.thinking_tags.get("end", [])
    
    @property
    def response_start_tags(self) -> List[str]:
        """Get response start tags."""
        if not self.response_tags:
            return []
        return self.response_tags.get("start", [])
    
    @property
    def response_end_tags(self) -> List[str]:
        """Get response end tags."""
        if not self.response_tags:
            return []
        return self.response_tags.get("end", [])
    
    def get_thinking_pattern(self) -> Optional[re.Pattern]:
        """Build regex pattern to match thinking blocks."""
        if not self.thinking_tags:
            return None
        
        start_tags = self.thinking_start_tags
        end_tags = self.thinking_end_tags
        
        if not start_tags or not end_tags:
            return None
        
        # Build alternation pattern
        start_pattern = "|".join(re.escape(t) for t in start_tags)
        end_pattern = "|".join(re.escape(t) for t in end_tags)
        
        pattern = f"({start_pattern}).*?({end_pattern})"
        return re.compile(pattern, re.DOTALL | re.IGNORECASE)
    
    @classmethod
    def from_dict(cls, model_id: str, data: Dict[str, Any]) -> "ModelDefinition":
        """Create from config dictionary."""
        return cls(
            id=model_id,
            family=data.get("family", "unknown"),
            path=data.get("path", ""),
            thinking_tags=data.get("thinking_tags"),
            response_tags=data.get("response_tags"),
            stop_tokens=data.get("stop_tokens", []),
            prompt_template=data.get("prompt_template", "default"),
            supports_thinking=data.get("supports_thinking", False),
            trust_remote_code=data.get("trust_remote_code", True),
        )


class ModelRegistry:
    """
    Registry of available models loaded from config.
    
    Usage:
        registry = ModelRegistry()
        model = registry.get_active_model()
        patterns = model.get_thinking_pattern()
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._models: Dict[str, ModelDefinition] = {}
        self._active_model_id: str = ""
        self._load_from_config()
        self._initialized = True
    
    def _load_from_config(self):
        """Load model definitions from config."""
        try:
            from app.config import config
            
            models_config = config.get_section("models")
            if not models_config:
                self._load_defaults()
                return
            
            registry = models_config.get("registry", {})
            self._active_model_id = models_config.get("active", "")
            
            for model_id, model_data in registry.items():
                self._models[model_id] = ModelDefinition.from_dict(model_id, model_data)
            
            if not self._active_model_id and self._models:
                self._active_model_id = next(iter(self._models))
            
            logger.info(f"Loaded {len(self._models)} model definitions, active: {self._active_model_id}")
            
        except Exception as e:
            logger.warning(f"Failed to load models from config: {e}, using defaults")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default model definitions."""
        self._models = {
            "qwen3-4b": ModelDefinition(
                id="qwen3-4b",
                family="qwen",
                path="./offline_models/qwen3-4b",
                thinking_tags={
                    "start": ["<think>", "<thinking>", "<σκέψη>"],
                    "end": ["</think>", "</thinking>", "</σκέψη>"]
                },
                response_tags={
                    "start": ["<response>", "<απάντηση>"],
                    "end": ["</response>", "</απάντηση>"]
                },
                stop_tokens=["<|im_end|>", "<|endoftext|>"],
                prompt_template="qwen_chat",
                supports_thinking=True,
            )
        }
        self._active_model_id = "qwen3-4b"
        logger.info("Loaded default model definitions")
    
    def get_model(self, model_id: str) -> Optional[ModelDefinition]:
        """Get model definition by ID."""
        return self._models.get(model_id)
    
    def get_active_model(self) -> Optional[ModelDefinition]:
        """Get the currently active model."""
        return self._models.get(self._active_model_id)
    
    def set_active_model(self, model_id: str) -> bool:
        """Set the active model."""
        if model_id not in self._models:
            logger.error(f"Model not found: {model_id}")
            return False
        
        self._active_model_id = model_id
        logger.info(f"Active model set to: {model_id}")
        return True
    
    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self._models.keys())
    
    def register_model(self, model_id: str, definition: ModelDefinition):
        """Register a new model definition."""
        self._models[model_id] = definition
        logger.info(f"Registered model: {model_id}")
    
    def reload(self):
        """Reload models from config."""
        self._models.clear()
        self._load_from_config()


# Global singleton
model_registry = ModelRegistry()


def get_active_model() -> Optional[ModelDefinition]:
    """Convenience function to get active model."""
    return model_registry.get_active_model()


def get_thinking_pattern() -> Optional[re.Pattern]:
    """Get thinking pattern for active model."""
    model = model_registry.get_active_model()
    if model:
        return model.get_thinking_pattern()
    return None
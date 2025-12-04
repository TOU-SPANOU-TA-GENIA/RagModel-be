# app/llm/prompt_templates/registry.py
"""
Prompt template registry - loads templates from config.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PromptTemplate:
    """A prompt template definition."""
    name: str
    system_format: str
    user_format: str
    assistant_format: str
    supports_multi_turn: bool = True
    
    def format_system(self, content: str) -> str:
        """Format system message."""
        return self.system_format.format(system=content)
    
    def format_user(self, content: str) -> str:
        """Format user message."""
        return self.user_format.format(user=content)
    
    def format_assistant(self, content: str = "") -> str:
        """Format assistant message (or start marker)."""
        if "{assistant}" in self.assistant_format:
            return self.assistant_format.format(assistant=content)
        return self.assistant_format + content
    
    def build_prompt(
        self,
        system: str,
        messages: List[Dict[str, str]],
        include_assistant_start: bool = True
    ) -> str:
        """
        Build complete prompt from system and messages.
        
        Args:
            system: System instruction
            messages: List of {"role": "user"|"assistant", "content": "..."}
            include_assistant_start: Add assistant format at end
        """
        parts = [self.format_system(system)]
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                parts.append(self.format_user(content))
            elif role == "assistant":
                parts.append(self.format_assistant(content))
        
        if include_assistant_start:
            parts.append(self.assistant_format)
        
        return "".join(parts)
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from config dictionary."""
        return cls(
            name=name,
            system_format=data.get("system_format", "System: {system}\n\n"),
            user_format=data.get("user_format", "User: {user}\n\n"),
            assistant_format=data.get("assistant_format", "Assistant: "),
            supports_multi_turn=data.get("supports_multi_turn", True),
        )


class PromptTemplateRegistry:
    """
    Registry of prompt templates loaded from config.
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
        
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_from_config()
        self._initialized = True
    
    def _load_from_config(self):
        """Load templates from config."""
        try:
            from app.config import config
            
            templates_config = config.get_section("prompt_templates")
            if templates_config:
                templates_dict = templates_config.get("templates", {})
                for name, data in templates_dict.items():
                    self._templates[name] = PromptTemplate.from_dict(name, data)
                logger.info(f"Loaded {len(self._templates)} prompt templates from config")
        except Exception as e:
            logger.warning(f"Failed to load templates from config: {e}")
        
        # Ensure defaults exist
        self._ensure_defaults()
    
    def _ensure_defaults(self):
        """Ensure default templates exist."""
        if "default" not in self._templates:
            self._templates["default"] = PromptTemplate(
                name="default",
                system_format="System: {system}\n\n",
                user_format="User: {user}\n\n",
                assistant_format="Assistant: ",
            )
        
        if "qwen_chat" not in self._templates:
            self._templates["qwen_chat"] = PromptTemplate(
                name="qwen_chat",
                system_format="<|im_start|>system\n{system}<|im_end|>\n",
                user_format="<|im_start|>user\n{user}<|im_end|>\n",
                assistant_format="<|im_start|>assistant\n",
            )
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        return self._templates.get(name)
    
    def get_or_default(self, name: str) -> PromptTemplate:
        """Get template by name, falling back to default."""
        return self._templates.get(name, self._templates["default"])
    
    def register(self, template: PromptTemplate):
        """Register a new template."""
        self._templates[template.name] = template
    
    def list_templates(self) -> List[str]:
        """List all registered template names."""
        return list(self._templates.keys())
    
    def reload(self):
        """Reload templates from config."""
        self._templates.clear()
        self._load_from_config()


# Global singleton
template_registry = PromptTemplateRegistry()


def get_template_for_model(model_id: str = None) -> PromptTemplate:
    """
    Get the appropriate prompt template for a model.
    
    Looks up model in registry to find its template name,
    then returns that template.
    """
    try:
        from app.llm.model_registry import model_registry
        
        if model_id:
            model = model_registry.get_model(model_id)
        else:
            model = model_registry.get_active_model()
        
        if model:
            template_name = model.prompt_template
            return template_registry.get_or_default(template_name)
    except Exception as e:
        logger.debug(f"Could not get model template: {e}")
    
    return template_registry.get_or_default("default")
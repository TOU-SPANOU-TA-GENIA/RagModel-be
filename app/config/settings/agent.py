from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from .base import ConfigField, ConfigCategory

@dataclass
class AgentSettings:
    """Agent core behavior configuration."""
    mode: str = "production"
    debug_mode: bool = False
    system_prompt_key: str = "default_greek"
    max_tool_retries: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PromptTemplatesSettings:
    """
    Dynamic prompt templates.
    Allows changing the persona or system instruction via JSON.
    """
    templates: Dict[str, str] = field(default_factory=lambda: {
        "default_greek": "Είσαι ένας χρήσιμος βοηθός AI. Απάντησε με ακρίβεια.",
        "strict_analyst": "Είσαι ένας αυστηρός αναλυτής δεδομένων. Μην κάνεις υποθέσεις."
    })
    
    def get_template(self, key: str) -> str:
        return self.templates.get(key, self.templates.get("default_greek", ""))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ToolSettings:
    """Tool execution restrictions and settings."""
    enabled: bool = True
    max_file_size_mb: int = 10
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.txt', '.md', '.json', '.csv', '.py', '.log'
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
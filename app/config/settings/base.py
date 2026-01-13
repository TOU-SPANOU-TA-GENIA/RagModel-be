from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Dict

class ConfigCategory(Enum):
    """Categories for grouping configurations in the UI."""
    LLM = "llm"
    RAG = "rag"
    AGENT = "agent"
    TOOLS = "tools"
    SERVER = "server"
    STORAGE = "storage"
    LOGGING = "logging"

@dataclass
class ConfigField:
    """Metadata for a configuration field to generate UI forms automatically."""
    name: str
    category: ConfigCategory
    description: str
    field_type: str  # "int", "float", "str", "bool", "list", "dict", "path"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[List[str]] = None
    editable: bool = True
    requires_restart: bool = False
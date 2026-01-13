from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum

class AgentIntent(str, Enum):
    CHAT = "chat"
    TOOL_USE = "tool_use"
    UNKNOWN = "unknown"

@dataclass
class AgentContext:
    """State object passed through the pipeline."""
    query: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline state
    intent: AgentIntent = AgentIntent.UNKNOWN
    suggested_tool: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)
    rag_docs: List[Any] = field(default_factory=list)
    
    # Outputs
    response: Optional[str] = None
    thinking: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    
    def add_source(self, source: str):
        if source not in self.sources:
            self.sources.append(source)

@dataclass
class AgentResponse:
    """Final output returned to the application."""
    answer: str
    thinking: str = ""
    sources: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
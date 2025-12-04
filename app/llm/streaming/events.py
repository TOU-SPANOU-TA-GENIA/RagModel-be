# app/llm/streaming/events.py
"""
Streaming event types and data structures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import json


class StreamEventType(Enum):
    """Types of streaming events."""
    TOKEN = "token"
    THINKING_START = "thinking_start"
    THINKING_END = "thinking_end"
    RESPONSE_START = "response_start"
    RESPONSE_END = "response_end"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamEvent:
    """Single streaming event."""
    event_type: StreamEventType
    data: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        payload = {
            "type": self.event_type.value,
            "data": self.data
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.event_type.value,
            "data": self.data,
            "metadata": self.metadata
        }
    
    @classmethod
    def token(cls, text: str) -> "StreamEvent":
        """Create a token event."""
        return cls(StreamEventType.TOKEN, text)
    
    @classmethod
    def thinking_start(cls) -> "StreamEvent":
        """Create thinking start event."""
        return cls(StreamEventType.THINKING_START)
    
    @classmethod
    def thinking_end(cls) -> "StreamEvent":
        """Create thinking end event."""
        return cls(StreamEventType.THINKING_END)
    
    @classmethod
    def response_start(cls) -> "StreamEvent":
        """Create response start event."""
        return cls(StreamEventType.RESPONSE_START)
    
    @classmethod
    def response_end(cls) -> "StreamEvent":
        """Create response end event."""
        return cls(StreamEventType.RESPONSE_END)
    
    @classmethod
    def error(cls, message: str) -> "StreamEvent":
        """Create error event."""
        return cls(StreamEventType.ERROR, message)
    
    @classmethod
    def done(cls) -> "StreamEvent":
        """Create done event."""
        return cls(StreamEventType.DONE)
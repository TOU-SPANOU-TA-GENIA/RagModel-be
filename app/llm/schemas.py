from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class GenerationRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = Field(default_factory=list)
    # Allows overriding the specific model config for a single request
    model_config_override: Optional[Dict[str, Any]] = None

class GenerationResponse(BaseModel):
    content: str          # The final clean answer
    raw_content: str      # The full raw string (including <think> tags)
    thinking: str = ""    # Extracted thought process
    usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: str = "stop"
    model_used: str
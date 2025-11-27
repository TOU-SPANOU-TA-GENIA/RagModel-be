# app/llm/config.py
"""
LLM configuration and settings.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    model_name: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"
    quantization: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "device": self.device,
            "quantization": self.quantization
        }
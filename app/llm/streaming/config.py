# app/llm/streaming/config.py
"""
Streaming configuration loaded from config.json.
"""

from dataclasses import dataclass
from typing import Optional

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class StreamConfig:
    """Configuration for streaming generation."""
    enabled: bool = True
    token_timeout: int = 60
    skip_prompt: bool = True
    skip_special_tokens: bool = True
    chunk_delay_ms: int = 20
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    @classmethod
    def from_config(cls) -> "StreamConfig":
        """Load from app config."""
        try:
            from app.config import config, LLM
            
            streaming_section = config.get_section("streaming") or {}
            
            return cls(
                enabled=streaming_section.get("enabled", True),
                token_timeout=streaming_section.get("token_timeout", 60),
                skip_prompt=streaming_section.get("skip_prompt", True),
                skip_special_tokens=streaming_section.get("skip_special_tokens", True),
                chunk_delay_ms=streaming_section.get("chunk_delay_ms", 20),
                max_tokens=LLM.max_new_tokens,
                temperature=LLM.temperature,
                top_p=LLM.top_p,
                repetition_penalty=LLM.repetition_penalty,
            )
        except Exception as e:
            logger.warning(f"Could not load streaming config: {e}, using defaults")
            return cls()
    
    @property
    def chunk_delay_seconds(self) -> float:
        """Get chunk delay in seconds."""
        return self.chunk_delay_ms / 1000.0


def get_stream_config() -> StreamConfig:
    """Get streaming configuration."""
    return StreamConfig.from_config()
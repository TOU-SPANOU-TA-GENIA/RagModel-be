# app/llm/response_cleaners/base.py
"""
Base classes for response cleaning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CleaningResult:
    """Result of response cleaning with metadata."""
    cleaned: str
    original_length: int
    cleaned_length: int
    removed_thinking: bool = False
    removed_tags: int = 0
    removed_meta: int = 0
    phases_applied: List[str] = field(default_factory=list)
    
    @property
    def reduction_percent(self) -> float:
        """Calculate percent reduction."""
        if self.original_length == 0:
            return 0.0
        return (1 - self.cleaned_length / self.original_length) * 100


class ResponseCleanerBase(ABC):
    """Base class for all response cleaners."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Cleaner name for logging."""
        pass
    
    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean the text and return result."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if this cleaner is enabled via config."""
        return True
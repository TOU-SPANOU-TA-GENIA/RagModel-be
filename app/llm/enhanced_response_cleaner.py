# app/llm/enhanced_response_cleaner.py
"""
Enhanced Response Cleaner - Backward compatible wrapper.

This module maintains the original API while using the new modular
cleaning pipeline internally.

Original functionality preserved:
- clean_response(text) -> str
- clean_response_detailed(text) -> CleaningResult
- clean_streaming_token(token, state) -> (str, dict)
"""

from typing import Tuple

# Import from new modular system
from app.llm.response_cleaners import (
    CleaningResult,
    CleaningPipeline,
    create_default_pipeline,
    StreamingCleaner,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class EnhancedResponseCleaner:
    """
    Multi-phase response cleaner for production quality.
    
    This is a facade over the new modular cleaning pipeline,
    maintaining backward compatibility with existing code.
    """
    
    def __init__(self):
        self._pipeline = create_default_pipeline()
        self._streaming_cleaner = StreamingCleaner()
    
    def clean(self, response: str) -> str:
        """
        Clean response with all phases.
        Returns clean, user-ready text.
        """
        if not response:
            return response
        
        return self._pipeline.clean(response)
    
    def clean_with_details(self, response: str) -> CleaningResult:
        """
        Clean response and return detailed results.
        Useful for debugging and logging.
        """
        if not response:
            return CleaningResult(
                cleaned="",
                original_length=0,
                cleaned_length=0,
            )
        
        return self._pipeline.clean_with_details(response)
    
    def clean_streaming_token(
        self,
        token: str,
        state: dict
    ) -> Tuple[str, dict]:
        """
        Clean a single streaming token.
        
        Maintains state to track tag boundaries across tokens.
        
        Args:
            token: Current token
            state: Mutable state dict with keys:
                   - 'in_think': bool
                   - 'buffer': str
        
        Returns:
            (output_token, updated_state)
        """
        return self._streaming_cleaner.clean_token(token, state)


# Global instance (backward compatible)
_cleaner = EnhancedResponseCleaner()


def clean_response(response: str) -> str:
    """Clean response - main entry point."""
    return _cleaner.clean(response)


def clean_response_detailed(response: str) -> CleaningResult:
    """Clean response with detailed results."""
    return _cleaner.clean_with_details(response)


def clean_streaming_token(token: str, state: dict) -> Tuple[str, dict]:
    """Clean single streaming token."""
    return _cleaner.clean_streaming_token(token, state)


# Re-export CleaningResult for backward compatibility
__all__ = [
    "EnhancedResponseCleaner",
    "CleaningResult",
    "clean_response",
    "clean_response_detailed",
    "clean_streaming_token",
]
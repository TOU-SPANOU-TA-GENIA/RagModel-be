# app/llm/response_cleaners/__init__.py
"""
Response cleaning system - config-driven cleaning pipeline.

Re-exports all cleaning components for backward compatibility.
"""

from app.llm.response_cleaners.base import (
    CleaningResult,
    ResponseCleanerBase
)
from app.llm.response_cleaners.pipeline import (
    CleaningPipeline,
    create_default_pipeline
)
from app.llm.response_cleaners.thinking import ThinkingCleaner
from app.llm.response_cleaners.tags import TagCleaner
from app.llm.response_cleaners.meta import MetaCommentaryCleaner
from app.llm.response_cleaners.whitespace import WhitespaceCleaner
from app.llm.response_cleaners.streaming import StreamingCleaner

__all__ = [
    "CleaningResult",
    "ResponseCleanerBase",
    "CleaningPipeline",
    "create_default_pipeline",
    "ThinkingCleaner",
    "TagCleaner",
    "MetaCommentaryCleaner",
    "WhitespaceCleaner",
    "StreamingCleaner",
    "clean_response",
    "clean_response_detailed",
    "clean_streaming_token",
]


# Convenience functions (backward compatible)
_pipeline = None


def _get_pipeline() -> CleaningPipeline:
    """Get or create default pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = create_default_pipeline()
    return _pipeline


def clean_response(response: str) -> str:
    """Clean response - main entry point."""
    return _get_pipeline().clean(response)


def clean_response_detailed(response: str) -> CleaningResult:
    """Clean response with detailed results."""
    return _get_pipeline().clean_with_details(response)


def clean_streaming_token(token: str, state: dict) -> tuple:
    """Clean single streaming token."""
    from app.llm.response_cleaners.streaming import StreamingCleaner
    cleaner = StreamingCleaner()
    return cleaner.clean_token(token, state)
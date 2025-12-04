# app/llm/response_cleaners/pipeline.py
"""
Cleaning pipeline - orchestrates multiple cleaners.
"""

from typing import List, Optional

from app.llm.response_cleaners.base import CleaningResult, ResponseCleanerBase
from app.llm.response_cleaners.thinking import ThinkingCleaner
from app.llm.response_cleaners.tags import TagCleaner
from app.llm.response_cleaners.meta import MetaCommentaryCleaner
from app.llm.response_cleaners.whitespace import WhitespaceCleaner
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class CleaningPipeline:
    """
    Orchestrates multiple cleaners in sequence.
    
    Order matters - cleaners are applied in registration order.
    Each cleaner can be enabled/disabled via config.
    """
    
    def __init__(self, cleaners: Optional[List[ResponseCleanerBase]] = None):
        self._cleaners: List[ResponseCleanerBase] = cleaners or []
    
    def add_cleaner(self, cleaner: ResponseCleanerBase):
        """Add a cleaner to the pipeline."""
        self._cleaners.append(cleaner)
    
    def clean(self, text: str) -> str:
        """Run all enabled cleaners and return cleaned text."""
        result = self.clean_with_details(text)
        return result.cleaned
    
    def clean_with_details(self, text: str) -> CleaningResult:
        """Run all enabled cleaners and return detailed result."""
        if not text:
            return CleaningResult(
                cleaned="",
                original_length=0,
                cleaned_length=0,
            )
        
        original_length = len(text)
        result = text
        phases_applied = []
        removed_thinking = False
        removed_tags = 0
        removed_meta = 0
        
        for cleaner in self._cleaners:
            if not cleaner.is_enabled():
                continue
            
            before = result
            before_len = len(result)
            
            result = cleaner.clean(result)
            
            if result != before:
                phases_applied.append(cleaner.name)
                
                # Track specific metrics
                if cleaner.name == "thinking_cleaner":
                    removed_thinking = True
                elif cleaner.name == "tag_cleaner":
                    removed_tags += 1
                elif cleaner.name == "meta_cleaner":
                    removed_meta += 1
        
        return CleaningResult(
            cleaned=result,
            original_length=original_length,
            cleaned_length=len(result),
            removed_thinking=removed_thinking,
            removed_tags=removed_tags,
            removed_meta=removed_meta,
            phases_applied=phases_applied,
        )


def create_default_pipeline() -> CleaningPipeline:
    """
    Create the default cleaning pipeline.
    
    Order:
    1. Thinking blocks (must be first)
    2. Response tag extraction
    3. Tag removal
    4. Meta-commentary
    5. Whitespace normalization (must be last)
    """
    pipeline = CleaningPipeline()
    
    # Phase 1: Remove thinking blocks
    pipeline.add_cleaner(ThinkingCleaner())
    
    # Phase 2: Extract from response tags (handled by TagCleaner)
    tag_cleaner = TagCleaner()
    pipeline.add_cleaner(tag_cleaner)
    
    # Phase 3: Remove meta-commentary
    pipeline.add_cleaner(MetaCommentaryCleaner())
    
    # Phase 4: Normalize whitespace (always last)
    pipeline.add_cleaner(WhitespaceCleaner())
    
    logger.debug(f"Created cleaning pipeline with {len(pipeline._cleaners)} cleaners")
    return pipeline


def create_minimal_pipeline() -> CleaningPipeline:
    """Create minimal pipeline for performance-critical paths."""
    pipeline = CleaningPipeline()
    pipeline.add_cleaner(ThinkingCleaner())
    pipeline.add_cleaner(WhitespaceCleaner())
    return pipeline
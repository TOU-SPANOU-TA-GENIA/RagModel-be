# app/llm/response_cleaners/thinking.py
"""
Thinking block cleaner - removes <think> blocks based on model config.
"""

import re
from typing import List, Tuple, Optional

from app.llm.response_cleaners.base import ResponseCleanerBase
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ThinkingCleaner(ResponseCleanerBase):
    """
    Removes thinking blocks from responses.
    
    Patterns are loaded from model registry config, with fallbacks.
    """
    
    def __init__(self):
        self._patterns: List[Tuple[re.Pattern, str]] = []
        self._load_patterns()
    
    @property
    def name(self) -> str:
        return "thinking_cleaner"
    
    def _load_patterns(self):
        """Load thinking patterns from model config."""
        try:
            from app.llm.model_registry import get_active_model
            model = get_active_model()
            
            if model and model.thinking_tags:
                pattern = model.get_thinking_pattern()
                if pattern:
                    self._patterns.append((pattern, "model_config"))
                    logger.debug(f"Loaded thinking pattern from model config")
        except Exception as e:
            logger.debug(f"Could not load model config: {e}")
        
        # Always add fallback patterns
        self._add_fallback_patterns()
    
    def _add_fallback_patterns(self):
        """Add fallback patterns for common thinking formats."""
        fallback_patterns = [
            # 1. Specific Time-based Thinking (e.g., "Σκέφτηκα για 32 δευτερόλεπτα")
            # Matches the timing line + English text, stopping at a Greek letter or Bold marker.
            # This handles the "Ρώτα..." case where there is no header, just Greek text.
            (r'Σκέφτηκα για \d+ δευτερόλεπτα.*?(?=\n\s*(?:[Α-Ωα-ω]|\*\*))', re.DOTALL),
            
            # 2. Structural Anchor filtering (The General Approach)
            # Removed '^' to catch thinking blocks even if they aren't at the absolute start
            (r'.*?(?=\n\s*\*\*ΑΠΑΝΤΗΣΗ:\*\*)', re.DOTALL),
            (r'.*?(?=\n\s*\*\*ANSWER:\*\*)', re.DOTALL),
            (r'.*?(?=\n\s*\*\*Found Name:\*\*)', re.DOTALL),
            (r'.*?(?=\n\s*\*\*Decision:\*\*)', re.DOTALL),

            # 3. Tag-based filtering (Explicit boundaries)
            (r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
            (r'<thinking>.*?</thinking>', re.DOTALL | re.IGNORECASE),
            (r'<σκέψη>.*?</σκέψη>', re.DOTALL | re.IGNORECASE),
            (r'<internal_analysis>.*?</internal_analysis>', re.DOTALL | re.IGNORECASE),
            (r'<response_guidance>.*?</response_guidance>', re.DOTALL | re.IGNORECASE),
            
            # 4. Header-based filtering
            (r'^Thinking:.*?(?=\n\n[Α-Ωα-ωA-Z]|\n\nResponse:|\Z)', re.DOTALL | re.MULTILINE),
            (r'^Analysis:.*?(?=\n\n[Α-Ωα-ωA-Z]|\n\nResponse:|\Z)', re.DOTALL | re.MULTILINE),
            (r'^Σκέψη:.*?(?=\n\n|\Z)', re.DOTALL | re.MULTILINE),
            (r'^Ανάλυση:.*?(?=\n\n|\Z)', re.DOTALL | re.MULTILINE),
        ]
        
        for pattern_str, flags in fallback_patterns:
            try:
                compiled = re.compile(pattern_str, flags)
                self._patterns.append((compiled, "fallback"))
            except re.error as e:
                logger.warning(f"Invalid pattern {pattern_str}: {e}")
    
    def clean(self, text: str) -> str:
        """Remove all thinking blocks from text."""
        if not text:
            return text
        
        result = text
        for pattern, source in self._patterns:
            before_len = len(result)
            result = pattern.sub('', result)
            if len(result) < before_len:
                logger.debug(f"Removed thinking block using {source} pattern")
        
        # Clean up leading whitespace that might remain after removal
        return result.lstrip()
    
    def is_enabled(self) -> bool:
        """Check if thinking cleaning is enabled."""
        try:
            from app.config import RESPONSE
            return getattr(RESPONSE, 'clean_thinking_tags', True)
        except:
            return True
    
    def extract_thinking(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Extract thinking content separately from response.
        
        Returns:
            (cleaned_text, thinking_content or None)
        """
        if not text:
            return text, None
        
        thinking_content = []
        result = text
        
        for pattern, _ in self._patterns:
            matches = pattern.findall(result)
            if matches:
                thinking_content.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
                result = pattern.sub('', result)
        
        thinking = "\n".join(str(t) for t in thinking_content) if thinking_content else None
        return result.lstrip(), thinking
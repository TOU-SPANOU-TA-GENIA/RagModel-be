# app/llm/response_cleaners/tags.py
"""
Tag cleaner - removes XML/HTML style tags from responses.
"""

import re
from typing import List

from app.llm.response_cleaners.base import ResponseCleanerBase
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class TagCleaner(ResponseCleanerBase):
    """
    Removes various XML/HTML tags from responses.
    
    Handles:
    - System/assistant/user tags
    - Standalone tag remnants
    - Response wrapper tags
    """
    
    def __init__(self):
        self._patterns: List[re.Pattern] = []
        self._load_patterns()
    
    @property
    def name(self) -> str:
        return "tag_cleaner"
    
    def _load_patterns(self):
        """Load tag patterns."""
        tag_patterns = [
            # Common LLM tags (remove entirely)
            r'</?(?:s|assistant|system|user|end)>',
            r'<\|(?:system|user|assistant|end)\|>',
            r'</?\w+_(?:context|guidance|rules|instruction)>',
            
            # Standalone remnants
            r'</?think>',
            r'</?thinking>',
            r'</?response>',
            r'</?σκέψη>',
            r'</?απάντηση>',
            
            # Special tokens that leak through
            r'<\|im_start\|>.*?(?=<\|im_end\|>|$)',
            r'<\|im_end\|>',
            r'<\|endoftext\|>',
        ]
        
        for pattern_str in tag_patterns:
            try:
                self._patterns.append(re.compile(pattern_str, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid tag pattern {pattern_str}: {e}")
    
    def clean(self, text: str) -> str:
        """Remove tags from text."""
        if not text:
            return text
        
        result = text
        for pattern in self._patterns:
            result = pattern.sub('', result)
        
        return result
    
    def is_enabled(self) -> bool:
        """Check if tag cleaning is enabled."""
        try:
            from app.config import RESPONSE
            return getattr(RESPONSE, 'clean_xml_tags', True)
        except:
            return True
    
    def extract_from_response_tags(self, text: str) -> str:
        """
        Extract content from <response> tags if present.
        
        If text contains <response>...</response>, returns only inner content.
        Otherwise returns original text.
        """
        response_pattern = re.compile(
            r'<response>(.*?)</response>',
            re.DOTALL | re.IGNORECASE
        )
        
        match = response_pattern.search(text)
        if match:
            return match.group(1).strip()
        
        return text
# app/llm/response_cleaners/whitespace.py
"""
Whitespace cleaner - normalizes spacing and formatting.
"""

import re

from app.llm.response_cleaners.base import ResponseCleanerBase
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class WhitespaceCleaner(ResponseCleanerBase):
    """
    Normalizes whitespace in responses.
    
    Handles:
    - Multiple spaces
    - Excessive newlines
    - Line start/end spaces
    - Punctuation spacing
    """
    
    @property
    def name(self) -> str:
        return "whitespace_cleaner"
    
    def clean(self, text: str) -> str:
        """Normalize whitespace in text."""
        if not text:
            return text
        
        result = text
        
        # Multiple spaces to single
        result = re.sub(r' +', ' ', result)
        
        # Multiple newlines to max two (paragraph break)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        # Spaces at line start/end
        lines = [line.strip() for line in result.split('\n')]
        result = '\n'.join(lines)
        
        # Fix spacing after punctuation
        result = re.sub(r'([.!?])\s{2,}', r'\1 ', result)
        
        # Final trim
        result = result.strip()
        
        return result
    
    def is_enabled(self) -> bool:
        """Check if whitespace cleaning is enabled."""
        try:
            from app.config import RESPONSE
            return getattr(RESPONSE, 'normalize_whitespace', True)
        except:
            return True
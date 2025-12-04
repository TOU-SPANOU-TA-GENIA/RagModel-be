# app/llm/response_cleaners/meta.py
"""
Meta-commentary cleaner - removes process narration and internal reasoning.
"""

import re
from typing import List

from app.llm.response_cleaners.base import ResponseCleanerBase
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetaCommentaryCleaner(ResponseCleanerBase):
    """
    Removes meta-commentary from responses.
    
    Handles:
    - Process narration ("Let me think...", "I'll analyze...")
    - Internal reasoning markers
    - Response structure narration
    - Instruction acknowledgments
    """
    
    def __init__(self):
        self._patterns: List[re.Pattern] = []
        self._prefix_patterns: List[re.Pattern] = []
        self._load_patterns()
    
    @property
    def name(self) -> str:
        return "meta_cleaner"
    
    def _load_patterns(self):
        """Load meta-commentary patterns."""
        # Main patterns (remove entire sentences)
        meta_patterns = [
            # English process narration
            r"(?:Let me|I'll|I will)\s+(?:think|analyze|consider|examine)[^.]*\.",
            r"(?:However|But|Therefore),?\s+(?:we cannot|I cannot|since we|unfortunately)[^.]*(?:evidence|information|data)[^.]*\.",
            
            # Greek process narration
            r"(?:Ας|Θα)\s+(?:σκεφτώ|αναλύσω|εξετάσω)[^.]*\.",
            
            # Internal reasoning leakage
            r'\(DO NOT include in response\)',
            r'\(Μην συμπεριλάβεις στην απάντηση\)',
            r'Internal analysis[^:]*:.*?(?=\n\n|\Z)',
            r'Guidelines:.*?(?=\n\n|\Z)',
            
            # Response structure narration
            r'This response follows.*?\.',
            r'Αυτή η απάντηση ακολουθεί.*?\.',
            
            # Instruction acknowledgment
            r'Following (?:the|your) instruction[s]?.*?\.',
            r'Ακολουθώντας τις οδηγίες.*?\.',
        ]
        
        for pattern_str in meta_patterns:
            try:
                self._patterns.append(
                    re.compile(pattern_str, re.IGNORECASE | re.DOTALL)
                )
            except re.error as e:
                logger.warning(f"Invalid meta pattern {pattern_str}: {e}")
        
        # Prefix patterns (remove from start)
        prefix_patterns = [
            r'^Response:\s*',
            r'^Απάντηση:\s*',
            r'^Here(?:\'s| is) (?:my|the) response:\s*',
            r'^Η απάντησή μου:\s*',
            r'^Certainly[!.]?\s*',
            r'^Of course[!.]?\s*',
            r'^Sure[!.]?\s*',
        ]
        
        for pattern_str in prefix_patterns:
            try:
                self._prefix_patterns.append(
                    re.compile(pattern_str, re.IGNORECASE)
                )
            except re.error as e:
                logger.warning(f"Invalid prefix pattern {pattern_str}: {e}")
    
    def clean(self, text: str) -> str:
        """Remove meta-commentary from text."""
        if not text:
            return text
        
        result = text
        
        # Remove meta patterns
        for pattern in self._patterns:
            result = pattern.sub('', result)
        
        # Remove prefixes
        for pattern in self._prefix_patterns:
            result = pattern.sub('', result)
        
        return result
    
    def is_enabled(self) -> bool:
        """Check if meta cleaning is enabled."""
        try:
            from app.config import RESPONSE
            return getattr(RESPONSE, 'clean_meta_commentary', True)
        except:
            return True
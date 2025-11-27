# app/llm/response_cleaner.py
"""
Response cleaner - removes meta-instructions, tags, and thinking traces from LLM outputs.
Ensures users see clean, professional responses without implementation details.
"""

import re
from typing import List, Tuple
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ResponseCleaner:
    """
    Cleans LLM responses by removing:
    - XML/HTML tags
    - Meta-commentary about the response process
    - Internal instruction references
    - Thinking traces and system tokens
    """
    
    def __init__(self):
        # Patterns for meta-commentary (phrases that explain the reasoning)
        self.meta_commentary_patterns = [
            # Explanation of following principles
            r'[Tt]his (?:answer|response) (?:is|follows|adheres to|demonstrates)[^.]{0,150}\.',
            r'[Ff]ollowing the (?:principle|guideline|rule)[^.]{0,100}\.',
            r'[Aa]ccording to the (?:principles|guidelines|rules)[^.]{0,100}\.',
            
            # Internal process narration
            r'[Ii]n response to the active instruction[^.]{0,150}\.',
            r'[Tt]he (?:assistant|AI) (?:responds|acknowledges|demonstrates)[^.]{0,150}\.',
            r'[Ll]et me know if (?:this|that) meets the requirements[^.]{0,100}\.',
            r'[Ii]\'ll (?:make sure to|ensure that|try to)[^.]{0,100}\.',
            
            # Revision markers
            r'\(Note:.*?\)',
            r'Here is a revised version[^:]*:',
            
            # Mathematical formatting artifacts
            r'The final answer is:\s*\$?\\boxed\{[^}]+\}\$?',
            
            # Internal context references
            r'<current_context>.*?</current_context>',
            r'<assistant_preferences:.*?>',
            r'<response_structure>.*?</response_structure>',
            r'<response_rules>.*?</response_rules>',
        ]
        
        # Patterns for XML/HTML tags
        self.tag_patterns = [
            r'</?s>',
            r'</?assistant[^>]*>',
            r'</assistant_response>',
            r'<response_[^>]+>',
            r'<next_step>.*?(?:</next_step>|$)',  # NEW: Remove next_step tags
            r'<thinking>.*?(?:</thinking>|$)',     # NEW: Remove thinking tags
            r'<reasoning>.*?(?:</reasoning>|$)',   # NEW: Remove reasoning tags
            r'</[^>]+>',  # Catch-all for remaining closing tags
        ]
        
        self.reasoning_patterns = [
            r'\[REASONING:.*?\]',
            r'\[INTERNAL:.*?\]',
            r'\[DEBUG:.*?\]',
            r'\[THOUGHT:.*?\]',
        ]
        
        # Patterns for instruction references
        self.instruction_ref_patterns = [
            r'when user says ["\']rule_\d+["\']',
            r'active instruction ["\']rule_\d+["\']',
        ]
    
    def clean(self, response: str) -> str:
        """Clean the response by removing all meta-instructions and artifacts."""
        if not response:
            return response
        
        original_length = len(response)
        cleaned = response
        
        # Step 1: Remove meta-commentary
        for pattern in self.meta_commentary_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Step 2: Remove XML/HTML tags
        for pattern in self.tag_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Step 3: Remove reasoning markers (NEW)
        for pattern in self.reasoning_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Step 4: Remove instruction references
        for pattern in self.instruction_ref_patterns:
            cleaned = re.sub(pattern, 'your instruction', cleaned, flags=re.IGNORECASE)
        
        # Step 5: Clean up whitespace
        cleaned = self._normalize_whitespace(cleaned)
        
        # Step 6: Remove empty parentheses
        cleaned = re.sub(r'\(\s*\)', '', cleaned)
        
        # Step 7: Final cleanup
        cleaned = cleaned.strip()
        
        # Log if significant cleaning occurred
        if len(cleaned) < original_length * 0.8:
            logger.debug(f"Response cleaned: {original_length} -> {len(cleaned)} chars")
        
        return cleaned
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with maximum two (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove spaces at start/end of lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove multiple spaces after punctuation
        text = re.sub(r'([.!?])\s{2,}', r'\1 ', text)
        
        return text.strip()
    
    def clean_batch(self, responses: List[str]) -> List[str]:
        """Clean multiple responses."""
        return [self.clean(response) for response in responses]


# Global instance
response_cleaner = ResponseCleaner()


def clean_response(response: str) -> str:
    """
    Convenience function to clean a single response.
    
    Usage:
        from app.llm.response_cleaner import clean_response
        
        raw_response = llm.generate(prompt)
        clean = clean_response(raw_response)
    """
    return response_cleaner.clean(response)
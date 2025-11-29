# app/llm/response_cleaner.py
"""
Response cleaner - removes meta-instructions, tags, and thinking traces from LLM outputs.
Ensures users see clean, professional responses without implementation details.
"""

import re
from typing import List
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
            
            # Thinking-related meta-commentary
            r'Internal analysis[^:]*:.*?(?=\n\n|\Z)',
            r'\(DO NOT include in response\)',
            r'Guidelines:.*?(?=\n\n|\Z)',
        ]
        
        # Patterns for XML/HTML tags
        self.tag_patterns = [
            r'</?s>',
            r'</?assistant[^>]*>',
            r'</assistant_response>',
            r'<response_[^>]+>',
            r'<next_step>.*?(?:</next_step>|$)',
            r'<reasoning>.*?(?:</reasoning>|$)',
            r'</[^>]+>',  # Catch-all for remaining closing tags
            
            # Thinking-related tags (content already parsed out, just clean stray tags)
            r'</?thinking[^>]*>',
            r'</?response[^>]*>',
            r'</?internal_analysis[^>]*>',
            r'</?response_guidance[^>]*>',
            r'</?internal_guidance[^>]*>',
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
        
        # Step 3: Remove reasoning markers
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


class EnhancedResponseCleaner:
    """
    Enhanced response cleaner with two-phase cleaning:
    1. Extract response from thinking/response structure
    2. Clean any remaining artifacts
    
    This catches "Thinking:" prefix patterns that the base cleaner misses.
    """
    
    def __init__(self):
        self._base_cleaner = ResponseCleaner()
        
        # Patterns that indicate thinking content (to be removed entirely)
        self.thinking_block_patterns = [
            # Explicit thinking blocks
            (r'<thinking>.*?</thinking>', re.DOTALL | re.IGNORECASE),
            (r'<internal_analysis>.*?</internal_analysis>', re.DOTALL | re.IGNORECASE),
            (r'<response_guidance>.*?</response_guidance>', re.DOTALL | re.IGNORECASE),
            
            # "Thinking:" style blocks (everything from Thinking: to double newline or Response:)
            (r'Thinking:.*?(?=Response:|$|\n\n[A-Z])', re.DOTALL | re.IGNORECASE),
            
            # Analysis blocks at start
            (r'^(?:Analysis|Internal analysis|Reasoning):.*?(?=\n\n|\nResponse:)', re.DOTALL | re.IGNORECASE | re.MULTILINE),
        ]
        
        # Patterns for stray thinking artifacts
        self.artifact_patterns = [
            r'^Thinking:\s*',
            r'^Response:\s*',
            r'^Analysis:\s*',
            r'\(DO NOT include in response\)',
            r'Guidelines:.*?(?=\n\n|$)',
            r'Internal analysis[^:]*:',
            
            # Reasoning about process
            r'(?:However|But|Therefore),?\s+(?:we cannot|I cannot|since we|unfortunately)[^.]*(?:evidence|information|data)[^.]*\.',
            
            # Let me/I will style thinking
            r"(?:Let me|I'll|I will)\s+(?:analyze|think|consider|examine)[^.]*\.",
        ]
    
    def clean(self, response: str) -> str:
        """Clean response with enhanced two-phase approach."""
        if not response:
            return response
        
        cleaned = response
        
        # Phase 1: Remove thinking blocks
        for pattern, flags in self.thinking_block_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=flags)
        
        # Phase 2: Extract from <response> tags if present
        response_match = re.search(
            r'<response>(.*?)</response>',
            cleaned,
            re.DOTALL | re.IGNORECASE
        )
        if response_match:
            cleaned = response_match.group(1)
        
        # Phase 3: Clean artifacts
        for pattern in self.artifact_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        # Phase 4: Apply base cleaner
        cleaned = self._base_cleaner.clean(cleaned)
        
        # Phase 5: Final whitespace normalization
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned


# Global instances
_base_cleaner = ResponseCleaner()
_enhanced_cleaner = EnhancedResponseCleaner()


def clean_response(response: str) -> str:
    """
    Convenience function to clean a single response.
    Uses enhanced cleaner for better thinking removal.
    
    Usage:
        from app.llm.response_cleaner import clean_response
        
        raw_response = llm.generate(prompt)
        clean = clean_response(raw_response)
    """
    return _enhanced_cleaner.clean(response)


def clean_response_basic(response: str) -> str:
    """
    Basic cleaning without thinking extraction.
    Use this if you've already extracted the response from thinking tags.
    """
    return _base_cleaner.clean(response)
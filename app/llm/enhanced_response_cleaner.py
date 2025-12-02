# app/llm/enhanced_response_cleaner.py
"""
Enhanced Response Cleaner - Comprehensive cleaning of LLM outputs.

Removes:
- <think> blocks and their content
- Qwen-style thinking patterns
- Internal reasoning artifacts
- System instruction leakage
- XML/HTML tags
- Meta-commentary

Preserves:
- Actual user-facing content
- Code blocks (with technical content)
- Proper formatting
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CleaningResult:
    """Result of response cleaning."""
    cleaned: str
    removed_thinking: bool
    removed_tags: int
    original_length: int
    cleaned_length: int


class EnhancedResponseCleaner:
    """
    Multi-phase response cleaner for production quality.
    """
    
    def __init__(self):
        # Phase 1: Thinking block patterns (remove entire blocks)
        self.thinking_blocks = [
            # Explicit think tags
            (r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
            (r'<thinking>.*?</thinking>', re.DOTALL | re.IGNORECASE),
            (r'<internal_analysis>.*?</internal_analysis>', re.DOTALL | re.IGNORECASE),
            (r'<response_guidance>.*?</response_guidance>', re.DOTALL | re.IGNORECASE),
            (r'<σκέψη>.*?</σκέψη>', re.DOTALL | re.IGNORECASE),
            
            # Thinking: style blocks (to next double newline or end)
            (r'^Thinking:.*?(?=\n\n[Α-Ωα-ωA-Z]|\n\nResponse:|\Z)', re.DOTALL | re.MULTILINE),
            (r'^Analysis:.*?(?=\n\n[Α-Ωα-ωA-Z]|\n\nResponse:|\Z)', re.DOTALL | re.MULTILINE),
            (r'^Σκέψη:.*?(?=\n\n|\Z)', re.DOTALL | re.MULTILINE),
            (r'^Ανάλυση:.*?(?=\n\n|\Z)', re.DOTALL | re.MULTILINE),
        ]
        
        # Phase 2: Tag patterns (remove tags, keep content if appropriate)
        self.tag_patterns = [
            # Tags to remove entirely (including content)
            r'</?(?:s|assistant|system|user|end)>',
            r'<\|(?:system|user|assistant|end)\|>',
            r'</?\w+_(?:context|guidance|rules|instruction)>',
            
            # Standalone tag remnants
            r'</?think>',
            r'</?thinking>',
            r'</?response>',
            r'</?σκέψη>',
            r'</?απάντηση>',
        ]
        
        # Phase 3: Meta-commentary patterns (remove entire sentences)
        self.meta_patterns = [
            # Process narration
            r"(?:Let me|I'll|I will)\s+(?:think|analyze|consider|examine)[^.]*\.",
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
        
        # Phase 4: Prefix/suffix cleanup
        self.prefix_patterns = [
            r'^Response:\s*',
            r'^Απάντηση:\s*',
            r'^Here(?:\'s| is) (?:my|the) response:\s*',
            r'^Η απάντησή μου:\s*',
        ]
    
    def clean(self, response: str) -> str:
        """
        Clean response with all phases.
        Returns clean, user-ready text.
        """
        if not response:
            return response
        
        result = self.clean_with_details(response)
        return result.cleaned
    
    def clean_with_details(self, response: str) -> CleaningResult:
        """
        Clean response and return detailed results.
        Useful for debugging and logging.
        """
        original_length = len(response)
        removed_thinking = False
        removed_tags = 0
        
        cleaned = response
        
        # Phase 1: Remove thinking blocks
        for pattern, flags in self.thinking_blocks:
            before = cleaned
            cleaned = re.sub(pattern, '', cleaned, flags=flags)
            if cleaned != before:
                removed_thinking = True
        
        # Phase 2: Remove tags
        for pattern in self.tag_patterns:
            before = cleaned
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            if cleaned != before:
                removed_tags += 1
        
        # Phase 3: Remove meta-commentary
        for pattern in self.meta_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Phase 4: Remove prefixes
        for pattern in self.prefix_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Phase 5: Normalize whitespace
        cleaned = self._normalize_whitespace(cleaned)
        
        # Phase 6: Final cleanup
        cleaned = cleaned.strip()
        
        return CleaningResult(
            cleaned=cleaned,
            removed_thinking=removed_thinking,
            removed_tags=removed_tags,
            original_length=original_length,
            cleaned_length=len(cleaned)
        )
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        # Multiple spaces to single
        text = re.sub(r' +', ' ', text)
        
        # Multiple newlines to max two
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Spaces at line start/end
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Fix spacing after punctuation
        text = re.sub(r'([.!?])\s{2,}', r'\1 ', text)
        
        return text
    
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
                   - 'tag_buffer': str
        
        Returns:
            (output_token, updated_state)
        """
        buffer = state.get('buffer', '') + token
        in_think = state.get('in_think', False)
        
        # Check for think block start
        if '<think>' in buffer.lower() and not in_think:
            parts = buffer.lower().split('<think>', 1)
            # Find actual split point in original
            idx = buffer.lower().find('<think>')
            output = buffer[:idx]
            state['in_think'] = True
            state['buffer'] = buffer[idx + 7:]  # After <think>
            return output, state
        
        # Check for think block end
        if '</think>' in buffer.lower() and in_think:
            idx = buffer.lower().find('</think>')
            state['in_think'] = False
            state['buffer'] = buffer[idx + 8:]  # After </think>
            # Return content after think block
            remaining = state['buffer']
            state['buffer'] = ''
            return remaining, state
        
        # In think mode - suppress output
        if in_think:
            state['buffer'] = buffer
            return '', state
        
        # Normal mode - output token
        state['buffer'] = ''
        return token, state


# Global instance
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
# app/llm/streaming/token_filter.py
"""
Token filter for streaming generation.
Handles early stopping detection and trash token filtering.
"""

import re
from typing import Tuple, Optional, Set
from dataclasses import dataclass, field

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TokenFilterConfig:
    """Configuration for token filtering."""
    # Tokens/patterns that indicate end of meaningful generation
    stop_patterns: Set[str] = field(default_factory=lambda: {
        '</s>', '<|im_end|>', '<|endoftext|>', '<|end|>',
        '[EOS]', '[PAD]', '<pad>', '<eos>',
    })
    
    # Patterns that are trash when repeated
    repetition_patterns: Set[str] = field(default_factory=lambda: {
        '</s>', '<s>', '<pad>', '<unk>',
    })
    
    # Max consecutive repetitions before stopping
    max_repetitions: int = 2
    
    # Language marker patterns to filter
    language_markers: Set[str] = field(default_factory=lambda: {
        '/zh', '/en', '/el', '/no_think', '/think', '//', '/',
    })


class TokenFilter:
    """
    Filters streaming tokens to detect early stopping and remove trash.
    
    This solves the problem of models generating padding tokens when
    the natural response is shorter than max_new_tokens.
    """
    
    def __init__(self, config: Optional[TokenFilterConfig] = None):
        self.config = config or TokenFilterConfig()
        self._reset_state()
    
    def _reset_state(self):
        """Reset filter state for new generation."""
        self._last_token: str = ""
        self._repetition_count: int = 0
        self._should_stop: bool = False
        self._meaningful_content_seen: bool = False
        self._buffer: str = ""
    
    def reset(self):
        """Public method to reset state."""
        self._reset_state()
    
    def process_token(self, token: str) -> Tuple[Optional[str], bool]:
        """
        Process a token and determine if it should be yielded.
        
        Args:
            token: The raw token from the model
            
        Returns:
            Tuple of (filtered_token or None, should_stop)
            - filtered_token: The token to yield, or None if should skip
            - should_stop: True if generation should stop
        """
        if self._should_stop:
            return None, True
        
        # Check for stop patterns (end of sequence markers)
        for pattern in self.config.stop_patterns:
            if pattern in token:
                # If we've seen meaningful content, this is a legitimate stop
                if self._meaningful_content_seen:
                    self._should_stop = True
                    # Remove the stop pattern from token if there's other content
                    cleaned = token.replace(pattern, '').strip()
                    return (cleaned if cleaned else None), True
                # Haven't seen meaningful content yet - might be artifact
                return None, False
        
        # Check for repetition of trash patterns
        token_stripped = token.strip()
        
        if token_stripped in self.config.repetition_patterns:
            self._repetition_count += 1
            if self._repetition_count >= self.config.max_repetitions:
                self._should_stop = True
                return None, True
            return None, False
        
        # Check for language markers
        if token_stripped in self.config.language_markers:
            return None, False
        
        # Filter Chinese characters (Qwen's default thinking language)
        chinese_ratio = self._calculate_chinese_ratio(token)
        if chinese_ratio > 0.5:
            return None, False
        
        # Reset repetition counter for meaningful content
        self._repetition_count = 0
        
        # Mark that we've seen meaningful content
        if token_stripped:
            self._meaningful_content_seen = True
        
        # Check if token is effectively empty
        if not token or not token.strip():
            return None, False
        
        self._last_token = token
        return token, False
    
    def _calculate_chinese_ratio(self, text: str) -> float:
        """Calculate the ratio of Chinese characters in text."""
        if not text:
            return 0.0
        chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return chinese_count / len(text)
    
    @property
    def should_stop(self) -> bool:
        """Check if generation should stop."""
        return self._should_stop


class StreamingTokenProcessor:
    """
    Higher-level processor that combines token filtering with response cleaning.
    """
    
    def __init__(self):
        self._filter = TokenFilter()
        self._in_thinking = False
        self._thinking_content = []
        self._response_content = []
    
    def reset(self):
        """Reset for new generation."""
        self._filter.reset()
        self._in_thinking = False
        self._thinking_content = []
        self._response_content = []
    
    def process(self, token: str) -> Tuple[Optional[str], str, bool]:
        """
        Process a token with full context awareness.
        
        Args:
            token: Raw token from model
            
        Returns:
            Tuple of (display_token, context, should_stop)
            - display_token: Token to show (None to skip)
            - context: 'thinking', 'response', or 'meta'
            - should_stop: Whether generation should stop
        """
        # Detect thinking block transitions
        if '<think>' in token or '<think' in token.strip():
            self._in_thinking = True
            cleaned = token.replace('<think>', '').replace('<think', '')
            return None, 'thinking_start', False
        
        if '</think>' in token or '</think' in token.strip():
            self._in_thinking = False
            cleaned = token.replace('</think>', '').replace('</think', '')
            return (cleaned if cleaned.strip() else None), 'thinking_end', False
        
        # Apply token filtering
        filtered_token, should_stop = self._filter.process_token(token)
        
        if self._in_thinking:
            if filtered_token:
                self._thinking_content.append(filtered_token)
            return filtered_token, 'thinking', should_stop
        else:
            if filtered_token:
                self._response_content.append(filtered_token)
            return filtered_token, 'response', should_stop
    
    @property
    def full_thinking(self) -> str:
        """Get accumulated thinking content."""
        return ''.join(self._thinking_content)
    
    @property
    def full_response(self) -> str:
        """Get accumulated response content."""
        return ''.join(self._response_content)


def create_token_filter() -> TokenFilter:
    """Factory function for token filter."""
    return TokenFilter()


def create_streaming_processor() -> StreamingTokenProcessor:
    """Factory function for streaming processor."""
    return StreamingTokenProcessor()
# app/llm/streaming/thinking_filter.py
"""
Thinking filter for streaming - detects and filters thinking blocks in real-time.
"""

from typing import Generator, Tuple, Optional, List
from dataclasses import dataclass, field

from app.llm.streaming.events import StreamEvent, StreamEventType
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ThinkingFilterState:
    """State for tracking thinking blocks during streaming."""
    in_thinking: bool = False
    buffer: str = ""
    thinking_buffer: str = ""
    response_started: bool = False
    
    # Tags loaded from config
    start_tags: List[str] = field(default_factory=list)
    end_tags: List[str] = field(default_factory=list)


class ThinkingFilter:
    """
    Filters thinking content from token stream.
    
    Uses model config for thinking tag detection.
    Can optionally pass through thinking content for debugging.
    """
    
    def __init__(self):
        self._start_tags: List[str] = []
        self._end_tags: List[str] = []
        self._load_from_config()
    
    def _load_from_config(self):
        """Load thinking tags from model config."""
        # Defaults
        self._start_tags = ['<think>', '<thinking>', '<σκέψη>']
        self._end_tags = ['</think>', '</thinking>', '</σκέψη>']
        
        try:
            from app.llm.model_registry import get_active_model
            model = get_active_model()
            
            if model and model.thinking_tags:
                start = model.thinking_start_tags
                end = model.thinking_end_tags
                if start:
                    self._start_tags = start
                if end:
                    self._end_tags = end
                    
            logger.debug(f"Loaded thinking tags: start={self._start_tags}, end={self._end_tags}")
        except Exception as e:
            logger.debug(f"Using default thinking tags: {e}")
    
    def create_state(self) -> ThinkingFilterState:
        """Create initial filter state."""
        return ThinkingFilterState(
            start_tags=self._start_tags.copy(),
            end_tags=self._end_tags.copy()
        )
    
    def filter_token(
        self,
        token: str,
        state: ThinkingFilterState,
        include_thinking: bool = False
    ) -> Generator[StreamEvent, None, None]:
        """
        Filter a single token through thinking detection.
        
        Args:
            token: The token to process
            state: Current filter state (modified in place)
            include_thinking: If True, yield thinking tokens too
            
        Yields:
            StreamEvent objects
        """
        state.buffer += token
        
        # Check for thinking start
        if not state.in_thinking:
            start_tag, idx = self._find_tag(state.buffer, state.start_tags)
            if start_tag:
                # Yield any content before the tag
                if idx > 0:
                    if not state.response_started:
                        state.response_started = True
                        yield StreamEvent.response_start()
                    yield StreamEvent.token(state.buffer[:idx])
                
                # Enter thinking mode
                state.in_thinking = True
                state.buffer = state.buffer[idx + len(start_tag):]
                state.thinking_buffer = ""
                
                if include_thinking:
                    yield StreamEvent.thinking_start()
                return
        
        # Check for thinking end
        if state.in_thinking:
            end_tag, idx = self._find_tag(state.buffer, state.end_tags)
            if end_tag:
                # Capture thinking content
                state.thinking_buffer += state.buffer[:idx]
                
                if include_thinking:
                    yield StreamEvent.token(state.buffer[:idx])
                    yield StreamEvent.thinking_end()
                
                # Exit thinking mode
                state.in_thinking = False
                state.buffer = state.buffer[idx + len(end_tag):]
                
                # Yield response start if we have content after
                if state.buffer.strip() and not state.response_started:
                    state.response_started = True
                    yield StreamEvent.response_start()
                
                # Yield any remaining buffer
                if state.buffer:
                    yield StreamEvent.token(state.buffer)
                    state.buffer = ""
                return
            
            # Still in thinking - accumulate
            state.thinking_buffer += token
            if include_thinking:
                yield StreamEvent.token(token)
            state.buffer = self._keep_end_buffer(state.buffer)
            return
        
        # Normal mode - check for partial start tag
        partial = self._get_partial_tag(state.buffer, state.start_tags)
        if partial:
            output = state.buffer[:-len(partial)]
            state.buffer = partial
        else:
            output = state.buffer
            state.buffer = ""
        
        if output:
            if not state.response_started:
                state.response_started = True
                yield StreamEvent.response_start()
            yield StreamEvent.token(output)
    
    def flush(self, state: ThinkingFilterState) -> Generator[StreamEvent, None, None]:
        """Flush remaining buffer at end of stream."""
        if state.buffer and not state.in_thinking:
            if not state.response_started:
                yield StreamEvent.response_start()
            yield StreamEvent.token(state.buffer)
        
        yield StreamEvent.response_end()
        yield StreamEvent.done()
    
    def _find_tag(self, text: str, tags: List[str]) -> Tuple[Optional[str], int]:
        """Find first occurrence of any tag (case-insensitive)."""
        text_lower = text.lower()
        best_idx = len(text) + 1
        best_tag = None
        
        for tag in tags:
            idx = text_lower.find(tag.lower())
            if idx != -1 and idx < best_idx:
                best_idx = idx
                best_tag = tag
        
        return best_tag, best_idx if best_tag else -1
    
    def _get_partial_tag(self, text: str, tags: List[str]) -> str:
        """Check if text ends with start of any tag."""
        text_lower = text.lower()
        
        for tag in tags:
            tag_lower = tag.lower()
            for i in range(1, len(tag_lower)):
                if text_lower.endswith(tag_lower[:i]):
                    return text[-i:]
        return ""
    
    def _keep_end_buffer(self, buffer: str) -> str:
        """Keep only end of buffer for partial tag detection."""
        max_len = max(len(t) for t in self._end_tags) if self._end_tags else 10
        return buffer[-max_len:] if len(buffer) > max_len else buffer
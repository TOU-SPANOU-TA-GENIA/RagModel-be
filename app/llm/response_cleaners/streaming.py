# app/llm/response_cleaners/streaming.py
"""
Streaming cleaner - cleans tokens in real-time during streaming.
"""

from typing import Tuple, Dict, Any, Optional

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class StreamingCleaner:
    """
    Cleans tokens during streaming generation.
    
    Maintains state to track tag boundaries across tokens.
    Uses model config for thinking tag patterns.
    """
    
    def __init__(self):
        self._thinking_start_tags = ['<think>', '<thinking>', '<σκέψη>']
        self._thinking_end_tags = ['</think>', '</thinking>', '</σκέψη>']
        self._load_from_config()
    
    def _load_from_config(self):
        """Load thinking tags from model config."""
        try:
            from app.llm.model_registry import get_active_model
            model = get_active_model()
            
            if model and model.thinking_tags:
                start = model.thinking_start_tags
                end = model.thinking_end_tags
                if start:
                    self._thinking_start_tags = start
                if end:
                    self._thinking_end_tags = end
        except Exception as e:
            logger.debug(f"Using default thinking tags: {e}")
    
    def clean_token(
        self,
        token: str,
        state: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Clean a single streaming token.
        
        Args:
            token: Current token
            state: Mutable state dict with keys:
                   - 'in_think': bool - currently inside thinking block
                   - 'buffer': str - accumulated partial tag
                   
        Returns:
            (output_token, updated_state)
        """
        buffer = state.get('buffer', '') + token
        in_think = state.get('in_think', False)
        
        # Check for thinking block start
        start_tag, start_idx = self._find_tag(buffer, self._thinking_start_tags)
        if start_tag and not in_think:
            # Output text before the tag
            output = buffer[:start_idx]
            state['in_think'] = True
            state['buffer'] = buffer[start_idx + len(start_tag):]
            return output, state
        
        # Check for thinking block end
        end_tag, end_idx = self._find_tag(buffer, self._thinking_end_tags)
        if end_tag and in_think:
            state['in_think'] = False
            state['buffer'] = buffer[end_idx + len(end_tag):]
            # Return content after think block
            remaining = state['buffer']
            state['buffer'] = ''
            return remaining, state
        
        # In thinking mode - suppress output but keep buffering
        if in_think:
            # Keep partial tag detection working
            state['buffer'] = self._keep_partial_buffer(buffer, self._thinking_end_tags)
            return '', state
        
        # Normal mode - check for partial tag at end
        partial = self._get_partial_tag(buffer, self._thinking_start_tags)
        if partial:
            # Hold back potential partial tag
            output = buffer[:-len(partial)]
            state['buffer'] = partial
            return output, state
        
        # Normal output
        state['buffer'] = ''
        return token, state
    
    def _find_tag(
        self,
        text: str,
        tags: list
    ) -> Tuple[Optional[str], int]:
        """Find first occurrence of any tag (case-insensitive)."""
        text_lower = text.lower()
        best_idx = len(text)
        best_tag = None
        
        for tag in tags:
            idx = text_lower.find(tag.lower())
            if idx != -1 and idx < best_idx:
                best_idx = idx
                best_tag = tag
        
        return best_tag, best_idx if best_tag else -1
    
    def _get_partial_tag(self, text: str, tags: list) -> str:
        """Check if text ends with partial tag."""
        text_lower = text.lower()
        
        for tag in tags:
            tag_lower = tag.lower()
            # Check all possible partial lengths
            for i in range(1, len(tag_lower)):
                if text_lower.endswith(tag_lower[:i]):
                    return text[-i:]
        
        return ''
    
    def _keep_partial_buffer(self, buffer: str, tags: list) -> str:
        """Keep only what's needed for partial tag detection."""
        max_tag_len = max(len(t) for t in tags) if tags else 10
        return buffer[-max_tag_len:] if len(buffer) > max_tag_len else buffer
    
    def create_initial_state(self) -> Dict[str, Any]:
        """Create initial state for streaming."""
        return {
            'in_think': False,
            'buffer': '',
        }
from typing import List, Dict, Optional, Any  # <--- Added Any here
from app.core.interfaces import MemoryStore
from app.config import get_config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class InMemoryStore(MemoryStore):
    """Simple ephemeral memory for development/testing."""
    def __init__(self):
        self._store: Dict[str, List[Dict[str, str]]] = {}

    def load_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._store.get(session_id, [])

    def save_message(self, session_id: str, role: str, content: str):
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append({"role": role, "content": content})

    def clear_history(self, session_id: str):
        self._store.pop(session_id, None)

class ContextWindowHandler:
    """
    Manages the context window to ensure prompts don't exceed model limits.
    Configurable via 'memory' settings in config.json.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().get("memory", {})
        self.max_messages = self.config.get("max_history_messages", 10)
        self.max_tokens = self.config.get("max_context_tokens", 2000)

    def prune_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Trims history to fit constraints. 
        Basic implementation: simpler is better for now.
        """
        if not history:
            return []
            
        # 1. Limit by message count
        pruned = history[-self.max_messages:]
        
        # 2. (Optional) Limit by estimated tokens could go here.
        # For fluid-dynamic refactoring, we keep it simple unless specific 
        # tokenization logic is requested.
        
        return pruned
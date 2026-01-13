import json
from pathlib import Path
from typing import Dict, List, Optional
from threading import RLock

from app.chat.schemas import ChatSession
from app.config import get_config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ChatRepository:
    """
    Persists chat sessions. 
    Can be configured to save to disk via config.json.
    """
    
    def __init__(self):
        self._chats: Dict[str, ChatSession] = {}
        self._lock = RLock()
        self._storage_path: Optional[Path] = None
        self._initialize_storage()

    def _initialize_storage(self):
        """Load persistence settings from config."""
        config = get_config()
        # Fluid configuration: Check if we should persist to disk
        storage_config = config.get("storage", {})
        
        if storage_config.get("persist_chats", False):
            path_str = storage_config.get("chat_data_path", "data/chats.json")
            self._storage_path = Path(path_str)
            self._load_from_disk()

    def _load_from_disk(self):
        """Load chats from JSON file."""
        if self._storage_path and self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    with self._lock:
                        for chat_data in data.values():
                            session = ChatSession(**chat_data)
                            self._chats[session.id] = session
                logger.info(f"Loaded {len(self._chats)} chats from disk.")
            except Exception as e:
                logger.error(f"Failed to load chats from disk: {e}")

    def _save_to_disk(self):
        """Save chats to JSON file."""
        if self._storage_path:
            try:
                self._storage_path.parent.mkdir(parents=True, exist_ok=True)
                data = {kid: session.dict() for kid, session in self._chats.items()}
                # Use a primitive JSON dump; simpler than custom encoders for this snippet
                # In production, use a proper serialization strategy for datetimes
                with open(self._storage_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, default=str, indent=2)
            except Exception as e:
                logger.error(f"Failed to save chats to disk: {e}")

    def get(self, chat_id: str) -> Optional[ChatSession]:
        with self._lock:
            return self._chats.get(chat_id)

    def list_all(self) -> List[ChatSession]:
        with self._lock:
            return list(self._chats.values())

    def save(self, session: ChatSession):
        with self._lock:
            self._chats[session.id] = session
            self._save_to_disk()

    def delete(self, chat_id: str):
        with self._lock:
            if chat_id in self._chats:
                del self._chats[chat_id]
                self._save_to_disk()

    def clear(self):
        with self._lock:
            self._chats.clear()
            self._save_to_disk()
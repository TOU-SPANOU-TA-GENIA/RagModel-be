from typing import List, Optional
from datetime import datetime

from app.chat.schemas import ChatSession, ChatMessage, ChatSummary
from app.chat.storage import ChatRepository
from app.core.exceptions import AppError

class ChatNotFoundException(AppError):
    def __init__(self, chat_id: str):
        super().__init__(f"Chat session {chat_id} not found", code="CHAT_NOT_FOUND")

class ChatService:
    """
    High-level chat operations.
    Bridge between API and Storage.
    """
    
    def __init__(self):
        # In a dependency injection system, this would be injected.
        self.repository = ChatRepository()

    def create_chat(self, title: str = "New Chat") -> ChatSession:
        session = ChatSession(title=title)
        self.repository.save(session)
        return session

    def get_chat(self, chat_id: str) -> ChatSession:
        session = self.repository.get(chat_id)
        if not session:
            raise ChatNotFoundException(chat_id)
        return session

    def list_chats(self) -> List[ChatSummary]:
        sessions = self.repository.list_all()
        # Sort by updated_at desc
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        
        return [
            ChatSummary(
                id=s.id,
                title=s.title,
                updated_at=s.updated_at,
                message_count=len(s.messages)
            )
            for s in sessions
        ]

    def add_message(self, chat_id: str, role: str, content: str, meta: dict = None) -> ChatMessage:
        session = self.get_chat(chat_id)
        
        message = ChatMessage(role=role, content=content, metadata=meta or {})
        session.messages.append(message)
        session.updated_at = datetime.utcnow()
        
        self.repository.save(session)
        return message

    def update_title(self, chat_id: str, new_title: str):
        session = self.get_chat(chat_id)
        session.title = new_title
        session.updated_at = datetime.utcnow()
        self.repository.save(session)

    def delete_chat(self, chat_id: str):
        self.get_chat(chat_id) # Ensure exists
        self.repository.delete(chat_id)

    def clear_all(self):
        self.repository.clear()

# Global instance for easy import in API routes
chat_service = ChatService()
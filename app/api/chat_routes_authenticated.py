# app/api/chat_routes_authenticated.py
"""
Authenticated chat routes using hybrid storage.
Replaces the old in-memory chat manager with persistent storage.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

from app.api.auth_routes import get_current_user_dep
from app.db.storage import storage
from app.agent.integration import get_agent
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/chats", tags=["Chats"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateChatRequest(BaseModel):
    """Request to create a new chat."""
    title: Optional[str] = Field("Νέα Συνομιλία", max_length=200)


class SendMessageRequest(BaseModel):
    """Request to send a message in a chat."""
    content: str = Field(..., min_length=1, max_length=10000)


class MessageResponse(BaseModel):
    """A chat message."""
    id: int
    role: str
    content: str
    timestamp: str


class ChatSummary(BaseModel):
    """Summary of a chat (for list view)."""
    id: str
    title: str
    updated_at: str
    message_count: int


class ChatDetail(BaseModel):
    """Detailed chat information."""
    id: str
    title: str
    created_at: str
    updated_at: str


class ChatWithMessages(BaseModel):
    """Chat with full message history."""
    chat: ChatDetail
    messages: List[MessageResponse]


class AgentResponse(BaseModel):
    """Response from the AI agent."""
    answer: str
    message_id: int
    timestamp: str

class UpdateChatRequest(BaseModel):
    """Request to update chat properties."""
    title: Optional[str] = Field(None, max_length=200)
    
# =============================================================================
# Chat Management Endpoints
# =============================================================================
# Add this endpoint to chat_routes_authenticated.py

@router.patch("/{chat_id}")
async def update_chat(
    chat_id: str,
    request: UpdateChatRequest,
    user: dict = Depends(get_current_user_dep)
):
    """
    Update chat properties (title).
    
    **Headers:**
    - Authorization: Bearer <token>
    
    **Returns:**
    - Updated chat info
    """
    # Verify chat belongs to user
    chat = storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    if chat["user_id"] != user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this chat"
        )
    
    try:
        if request.title is not None:
            storage.update_chat_title(chat_id, request.title)
        
        updated_chat = storage.get_chat(chat_id)
        return {
            "id": updated_chat["id"],
            "title": updated_chat["title"],
            "updated_at": updated_chat["updated_at"]
        }
    
    except Exception as e:
        logger.error(f"Failed to update chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update chat"
        )

@router.post("/", response_model=ChatSummary, status_code=status.HTTP_201_CREATED)
async def create_chat(
    request: CreateChatRequest,
    user: dict = Depends(get_current_user_dep)
):
    """
    Create a new chat session.
    
    **Headers:**
    - Authorization: Bearer <token>
    
    **Returns:**
    - Chat summary with ID
    """
    try:
        chat_id = storage.create_chat(
            user_id=user["id"],
            title=request.title
        )
        
        chat = storage.get_chat(chat_id)
        
        return {
            "id": chat["id"],
            "title": chat["title"],
            "updated_at": chat["updated_at"],
            "message_count": 0
        }
    
    except Exception as e:
        logger.error(f"Failed to create chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat"
        )


@router.get("/", response_model=List[ChatSummary])
async def list_chats(user: dict = Depends(get_current_user_dep)):
    """
    Get all chats for the current user.
    
    **Headers:**
    - Authorization: Bearer <token>
    
    **Returns:**
    - List of chat summaries, sorted by last updated
    """
    try:
        chats = storage.get_user_chats(user["id"])
        return chats
    
    except Exception as e:
        logger.error(f"Failed to list chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chats"
        )


@router.get("/{chat_id}", response_model=ChatWithMessages)
async def get_chat(
    chat_id: str,
    user: dict = Depends(get_current_user_dep)
):
    """
    Get a specific chat with full message history.
    
    **Headers:**
    - Authorization: Bearer <token>
    
    **Returns:**
    - Chat details and all messages
    """
    try:
        chat = storage.get_chat(chat_id)
        
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        # Verify chat belongs to user
        if chat["user_id"] != user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        messages = storage.get_messages(chat_id)
        
        return {
            "chat": {
                "id": chat["id"],
                "title": chat["title"],
                "created_at": chat["created_at"],
                "updated_at": chat["updated_at"]
            },
            "messages": messages
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat"
        )


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: str,
    user: dict = Depends(get_current_user_dep)
):
    """
    Delete a chat and all its messages.
    
    **Headers:**
    - Authorization: Bearer <token>
    """
    try:
        chat = storage.get_chat(chat_id)
        
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        # Verify chat belongs to user
        if chat["user_id"] != user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        storage.delete_chat(chat_id)
        return None
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat"
        )


# =============================================================================
# Message Endpoints
# =============================================================================

@router.post("/{chat_id}/messages", response_model=AgentResponse)
async def send_message(
    chat_id: str,
    request: SendMessageRequest,
    user: dict = Depends(get_current_user_dep)
):
    """
    Send a message in a chat and get AI response.
    
    **Headers:**
    - Authorization: Bearer <token>
    
    **Request body:**
    - content: Message text
    
    **Returns:**
    - AI agent's response
    """
    try:
        chat = storage.get_chat(chat_id)
        
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        # Verify chat belongs to user
        if chat["user_id"] != user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Get chat history BEFORE adding new message
        # Get chat history BEFORE adding new message
        messages = storage.get_messages(chat_id, limit=10)
        chat_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

        # DEBUG: Log what we're passing
        logger.info(f"🔍 DEBUG: Retrieved {len(messages)} messages from database")
        logger.info(f"🔍 DEBUG: Chat history has {len(chat_history)} messages")
        for i, msg in enumerate(chat_history[:3]):  # Show first 3
            logger.info(f"🔍 DEBUG: Message {i+1}: {msg['role']}: {msg['content'][:50]}")

        # Save user message
        user_msg_id = storage.add_message(chat_id, "user", request.content)
                
        # Get AI response with full history
        agent = get_agent()
        response = agent.process_query(
            query=request.content,
            chat_history=chat_history,
            metadata={"session_id": chat_id}
        )
        
        # Save AI response
        assistant_msg_id = storage.add_message(
            chat_id, 
            "assistant", 
            response.answer
        )
        
        return {
            "answer": response.answer,
            "message_id": assistant_msg_id,
            "timestamp": storage.get_messages(chat_id, limit=1)[-1]["timestamp"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )


@router.get("/{chat_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    chat_id: str,
    limit: Optional[int] = None,
    user: dict = Depends(get_current_user_dep)
):
    """
    Get messages for a chat.
    
    **Headers:**
    - Authorization: Bearer <token>
    
    **Query params:**
    - limit: Maximum number of recent messages to return
    
    **Returns:**
    - List of messages
    """
    try:
        chat = storage.get_chat(chat_id)
        
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        # Verify chat belongs to user
        if chat["user_id"] != user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        messages = storage.get_messages(chat_id, limit=limit)
        return messages
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve messages"
        )
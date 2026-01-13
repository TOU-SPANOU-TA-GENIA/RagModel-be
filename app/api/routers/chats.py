from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import uuid

from app.db.session import get_db
from app.db.models import Chat, User, Message # Import Message
from app.api.schemas import ChatResponse, ChatCreate, MessageSchema # Import MessageSchema
from app.auth.service import get_current_user

router = APIRouter()

# ... (Keep get_chats and create_chat as they were) ...
@router.get("/", response_model=List[ChatResponse])
async def get_chats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return db.query(Chat).filter(Chat.user_id == current_user.id).all()

@router.post("/", response_model=ChatResponse)
async def create_chat(
    chat_in: ChatCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    new_chat = Chat(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        title=chat_in.title
    )
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return new_chat

@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    db.delete(chat)
    db.commit()
    return {"status": "success", "id": chat_id}

# --- ADD THIS NEW ENDPOINT ---
@router.get("/{chat_id}/messages", response_model=List[MessageSchema])
async def get_chat_messages(
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Fetch all messages for a specific chat.
    """
    # 1. Verify ownership
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == current_user.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # 2. Return messages (SQLAlchemy relationship handles the fetch)
    return chat.messages
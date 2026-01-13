from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.auth.schemas import User, UserInDB 

class UserBase(BaseModel):
    username: str
    full_name: Optional[str] = None
    email: Optional[str] = None

class UserResponse(UserBase):
    id: int
    disabled: bool
    
    class Config:
        from_attributes = True

# --- Auth Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str
    user: Optional[UserResponse] = None # <--- ADD THIS FIELD
    
class LoginRequest(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    
class MessageSchema(BaseModel):
    role: str
    content: str
    created_at: Optional[datetime] = None

class ChatCreate(BaseModel):
    title: Optional[str] = "New Chat"

class ChatResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    messages: List[MessageSchema] = []

    class Config:
        from_attributes = True

class FileInfo(BaseModel):
    filename: str
    path: str
    size: int
    modified_at: float

class FileUploadResponse(BaseModel):
    filename: str
    location: str
    message: str

class WorkflowDefinition(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    enabled: bool = True
    trigger: Optional[Dict[str, Any]] = None
    steps: List[Dict[str, Any]] = []

class HealthCheck(BaseModel):
    status: str
    version: str

class AgentRequest(BaseModel):
    query: str
    stream: bool = False

class AgentResponse(BaseModel):
    response: str
    thinking_process: Optional[str] = None
    sources: List[Dict[str, Any]] = []
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class NewChatRequest(BaseModel):
    title: Optional[str] = Field(default="Νέα Συνομιλία", max_length=200)


class MessageRequest(BaseModel):
    role: str = Field(default="user", pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class ChatSummary(BaseModel):
    id: str
    title: str
    last_updated: str
    message_count: int


class Message(BaseModel):
    role: str
    content: str
    timestamp: str


class ChatDetail(BaseModel):
    id: str
    title: str
    messages: List[Message]
    created: str
    updated: str


class SourceDocument(BaseModel):
    content: str = Field(..., description="Document content snippet")
    source: str = Field(..., description="Source filename")
    relevance_score: float = Field(..., ge=0, le=1, description="Similarity score")


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    metadata: Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    filename: str
    status: str
    message: Optional[str] = None


class IngestionResponse(BaseModel):
    success: bool
    message: str
    documents_loaded: int
    chunks_created: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Any]


class StatsResponse(BaseModel):
    total_chats: int
    total_messages: int
    avg_messages_per_chat: float
    vector_store_initialized: bool
    
class AgentChatResponse(BaseModel):
    """
    Response from agent-powered chat endpoint.
    
    Extends ChatResponse with agent-specific information.
    """
    answer: str
    sources: List[SourceDocument]
    tool_used: Optional[str] = Field(None, description="Name of tool used, if any")
    tool_result: Optional[Dict[str, Any]] = Field(None, description="Result from tool execution")
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional metadata including intent, decision info"
    )
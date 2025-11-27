# app/api/models.py
"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


# =============================================================================
# Request Models
# =============================================================================

class NewChatRequest(BaseModel):
    """Request to create a new chat session."""
    title: Optional[str] = Field(default="Νέα Συνομιλία", max_length=200)


class MessageRequest(BaseModel):
    """Request to send a message."""
    role: str = Field(default="user", pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


# =============================================================================
# Response Models
# =============================================================================

class ChatSummary(BaseModel):
    """Summary of a chat session."""
    id: str
    title: str
    last_updated: str
    message_count: int


class Message(BaseModel):
    """A single message."""
    role: str
    content: str
    timestamp: str


class ChatDetail(BaseModel):
    """Detailed chat session information."""
    id: str
    title: str
    messages: List[Message]
    created: str
    updated: str


class SourceDocument(BaseModel):
    """A source document from RAG retrieval."""
    content: str = Field(..., description="Document content snippet")
    source: str = Field(..., description="Source filename")
    relevance_score: float = Field(..., ge=0, le=1, description="Similarity score")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    answer: str
    sources: List[SourceDocument]
    metadata: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    """Response from the agent."""
    answer: str
    sources: List[SourceDocument] = Field(default_factory=list)
    tool_used: Optional[str] = Field(None, description="Name of tool used")
    tool_result: Optional[Dict[str, Any]] = Field(None, description="Tool execution result")
    intent: str = Field(default="unknown", description="Detected intent")
    debug_info: List[str] = Field(default_factory=list, description="Debug information")
    execution_time: float = Field(default=0.0, description="Processing time in seconds")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    internal_thinking: Optional[str] = Field(None, description="Model's internal reasoning (debug)")


# =============================================================================
# File Operation Models
# =============================================================================

class UploadResponse(BaseModel):
    """Response from file upload."""
    filename: str
    status: str
    message: Optional[str] = None


class IngestionResponse(BaseModel):
    """Response from document ingestion."""
    success: bool
    message: str
    documents_loaded: int
    chunks_created: int


# =============================================================================
# System Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    services: Dict[str, Any]


class StatsResponse(BaseModel):
    """System statistics response."""
    total_chats: int
    total_messages: int
    avg_messages_per_chat: float
    vector_store_initialized: bool
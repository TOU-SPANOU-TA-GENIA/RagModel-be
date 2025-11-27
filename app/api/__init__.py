# app/api/__init__.py
"""
API routes and models.
"""

from app.api.config_routes import router as config_router
from app.api.models import (
    NewChatRequest,
    MessageRequest,
    ChatSummary,
    Message,
    ChatDetail,
    SourceDocument,
    ChatResponse,
    AgentResponse,
    UploadResponse,
    IngestionResponse,
    ErrorResponse,
    HealthResponse,
    StatsResponse
)

__all__ = [
    # Routers
    "config_router",
    
    # Request models
    "NewChatRequest",
    "MessageRequest",
    
    # Response models
    "ChatSummary",
    "Message",
    "ChatDetail",
    "SourceDocument",
    "ChatResponse",
    "AgentResponse",
    "UploadResponse",
    "IngestionResponse",
    "ErrorResponse",
    "HealthResponse",
    "StatsResponse"
]
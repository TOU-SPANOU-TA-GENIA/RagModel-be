# app/main.py
"""
FastAPI application entry point with authentication.
Integrates all routes, middleware, and lifecycle management.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
import mimetypes

# Import API routers
from app.api import config_router
from app.api.auth_routes import router as auth_router
from app.api.chat_routes_authenticated import router as chat_router

# Import models
from app.api import (
    NewChatRequest, MessageRequest,
    ChatSummary, ChatDetail, SourceDocument,
    AgentResponse, UploadResponse, IngestionResponse,
    HealthResponse, StatsResponse
)

# Import configuration
from app.config import (
    SERVER, PATHS, AGENT,
    SYSTEM_INSTRUCTION
)

# Import core modules
from app.core import (
    startup_manager,
    ChatNotFoundException,
    RAGException
)

# Import database initialization
from app.db.init_db import init_database
from app.db.storage import storage

# Import utilities
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - startup and shutdown."""
    logger.info("🚀 Starting application...")
    
    # Startup
    try:
        # Initialize database
        logger.info("Initializing database...")
        init_database()
        
        # Initialize system (RAG, models, etc.)
        logger.info("Initializing system components...")
        await startup_manager.initialize_system()
        
        logger.info("✅ Application ready")
        from pathlib import Path
        db_path = Path(PATHS.data_dir) / 'app.db'
        logger.info(f"   Database: SQLite at {db_path}")
        logger.info(f"   Redis: {'✅ Connected' if storage.redis_available else '⚠️  Not available (running without cache)'}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        from app.config import config_manager
        config_manager.save()
        logger.info("Configuration saved")
    except Exception as e:
        logger.warning(f"Config save failed: {e}")
    
    logger.info("🛑 Shutting down...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="RagModel-be with Authentication",
    description="AI Agent API with RAG capabilities, user authentication, and persistent chat storage",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================================================================
# CORS Middleware
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=SERVER.cors_origins,  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Include Routers
# =============================================================================

# Authentication routes
app.include_router(auth_router)

# Chat routes (authenticated)
app.include_router(chat_router)

# Configuration routes
app.include_router(config_router)


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(ChatNotFoundException)
async def chat_not_found_handler(request, exc):
    """Handle chat not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Chat not found", "detail": str(exc)}
    )


@app.exception_handler(RAGException)
async def rag_exception_handler(request, exc):
    """Handle RAG system errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "System error", "detail": str(exc)}
    )


# =============================================================================
# System Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "RagModel-be API",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "authentication": True,
            "chat_persistence": True,
            "rag": True,
            "file_operations": True
        },
        "endpoints": {
            "auth": "/auth",
            "chats": "/chats",
            "config": "/config",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from app.db.storage import storage
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "sqlite",
            "database_available": True,
            "redis_cache": storage.redis_available,
            "agent": AGENT.mode
        }
    }
    
    # Check if we can access the database
    try:
        import sqlite3
        from pathlib import Path
        db_path = Path(PATHS.data_dir) / "app.db"
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        health_status["services"]["database_status"] = "operational"
    except Exception as e:
        health_status["services"]["database_status"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


@app.get("/startup-status")
async def get_startup_status():
    """Check startup status and system readiness."""
    status_info = startup_manager.get_status()
    
    from pathlib import Path
    db_path = Path(PATHS.data_dir) / "app.db"
    
    return {
        "startup_complete": status_info.get("complete", False),
        "components": status_info.get("components", {}),
        "errors": status_info.get("errors", []),
        "auth_enabled": True,
        "database": {
            "type": "sqlite",
            "path": str(db_path),
            "cache": "redis" if storage.redis_available else "none"
        }
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    try:
        # Get database stats
        import sqlite3
        from pathlib import Path
        db_path = Path(PATHS.data_dir) / "app.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count users
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        # Count chats
        cursor.execute("SELECT COUNT(*) FROM chats")
        chat_count = cursor.fetchone()[0]
        
        # Count messages
        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Redis stats
        redis_stats = {}
        if storage.redis_available:
            redis_stats = {
                "cache_enabled": True,
                "cached_keys": len(storage.redis.keys("*")) if storage.redis else 0
            }
        else:
            redis_stats = {"cache_enabled": False}
        
        return {
            "users": user_count,
            "chats": chat_count,
            "messages": message_count,
            "uptime_seconds": 0,  # TODO: Track actual uptime
            "redis": redis_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


# =============================================================================
# Legacy/Public Endpoints (for backward compatibility)
# =============================================================================

@app.post("/chat")
async def simple_chat_endpoint(msg: MessageRequest):
    """
    Simple chat endpoint (no authentication required).
    For testing and backward compatibility.
    
    Note: This endpoint does not persist conversations.
    For persistent chats, use the authenticated /chats endpoints.
    """
    from app.agent.integration import get_agent
    from app.core.conversation_memory import conversation_memory
    import uuid
    
    # Create temporary session
    session_id = str(uuid.uuid4())
    session = conversation_memory.get_or_create_session(session_id)
    session.add_message("user", msg.content)
    history = session.get_recent_messages(max_messages=10)[:-1]
    
    agent = get_agent()
    response = agent.process_query(
        query=msg.content,
        chat_history=history,
        metadata={"session_id": session_id}
    )
    
    # Build response
    source_docs = [
        SourceDocument(
            content=src.get("content", ""),
            source=src.get("metadata", {}).get("source", "Unknown"),
            relevance_score=src.get("score", 0.0)
        )
        for src in response.sources
    ]
    
    return AgentResponse(
        answer=response.answer,
        sources=source_docs,
        tool_used=response.tool_used,
        tool_result=response.tool_result,
        intent=response.intent,
        debug_info=response.debug_info if AGENT.debug_mode else [],
        execution_time=response.execution_time,
        internal_thinking=response.internal_thinking,
        session_id=session_id
    )


@app.post("/query")
async def legacy_query(query: str):
    """
    Legacy query endpoint (deprecated - use authenticated /chats endpoints instead).
    Kept for backward compatibility.
    """
    from app.agent.integration import get_agent
    
    agent = get_agent()
    response = agent.process_query(query)
    
    return {
        "answer": response.answer,
        "sources": response.sources,
        "deprecated": True,
        "message": "This endpoint is deprecated. Please use /auth/login and /chats endpoints."
    }


# =============================================================================
# File Upload Endpoints (can be restricted to authenticated users if needed)
# =============================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a document to the knowledge base."""
    from app.rag.ingestion import save_upload
    
    try:
        await save_upload(file)
        return UploadResponse(
            filename=file.filename,
            status="uploaded",
            message=f"File uploaded. Call /ingest/file/{file.filename} to index it."
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@app.post("/ingest/file/{filename}", response_model=IngestionResponse)
async def ingest_file(filename: str):
    """Ingest a single uploaded file into the RAG system."""
    from app.rag.ingestion import ingest_single_file
    from pathlib import Path
    
    try:
        filepath = str(PATHS.knowledge_dir / filename)
        result = await ingest_single_file(filepath)
        return IngestionResponse(**result)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@app.post("/ingest/all", response_model=IngestionResponse)
async def ingest_all_documents(rebuild: bool = False):
    """Ingest all documents in the knowledge directory."""
    from app.rag.ingestion import ingest_all_documents
    
    try:
        result = await ingest_all_documents(rebuild=rebuild)
        return IngestionResponse(**result)
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch ingestion failed: {str(e)}"
        )


@app.get("/documents")
async def list_documents():
    """List all documents in the knowledge base."""
    from pathlib import Path
    
    knowledge_dir = PATHS.knowledge_dir
    if not knowledge_dir.exists():
        return {"documents": []}
    
    documents = []
    for file in knowledge_dir.iterdir():
        if file.is_file():
            documents.append({
                "filename": file.name,
                "size": file.stat().st_size,
                "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            })
    
    return {"documents": documents}


@app.get("/documents/{filename}")
async def download_document(filename: str):
    """Download a document from the knowledge base."""
    from pathlib import Path
    
    filepath = PATHS.knowledge_dir / filename
    
    if not filepath.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Guess mimetype
    mimetype, _ = mimetypes.guess_type(filename)
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type=mimetype or "application/octet-stream"
    )


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from the knowledge base."""
    from pathlib import Path
    
    filepath = PATHS.knowledge_dir / filename
    
    if not filepath.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        filepath.unlink()
        logger.info(f"Deleted document: {filename}")
        return {"message": f"Document {filename} deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


# =============================================================================
# Development/Debug Endpoints (disable in production)
# =============================================================================

if AGENT.debug_mode:
    @app.get("/debug/memory")
    async def debug_memory():
        """Debug endpoint to inspect in-memory storage."""
        from app.core.memory_store import InMemoryDatabase
        
        db = InMemoryDatabase()
        return {
            "documents": len(db.documents),
            "embeddings": len(db.document_embeddings),
            "chats": len(db.chats),
            "models": list(db.models.keys()),
            "embedding_cache_size": len(db.embedding_cache)
        }
    
    @app.get("/debug/config")
    async def debug_config():
        """Debug endpoint to view current configuration."""
        from app.config import config_manager
        return config_manager.get_all_values()


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
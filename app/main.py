# app/main.py
"""
FastAPI application entry point.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
import mimetypes

from app.api import (
    config_router,
    NewChatRequest, MessageRequest,
    ChatSummary, ChatDetail, SourceDocument,
    AgentResponse, UploadResponse, IngestionResponse,
    HealthResponse, StatsResponse
)
from app.chat import (
    create_chat, list_chats, get_chat,
    append_message, get_history, delete_chat, get_stats
)
from app.config import (
    SERVER, PATHS, AGENT,
    SYSTEM_INSTRUCTION
)
from app.core import (
    startup_manager,
    ChatNotFoundException,
    RAGException
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("🚀 Starting application...")
    
    try:
        await startup_manager.initialize_system()
        logger.info("✅ Application ready")
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


app = FastAPI(
    title="AI Agent RAG API",
    description="Modular AI Agent with RAG capabilities",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=SERVER.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(config_router)


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(ChatNotFoundException)
async def chat_not_found_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Chat not found", "detail": str(exc)}
    )


@app.exception_handler(RAGException)
async def rag_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "System error", "detail": str(exc)}
    )


# =============================================================================
# System Endpoints
# =============================================================================

@app.get("/startup-status")
async def get_startup_status():
    """Check startup status."""
    return {
        "status": "ready" if startup_manager.is_ready() else "starting",
        "timestamp": datetime.utcnow().isoformat(),
        "startup_info": startup_manager.get_status(),
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from app.agent.integration import get_agent
    
    try:
        agent = get_agent()
        agent_ok = True
    except:
        agent_ok = False
    
    llm_ready = False
    try:
        from app.llm.prewarmed_provider import prewarmed_llm
        llm_ready = prewarmed_llm.is_ready()
    except:
        pass
    
    services = {
        "server_ready": startup_manager.is_ready(),
        "llm_prewarmed": llm_ready,
        "agent_available": agent_ok,
    }
    
    return {
        "status": "healthy" if all(services.values()) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "services": services
    }


@app.get("/stats", response_model=StatsResponse)
async def get_system_stats():
    """Get system statistics."""
    from app.agent.integration import get_agent
    
    stats = get_stats()
    
    try:
        agent = get_agent()
        vector_store_ok = agent.retriever is not None
    except:
        vector_store_ok = False
    
    return {**stats, "vector_store_initialized": vector_store_ok}


# =============================================================================
# Chat Endpoints
# =============================================================================

@app.post("/chats", response_model=str, status_code=status.HTTP_201_CREATED)
async def create_new_chat(req: NewChatRequest):
    """Create a new chat session."""
    return create_chat(title=req.title)


@app.get("/chats", response_model=list[ChatSummary])
async def get_all_chats():
    """Get all chat sessions."""
    return list_chats()


@app.get("/chats/{chat_id}", response_model=ChatDetail)
async def get_chat_details(chat_id: str):
    """Get chat details."""
    return get_chat(chat_id)


@app.delete("/chats/{chat_id}")
async def delete_chat_endpoint(chat_id: str):
    """Delete a chat."""
    delete_chat(chat_id)
    return {"success": True}


@app.post("/chats/{chat_id}/message", response_model=AgentResponse)
async def post_message_to_chat(chat_id: str, msg: MessageRequest):
    """Send message to a chat session."""
    from app.agent.integration import get_agent
    
    _ = get_chat(chat_id)
    append_message(chat_id, msg.role, msg.content)
    history = get_history(chat_id)[:-1]
    
    agent = get_agent()
    response = agent.process_query(msg.content, history)
    append_message(chat_id, "assistant", response.answer)
    
    return _build_agent_response(response)


@app.post("/chat", response_model=AgentResponse)
async def simple_chat(msg: MessageRequest, session_id: Optional[str] = None):
    """Simple chat endpoint with session support."""
    from app.agent.integration import get_agent
    from app.core.conversation_memory import conversation_memory
    import uuid
    
    if not session_id:
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
    
    result = _build_agent_response(response)
    result.session_id = session_id
    return result


def _build_agent_response(response) -> AgentResponse:
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
        internal_thinking=response.internal_thinking  # ADD THIS LINE
    )

# =============================================================================
# File Endpoints
# =============================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a document."""
    from app.rag.ingestion import save_upload
    
    await save_upload(file)
    return UploadResponse(
        filename=file.filename,
        status="uploaded",
        message=f"Call /ingest/file/{file.filename} to index it."
    )


@app.post("/ingest/file/{filename}", response_model=IngestionResponse)
async def ingest_file(filename: str):
    """Ingest a single file."""
    from app.rag.ingestion import ingest_single_file
    from pathlib import Path
    
    filepath = str(Path(PATHS.knowledge_dir) / filename)
    result = await ingest_single_file(filepath)
    return IngestionResponse(**result)


@app.post("/ingest/all", response_model=IngestionResponse)
async def ingest_all_documents(rebuild: bool = False):
    """Ingest all documents."""
    from app.rag.ingestion import ingest_directory
    from app.agent.integration import agent_manager
    from pathlib import Path
    
    result = ingest_directory(Path(PATHS.knowledge_dir), rebuild=rebuild)
    
    if result["success"]:
        agent_manager.reset()
    
    return IngestionResponse(**result)


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a generated file."""
    from pathlib import Path
    
    file_path = Path(PATHS.outputs_dir) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    mime_type, _ = mimetypes.guess_type(str(file_path))
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=mime_type or 'application/octet-stream'
    )


@app.get("/agent/capabilities")
async def get_agent_capabilities():
    """Get agent capabilities."""
    from app.agent.integration import get_agent
    
    agent = get_agent()
    
    return {
        "version": "2.0.0",
        "mode": AGENT.mode,
        "capabilities": {
            "rag_enabled": agent.retriever is not None,
            "tools_enabled": len(agent.tools) > 0,
            "debug_mode": AGENT.debug_mode
        },
        "available_tools": [
            {"name": t.name, "description": t.description}
            for t in agent.tools.values()
        ]
    }
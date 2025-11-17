# app/main.py
"""
FastAPI application with new modular agent architecture.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from .models import (
    NewChatRequest, MessageRequest, ChatResponse, UploadResponse,
    ErrorResponse, HealthResponse, StatsResponse, IngestionResponse,
    ChatSummary, ChatDetail, SourceDocument, AgentResponse
)
from .chat_manager import (
    create_chat, list_chats, append_message, get_history, get_chat,
    update_chat_title, delete_chat, get_stats
)
from .config import (
    CORS_ORIGINS, SYSTEM_INSTRUCTION, KNOWLEDGE_DIR,
    AGENT_MODE, AGENT_DEBUG_MODE
)
from .exceptions import (
    ChatNotFoundException, VectorStoreNotInitializedException,
    ModelLoadException, RAGException
)
from .logger import setup_logger

# NEW: Import from new architecture
from .agent.integration import create_agent, get_agent
from .rag.ingestion import save_upload, ingest_single_file, ingest_directory

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ultra-fast startup with pre-warmed LLM."""
    logger.info("🚀 Starting RAG application with ULTRA-FAST startup...")
    
    try:
        # Ultra-fast initialization
        from app.startup import startup_manager
        await startup_manager.initialize_system_ultra_fast()
        
        logger.info("✅ Application READY - instant responses enabled")
        logger.info("   🔥 LLM pre-warming in background...")
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        raise
    
    yield
    
    logger.info("🛑 Shutting down RAG application...")

app = FastAPI(
    title="AI Agent RAG API v2",
    description="Modular AI Agent with RAG capabilities",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(ChatNotFoundException)
async def chat_not_found_handler(request, exc: ChatNotFoundException):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Chat not found", "detail": str(exc)}
    )

@app.exception_handler(RAGException)
async def rag_exception_handler(request, exc: RAGException):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "System error", "detail": str(exc)}
    )

@app.get("/startup-status")
async def get_startup_status():
    """Check startup and pre-loading status."""
    from app.startup import startup_manager
    
    status = startup_manager.get_status()
    
    return {
        "status": "ready" if startup_manager.is_ready() else "starting",
        "timestamp": datetime.utcnow().isoformat(),
        "startup_info": status,
        "message": "Server is ready to handle requests" if startup_manager.is_ready() else "Server is starting up..."
    }
    

# ============================================================================
# System Endpoints
# ============================================================================

# ============================================================================
# System Endpoints
# ============================================================================

@app.get("/memory/stats")
async def get_memory_stats():
    """Get in-memory storage statistics."""
    from app.core.memory_store import memory_db
    
    usage = memory_db.get_memory_usage()
    total_bytes = sum(usage.values())
    
    return {
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "breakdown": {
            key: round(value / (1024 * 1024), 2) 
            for key, value in usage.items()
        },
        "document_count": len(memory_db.documents),
        "embedding_cache_size": len(memory_db.embedding_cache),
        "models_cached": list(memory_db.models.keys())
    }

@app.post("/memory/clear-cache")
async def clear_memory_cache(keep_models: bool = True):
    """Clear memory caches to free RAM."""
    from app.core.memory_store import memory_db
    
    memory_db.clear_caches(keep_models=keep_models)
    
    return {"message": "Memory caches cleared", "models_kept": keep_models}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with LLM pre-warm status."""
    from app.startup import startup_manager
    from app.llm.prewarmed_provider import prewarmed_llm
    
    try:
        agent = get_agent()
        agent_ok = True
    except:
        agent_ok = False
    
    services = {
        "server_ready": startup_manager.is_ready(),
        "llm_prewarmed": prewarmed_llm.is_ready(),
        "agent_available": agent_ok,
        "knowledge_base": KNOWLEDGE_DIR.exists(),
        "startup_status": startup_manager.get_status(),
    }
    
    # Determine status
    if startup_manager.is_ready() and prewarmed_llm.is_ready():
        overall_status = "healthy"
    elif startup_manager.is_ready():
        overall_status = "degraded"  # Server ready, LLM still warming
    else:
        overall_status = "starting"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "services": services
    }
    

@app.get("/stats", response_model=StatsResponse)
async def get_system_stats():
    """Get system statistics."""
    stats = get_stats()
    
    # Check if vector store is initialized
    try:
        agent = get_agent()
        vector_store_initialized = agent.retriever is not None
    except:
        vector_store_initialized = False
    
    return {
        **stats,
        "vector_store_initialized": vector_store_initialized
    }

# ============================================================================
# Chat Management Endpoints
# ============================================================================

@app.post("/chats", response_model=str, status_code=status.HTTP_201_CREATED)
async def create_new_chat(req: NewChatRequest):
    """Create a new chat session."""
    try:
        chat_id = create_chat(title=req.title)
        logger.info(f"Created chat: {chat_id}")
        return chat_id
    except Exception as e:
        logger.error(f"Failed to create chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat"
        )

@app.get("/chats", response_model=list[ChatSummary])
async def get_all_chats():
    """Get all chat sessions."""
    try:
        chats = list_chats()
        return chats
    except Exception as e:
        logger.error(f"Failed to list chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chats"
        )

@app.get("/chats/{chat_id}", response_model=ChatDetail)
async def get_chat_details(chat_id: str):
    """Get details of a specific chat."""
    try:
        chat = get_chat(chat_id)
        return chat
    except ChatNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )

@app.delete("/chats/{chat_id}")
async def delete_chat_endpoint(chat_id: str):
    """Delete a chat session."""
    try:
        delete_chat(chat_id)
        return {"success": True, "message": "Chat deleted"}
    except ChatNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )

# ============================================================================
# Agent Chat Endpoints
# ============================================================================

@app.post("/chats/{chat_id}/message", response_model=AgentResponse)
async def post_message_to_chat(chat_id: str, msg: MessageRequest):
    """Send message to chat with conversation memory."""
    try:
        # Validate chat exists
        _ = get_chat(chat_id)
        
        # Store user message
        append_message(chat_id, msg.role, msg.content)
        
        # Get history (exclude just-added message)
        history = get_history(chat_id)[:-1]
        
        # Process with agent using chat ID as session ID
        agent = get_agent()
        
        # Create context with chat ID as session ID
        context_metadata = {"session_id": chat_id}
        
        response = agent.process_query(msg.content, history)
        
        # Store assistant response
        append_message(chat_id, "assistant", response.answer)
        
        logger.info(f"Processed message for chat {chat_id} - Intent: {response.intent}")
        
        # Convert sources
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
            debug_info=response.debug_info if AGENT_DEBUG_MODE else [],
            execution_time=response.execution_time
        )
        
    except ChatNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )

# app/main.py (FIX the simple_chat endpoint)
@app.post("/chat", response_model=AgentResponse)
async def simple_chat(msg: MessageRequest, session_id: Optional[str] = None):
    """Stateless chat endpoint with session support."""
    try:
        agent = get_agent()
        
        # Use provided session ID or create stateless session
        context_metadata = {}
        if session_id:
            context_metadata["session_id"] = session_id
        
        response = agent.process_query(msg.content, [])
        
        # Convert to dictionary and add session ID
        response_dict = {
            "answer": response.answer,
            "sources": response.sources,
            "tool_used": response.tool_used,
            "tool_result": response.tool_result,
            "intent": response.intent,
            "debug_info": response.debug_info,
            "execution_time": response.execution_time
        }
        if session_id:
            response_dict["session_id"] = session_id
        
        return response_dict
        
    except Exception as e:
        logger.error(f"Error in simple chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )

# ============================================================================
# Document Management Endpoints
# ============================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a document for ingestion."""
    try:
        filepath = await save_upload(file)
        logger.info(f"File uploaded: {file.filename}")
        
        return UploadResponse(
            filename=file.filename,
            status="uploaded",
            message=f"File saved. Call /ingest/file/{file.filename} to index it."
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )

@app.post("/ingest/file/{filename}", response_model=IngestionResponse)
async def ingest_file(filename: str):
    """Ingest a single file into the knowledge base."""
    try:
        filepath = str(KNOWLEDGE_DIR / filename)
        result = await ingest_single_file(filepath)
        
        if result["success"]:
            logger.info(f"File ingested: {filename}")
            return IngestionResponse(**result)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/ingest/all", response_model=IngestionResponse)
async def ingest_all_documents(rebuild: bool = False):
    """Ingest all documents in the knowledge directory."""
    try:
        logger.info(f"Starting full ingestion (rebuild={rebuild})")
        result = ingest_directory(KNOWLEDGE_DIR, rebuild=rebuild)
        
        if result["success"]:
            # Reinitialize agent with new index
            from .agent.integration import agent_manager
            agent_manager.reset()
            
            return IngestionResponse(**result)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
    except Exception as e:
        logger.error(f"Full ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ============================================================================
# Agent Capabilities Endpoint
# ============================================================================

@app.get("/agent/capabilities")
async def get_agent_capabilities():
    """Get information about agent capabilities."""
    try:
        agent = get_agent()
        
        tools_info = []
        for tool in agent.tools.values():
            tools_info.append({
                "name": tool.name,
                "description": tool.description
            })
        
        return {
            "version": "2.0.0",
            "mode": AGENT_MODE,
            "capabilities": {
                "rag_enabled": agent.retriever is not None,
                "tools_enabled": len(agent.tools) > 0,
                "debug_mode": AGENT_DEBUG_MODE
            },
            "available_tools": tools_info,
            "intents": ["question", "action", "conversation"]
        }
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get agent capabilities"
        )
# app/main.py
"""
FastAPI application entry point with authentication, streaming, and Greek language support.
FIXED: Network file indexing now properly triggers RAG ingestion.
"""

import asyncio
import json
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
from pydantic import BaseModel
from pathlib import Path

# Import API routers
from app.api import config_router
from app.api.auth_routes import router as auth_router
from app.api.chat_routes_authenticated import router as chat_router

# Import models
from app.api import HealthResponse

# Import configuration
from app.config import SERVER, PATHS, config

# Import core modules
from app.core import startup_manager, ChatNotFoundException, RAGException

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
        
        # ========== NETWORK FILESYSTEM + RAG INITIALIZATION ==========
        logger.info("Initializing network filesystem...")
        try:
            from app.core.network_filesystem import initialize_network_monitor, NetworkShare
            from app.core.network_rag_integration import initialize_network_rag
            from app.rag.ingestion import ingest_file
            
            network_config = config.get_section('network_filesystem')
            
            if network_config and network_config.get('enabled'):
                shares = []
                for share_data in network_config.get('shares', []):
                    if share_data.get('enabled'):
                        share = NetworkShare(
                            name=share_data['name'],
                            mount_path=Path(share_data['mount_path']),
                            share_type=share_data.get('share_type', 'smb'),
                            enabled=True,
                            auto_index=share_data.get('auto_index', True),
                            watch_for_changes=share_data.get('watch_for_changes', True),
                            scan_interval=share_data.get('scan_interval', 300),
                            include_extensions=share_data.get('include_extensions', []),
                            exclude_patterns=share_data.get('exclude_patterns', []),
                            max_file_size_mb=share_data.get('max_file_size_mb', 100)
                        )
                        shares.append(share)
                        logger.info(f"  📁 Configured share: {share.name} at {share.mount_path}")
                
                if shares:
                    # Initialize monitoring
                    monitor = initialize_network_monitor(shares)
                    logger.info(f"✅ Network monitoring initialized for {len(shares)} shares")
                    
                    # Initialize RAG integration with ingestion function
                    if network_config.get('auto_start_monitoring', True):
                        integrator = initialize_network_rag(monitor, ingest_file)
                        
                        # Trigger initial scan
                        logger.info("🔍 Starting initial file discovery...")
                        monitor.scan_all_shares()
                        stats = monitor.get_stats()
                        logger.info(f"📊 Discovery complete: {stats['total_files']} files found")
                        
                        for share_name, count in stats['by_share'].items():
                            logger.info(f"   - {share_name}: {count} files")
                        
                        # CRITICAL: Index files NOW before starting background thread
                        logger.info("📥 Starting initial RAG indexing...")
                        index_result = integrator.index_all_now()
                        if index_result.get("success"):
                            logger.info(f"✅ Indexed {index_result['files_indexed']} files into RAG")
                        else:
                            logger.warning(f"⚠️ Indexing issue: {index_result.get('message')}")
                        
                        # Start background monitoring
                        integrator.start()
                        logger.info("✅ Network-RAG integration started")
                    else:
                        logger.info("Auto-start monitoring disabled")
                else:
                    logger.warning("No enabled network shares configured")
            else:
                logger.info("Network filesystem disabled in configuration")
        
        except Exception as e:
            logger.error(f"Network filesystem initialization failed: {e}")
            import traceback
            traceback.print_exc()
        # ========== END NETWORK FILESYSTEM ==========
        
        # Initialize system (Agent, Models, etc.)
        logger.info("Initializing system components...")
        await startup_manager.initialize_system()
        
        logger.info("✅ Application ready")
        logger.info(f"   Language: Greek (Ελληνικά)")
        logger.info(f"   Streaming: Enabled")
        logger.info(f"   Thinking tags: Enabled")
        
        db_path = Path(PATHS.data_dir) / 'app.db'
        logger.info(f"   Database: SQLite at {db_path}")
        logger.info(f"   Redis: {'✅ Connected' if storage.redis_available else '⚠️  Not available'}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        logger.info("Stopping network monitoring...")
        try:
            from app.core.network_rag_integration import get_network_integrator
            integrator = get_network_integrator()
            if integrator:
                integrator.stop()
        except Exception as e:
            logger.debug(f"Network cleanup: {e}")
        
        logger.info("Shutdown complete")
    except Exception as e:
        logger.warning(f"Shutdown cleanup: {e}")
    
    logger.info("🛑 Shutting down...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="RagModel-be API",
    description="AI Agent API με υποστήριξη Ελληνικών, RAG, και streaming responses",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# =============================================================================
# CORS Middleware
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=SERVER.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Include Routers
# =============================================================================

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(config_router)


# =============================================================================
# Streaming Endpoints
# =============================================================================

class StreamingChatRequest(BaseModel):
    """Request for streaming chat."""
    content: str
    chat_id: Optional[str] = None
    include_thinking: bool = False


async def generate_streaming_response(
    content: str,
    chat_id: Optional[str] = None,
    include_thinking: bool = False
):
    """Async generator for streaming SSE responses."""
    from app.agent.integration import get_agent
    from app.core.interfaces import Context
    
    try:
        agent = get_agent()
        
        # Build context with all required arguments
        context = Context(
            query=content,
            chat_history=[],
            metadata={"chat_id": chat_id},
            debug_info=[]
        )
        
        # Run preprocessing (intent, RAG, tools)
        context = agent.run_preprocessing(context)
        prompt = context.metadata.get("prompt", content)
        
        # Get LLM provider
        llm = agent.llm_provider
        
        # Check if provider supports streaming
        if hasattr(llm, 'generate_stream'):
            async for event in llm.generate_stream(prompt, include_thinking=include_thinking):
                yield f"data: {json.dumps({'type': event.event_type.value, 'data': event.data}, ensure_ascii=False)}\n\n"
        else:
            # Fallback: generate full response and yield in chunks
            from app.config import LLM
            response = llm.generate(prompt, max_new_tokens=LLM.max_new_tokens)
            
            # Clean response
            from app.agent.orchestrator import ResponseCleaner
            thinking, clean_response = ResponseCleaner.extract_thinking(response)
            
            if include_thinking and thinking:
                yield f"data: {json.dumps({'type': 'thinking_start', 'data': ''}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'data': thinking}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'thinking_end', 'data': ''}, ensure_ascii=False)}\n\n"
            
            yield f"data: {json.dumps({'type': 'response_start', 'data': ''}, ensure_ascii=False)}\n\n"
            
            # Yield response in chunks
            chunk_size = 20
            for i in range(0, len(clean_response), chunk_size):
                chunk = clean_response[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'data': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
            
            yield f"data: {json.dumps({'type': 'response_end', 'data': ''}, ensure_ascii=False)}\n\n"
        
        yield f"data: {json.dumps({'type': 'done', 'data': ''}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"


@app.post("/stream/chat")
async def stream_chat_post(request: StreamingChatRequest):
    """Stream chat response using Server-Sent Events (SSE)."""
    return StreamingResponse(
        generate_streaming_response(
            content=request.content,
            chat_id=request.chat_id,
            include_thinking=request.include_thinking
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/stream/chat")
async def stream_chat_get(
    content: str = Query(..., description="Message content"),
    chat_id: Optional[str] = Query(None, description="Chat ID"),
    include_thinking: bool = Query(False, description="Include thinking in stream")
):
    """Stream chat using GET (for EventSource API)."""
    return StreamingResponse(
        generate_streaming_response(
            content=content,
            chat_id=chat_id,
            include_thinking=include_thinking
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# =============================================================================
# RAG Status Endpoint
# =============================================================================

@app.get("/rag/status")
async def get_rag_status():
    """Get RAG indexing status."""
    try:
        from app.core.network_rag_integration import get_network_integrator
        from app.core.network_filesystem import get_network_monitor
        
        integrator = get_network_integrator()
        monitor = get_network_monitor()
        
        if not integrator or not monitor:
            return {"status": "not_initialized", "message": "Network RAG not initialized"}
        
        status = integrator.get_status()
        return {
            "status": "active",
            "total_network_files": status["total_network_files"],
            "indexed_files": status["indexed_files"],
            "pending_files": status["pending_files"],
            "shares": status["shares"],
            "auto_indexing": status["auto_indexing_active"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/rag/reindex")
async def trigger_reindex():
    """Manually trigger RAG reindexing of network files."""
    try:
        from app.core.network_rag_integration import get_network_integrator
        
        integrator = get_network_integrator()
        if not integrator:
            raise HTTPException(status_code=503, detail="Network RAG not initialized")
        
        result = integrator.index_all_now()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(ChatNotFoundException)
async def chat_not_found_handler(request, exc):
    """Handle chat not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Η συνομιλία δεν βρέθηκε", "detail": str(exc)}
    )


@app.exception_handler(RAGException)
async def rag_exception_handler(request, exc):
    """Handle RAG system errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Σφάλμα συστήματος", "detail": str(exc)}
    )


# =============================================================================
# System Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "RagModel-be API",
        "version": "2.1.0",
        "language": "Greek (Ελληνικά)",
        "status": "running",
        "features": {
            "authentication": True,
            "chat_persistence": True,
            "rag": True,
            "streaming": True,
            "thinking_tags": True,
            "greek_language": True,
            "network_filesystem": True
        },
        "endpoints": {
            "auth": "/auth",
            "chats": "/chats",
            "stream": "/stream/chat",
            "config": "/config",
            "rag_status": "/rag/status",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Get RAG status
    rag_status = "unknown"
    try:
        from app.core.network_rag_integration import get_network_integrator
        integrator = get_network_integrator()
        if integrator:
            status_info = integrator.get_status()
            rag_status = f"{status_info['indexed_files']}/{status_info['total_network_files']} files indexed"
    except:
        pass
    
    return {
        "status": "healthy",
        "database": "sqlite",
        "redis_available": storage.redis_available,
        "language": "Greek",
        "streaming": True,
        "rag_status": rag_status
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
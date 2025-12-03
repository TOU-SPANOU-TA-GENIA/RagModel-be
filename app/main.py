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
        logger.info("=" * 60)
        logger.info("🔌 NETWORK FILESYSTEM INITIALIZATION")
        logger.info("=" * 60)

        try:
            from app.core.network_filesystem import initialize_network_monitor, NetworkShare
            from app.core.network_rag_integration import initialize_network_rag
            from app.rag.ingestion import ingest_file
            
            network_config = config.get_section('network_filesystem')
            
            if network_config:
                logger.info(f"Network config found: enabled={network_config.get('enabled')}")
            else:
                logger.warning("No 'network_filesystem' section in config.json")
            
            if network_config and network_config.get('enabled'):
                shares = []
                shares_config = network_config.get('shares', [])
                logger.info(f"Configured shares: {len(shares_config)}")
                
                for i, share_data in enumerate(shares_config):
                    logger.info(f"  Share {i+1}: {share_data.get('name', 'unnamed')}")
                    logger.info(f"    enabled: {share_data.get('enabled')}")
                    logger.info(f"    path: {share_data.get('mount_path')}")
                    
                    if share_data.get('enabled'):
                        mount_path = Path(share_data['mount_path'])
                        
                        # Check if path exists
                        if mount_path.exists():
                            logger.info(f"    ✅ Path accessible")
                            
                            share = NetworkShare(
                                name=share_data['name'],
                                mount_path=mount_path,
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
                        else:
                            logger.warning(f"    ❌ Path NOT accessible: {mount_path}")
                
                if shares:
                    logger.info(f"\n📁 Initializing monitor for {len(shares)} shares...")
                    monitor = initialize_network_monitor(shares)
                    
                    logger.info("🔍 Scanning all shares for files...")
                    monitor.scan_all_shares()
                    
                    stats = monitor.get_stats()
                    logger.info(f"\n📊 DISCOVERY RESULTS:")
                    logger.info(f"   Total files found: {stats['total_files']}")
                    for share_name, count in stats['by_share'].items():
                        logger.info(f"   - {share_name}: {count} files")
                    
                    if network_config.get('auto_start_monitoring', True):
                        logger.info("\n📥 Initializing RAG integration...")
                        integrator = initialize_network_rag(monitor, ingest_file)
                        
                        # Index files NOW
                        logger.info("\n🚀 STARTING INITIAL RAG INDEXING...")
                        logger.info("-" * 40)
                        
                        index_result = integrator.index_all_now()
                        
                        logger.info("-" * 40)
                        if index_result.get("success"):
                            logger.info(f"✅ INDEXING COMPLETE:")
                            logger.info(f"   Files indexed: {index_result['files_indexed']}")
                            logger.info(f"   Total files: {index_result['total_files']}")
                            logger.info(f"   Message: {index_result.get('message', '')}")
                        else:
                            logger.error(f"❌ INDEXING FAILED:")
                            logger.error(f"   {index_result.get('message', 'Unknown error')}")
                        
                        # Start background monitoring
                        logger.info("\n🔄 Starting background monitoring thread...")
                        integrator.start()
                        logger.info("✅ Background indexing active (checks every 60s)")
                    else:
                        logger.info("Auto-start monitoring disabled in config")
                else:
                    logger.warning("No enabled/accessible network shares found")
            else:
                logger.info("Network filesystem disabled or not configured")

        except Exception as e:
            logger.error(f"❌ Network filesystem initialization FAILED: {e}")
            import traceback
            traceback.print_exc()

        logger.info("=" * 60)
        logger.info("🔌 NETWORK FILESYSTEM INITIALIZATION COMPLETE")
        logger.info("=" * 60)
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

from pydantic import BaseModel
from typing import Optional
import json
import asyncio
from typing import Optional
from queue import Queue, Empty
from threading import Thread
from concurrent.futures import ThreadPoolExecutor


class StreamingChatRequest(BaseModel):
    """Request for streaming chat."""
    content: str
    chat_id: Optional[str] = None
    include_thinking: bool = False
    max_tokens: Optional[int] = 256 


async def generate_streaming_response(
    content: str,
    chat_id: Optional[str] = None,
    include_thinking: bool = False,
    max_tokens: int = 256
):
    """
    TRUE streaming using the EXISTING agent's model.
    
    Key fix: Uses agent.llm_provider.model/tokenizer instead of loading new model.
    """
    from app.agent.integration import get_agent
    from app.core.interfaces import Context
    
    try:
        # Phase 1: Acknowledge
        yield f"data: {json.dumps({'type': 'status', 'data': '🔍 Αναζήτηση...'}, ensure_ascii=False)}\n\n"
        
        agent = get_agent()
        
        # Build context
        context = Context(
            query=content,
            chat_history=[],
            metadata={"chat_id": chat_id},
            debug_info=[]
        )
        
        # Phase 2: Preprocessing (RAG, intent)
        logger.info(f"📥 Query: {content[:100]}...")
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            preprocessing_future = loop.run_in_executor(
                executor,
                agent.run_preprocessing,
                context
            )
            
            while not preprocessing_future.done():
                yield f"data: {json.dumps({'type': 'heartbeat', 'data': ''}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.5)
            
            context = preprocessing_future.result()
        
        prompt = context.metadata.get("prompt", content)
        
        # Add instruction for thinking language and response language
        # Qwen3 /think mode will be enabled, but we instruct it on language
        thinking_instruction = """

<instructions>
THINKING LANGUAGE: When using <think> tags, think in English or Greek - NOT Chinese.
RESPONSE LANGUAGE: Your final response MUST be in Greek (Ελληνικά).

Process:
1. Think through the problem (in English or Greek)
2. Respond to the user in Greek
</instructions>

"""
        
        # Add /think at the end to enable thinking mode
        prompt = prompt + thinking_instruction + "/think"
        
        logger.info(f"📝 Prompt ready: {len(prompt)} chars (thinking enabled)")
        
        # Check for RAG context
        rag_context = context.metadata.get("rag_context", "")
        if rag_context:
            yield f"data: {json.dumps({'type': 'status', 'data': '📚 Βρέθηκαν σχετικά έγγραφα'}, ensure_ascii=False)}\n\n"
        
        # Phase 3: STREAMING GENERATION using existing model
        yield f"data: {json.dumps({'type': 'status', 'data': '💭 Δημιουργία απάντησης...'}, ensure_ascii=False)}\n\n"
        logger.info("📤 Sent 'generating' status")
        
        # Force flush by yielding a small delay
        await asyncio.sleep(0.1)
        
        yield f"data: {json.dumps({'type': 'response_start', 'data': ''}, ensure_ascii=False)}\n\n"
        logger.info("📤 Sent 'response_start'")
        
        await asyncio.sleep(0.1)
        
        # Get the EXISTING model from agent
        llm = agent.llm_provider
        
        # Handle different provider types
        if hasattr(llm, 'provider') and llm.provider is not None:
            # PreWarmedLLMProvider wraps FastLocalModelProvider
            inner_provider = llm.provider
            inner_provider._ensure_initialized()
            model = inner_provider.model
            tokenizer = inner_provider.tokenizer
            logger.info("Using model from PreWarmedLLMProvider")
        elif hasattr(llm, '_ensure_initialized'):
            # Direct provider with _ensure_initialized
            llm._ensure_initialized()
            model = llm.model
            tokenizer = llm.tokenizer
            logger.info("Using model from direct provider")
        elif hasattr(llm, 'model') and llm.model is not None:
            # Already initialized provider
            model = llm.model
            tokenizer = llm.tokenizer
            logger.info("Using already-initialized model")
        else:
            raise RuntimeError(f"Cannot get model from provider type: {type(llm).__name__}")
        
        # Queue for streaming
        token_queue = Queue()
        generation_done = {"done": False}
        
        def generate_with_streamer():
            """Generate tokens using the existing model with TextIteratorStreamer."""
            try:
                import torch
                from transformers import TextIteratorStreamer
                
                logger.info(f"🔧 Starting generation thread...")
                logger.info(f"   Model type: {type(model).__name__}")
                logger.info(f"   Max tokens: {max_tokens}")
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt")
                input_len = inputs["input_ids"].shape[1]
                logger.info(f"   Input tokens: {input_len}")
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    logger.info(f"   Moved to CUDA")
                
                # Create streamer
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                    timeout=300  # 5 min timeout per token
                )
                logger.info(f"   Streamer created")
                
                # Generation config
                gen_kwargs = {
                    **inputs,
                    "streamer": streamer,
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                
                # Start generation in background
                logger.info(f"   Starting model.generate()...")
                
                def run_generate():
                    try:
                        model.generate(**gen_kwargs)
                        logger.info(f"   model.generate() completed")
                    except Exception as e:
                        logger.error(f"   model.generate() failed: {e}")
                        import traceback
                        traceback.print_exc()
                
                gen_thread = Thread(target=run_generate)
                gen_thread.start()
                
                # Stream tokens from streamer
                token_count = 0
                logger.info(f"   Waiting for tokens from streamer...")
                
                for token in streamer:
                    token_count += 1
                    token_queue.put(("token", token))
                    if token_count <= 5:
                        logger.info(f"   Token {token_count}: '{token[:20]}...'")
                
                logger.info(f"   Streamer finished. Total tokens: {token_count}")
                gen_thread.join(timeout=10)
                token_queue.put(("done", None))
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                import traceback
                traceback.print_exc()
                token_queue.put(("error", str(e)))
            finally:
                generation_done["done"] = True
                logger.info(f"   Generation thread finished")
        
        # Start generation thread
        gen_thread = Thread(target=generate_with_streamer, daemon=True)
        gen_thread.start()
        
        # Stream tokens as they arrive
        in_thinking = False
        token_count = 0
        
        while True:
            try:
                event_type, data = token_queue.get(timeout=0.5)
                
                if event_type == "token":
                    token_count += 1
                    
                    # Skip empty tokens
                    if not data or not data.strip():
                        continue
                    
                    # Detect thinking tags (handle partial matches too)
                    if "<think>" in data or data.strip().startswith("<think"):
                        in_thinking = True
                        if include_thinking:
                            yield f"data: {json.dumps({'type': 'thinking_start', 'data': ''}, ensure_ascii=False)}\n\n"
                        # Don't output the tag itself
                        data = data.replace("<think>", "").replace("<think", "")
                        if not data.strip():
                            continue
                    
                    if "</think>" in data or data.strip().endswith("</think"):
                        in_thinking = False
                        if include_thinking:
                            yield f"data: {json.dumps({'type': 'thinking_end', 'data': ''}, ensure_ascii=False)}\n\n"
                        # Don't output the tag itself
                        data = data.replace("</think>", "").replace("</think", "")
                        if not data.strip():
                            continue
                    
                    # Skip language markers and garbage
                    if data.strip() in ['/zh', '/en', '/el', '/', '//', '/no_think', '/think']:
                        continue
                    
                    # Filter Chinese characters - skip tokens that are mostly Chinese
                    # This applies to both thinking and response (we want English/Greek thinking)
                    chinese_chars = sum(1 for c in data if '\u4e00' <= c <= '\u9fff')
                    if chinese_chars > 0 and chinese_chars > len(data.strip()) * 0.3:
                        logger.debug(f"   Filtering Chinese: {data[:30]}")
                        continue
                    
                    # Stream token
                    if in_thinking:
                        if include_thinking:
                            yield f"data: {json.dumps({'type': 'token', 'data': data}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'token', 'data': data}, ensure_ascii=False)}\n\n"
                    
                    # Give event loop a chance
                    await asyncio.sleep(0.01)
                
                elif event_type == "done":
                    break
                
                elif event_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'data': data}, ensure_ascii=False)}\n\n"
                    break
                    
            except Empty:
                if generation_done["done"]:
                    break
                # Send heartbeat to keep connection alive
                yield f"data: {json.dumps({'type': 'heartbeat', 'data': ''}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.1)
        
        gen_thread.join(timeout=5)
        
        logger.info(f"✅ Streamed {token_count} tokens")
        
        yield f"data: {json.dumps({'type': 'response_end', 'data': ''}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'data': ''}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"






# Updated endpoint
@app.post("/stream/chat")
async def stream_chat_post(request: StreamingChatRequest):
    """Stream chat with true real-time token output."""
    return StreamingResponse(
        generate_streaming_response(
            content=request.content,
            chat_id=request.chat_id,
            include_thinking=request.include_thinking,
            max_tokens=request.max_tokens or 256
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
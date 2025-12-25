# app/main.py
"""
FastAPI application entry point with authentication, streaming, and Greek language support.
FIXED: Network file indexing now properly triggers RAG ingestion.
FIXED: Streaming now saves messages to database.
FIXED: File upload/download support added.
"""

import asyncio
import json
import re
from fastapi import FastAPI, HTTPException, status, Query, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, List  # FIXED: Import List from typing, not ast
from pydantic import BaseModel
from pathlib import Path
from app.api.intelligence_routes import router as intelligence_router
# Import API routers
from app.api import config_router
from app.api.auth_routes import router as auth_router, get_current_user_dep
from app.api.chat_routes_authenticated import router as chat_router
from app.api.file_routes import router as file_router  # NEW: File routes
from app.api.logistics_routes import router as logistics_router
from app.api.workflow_routes import router as workflow_router
from app.workflows import initialize_workflow_system, shutdown_workflow_system
# Import models
from app.api import HealthResponse

# Import configuration
from app.config import SERVER, PATHS, config

# Import core modules
from app.core import startup_manager, ChatNotFoundException, RAGException

# Import database initialization
from app.db.init_db import init_database
from app.db.storage import storage

# Import file service
from app.services.file_service import file_service  # NEW: File service

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
        await initialize_workflow_system()
        logger.info("✅ Application ready")
        logger.info(f"   Language: Greek (Ελληνικά)")
        logger.info(f"   Streaming: Enabled")
        logger.info(f"   Thinking tags: Enabled")
        logger.info(f"   File upload: {'✅ Available' if file_service.is_available else '⚠️ Not available'}")
        
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
            await shutdown_workflow_system()
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
    title="Prometheus AI API",
    description="AI Agent API με υποστήριξη Ελληνικών, RAG, file upload, και streaming responses",
    version="2.2.0",
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
app.include_router(file_router)  # NEW: File routes
app.include_router(intelligence_router)
app.include_router(logistics_router)
app.include_router(workflow_router)
# =============================================================================
# Streaming Endpoints
# =============================================================================

from queue import Queue, Empty
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import time 


class StreamingChatRequest(BaseModel):
    """Request for streaming chat."""
    content: str
    chat_id: Optional[str] = None
    include_thinking: bool = False
    max_tokens: Optional[int] = None
    file_ids: Optional[List[str]] = None  # File attachment IDs


# Token patterns that indicate end of meaningful generation
STOP_TOKENS = {
    '<|im_end|>',
    '<|endoftext|>',
    '<|im_start|>',
    '</s>',
    '<s>',
    '[PAD]',
    '<pad>',
    '<eos>',
    '<|end|>',
    '[EOS]',
}

LANGUAGE_MARKERS = {'/zh', '/en', '/el', '/no_think', '/think', '//', '/'}


def should_filter_token(token: str) -> bool:
    """Check if token should be filtered from output."""
    token_stripped = token.strip()
    
    # Check exact matches
    if token_stripped in STOP_TOKENS:
        return True
    
    # Check if token contains stop tokens
    for stop in STOP_TOKENS:
        if stop in token:
            return True
    
    return False


def clean_token_for_display(token: str) -> str:
    """Clean token before displaying to user."""
    result = token
    
    # Remove stop tokens from within the token
    for stop in STOP_TOKENS:
        result = result.replace(stop, '')
    
    return result


def is_junk_token(token: str) -> bool:
    """Check if token is junk/padding that should be filtered."""
    stripped = token.strip()
    
    # Empty
    if not stripped:
        return True
    
    # Language markers
    if stripped in LANGUAGE_MARKERS:
        return True
    
    # EOS/padding tokens
    if stripped in STOP_TOKENS:
        return True
    
    # Chinese characters (Qwen's default thinking)
    chinese_count = sum(1 for c in stripped if '\u4e00' <= c <= '\u9fff')
    if chinese_count > 0 and chinese_count > len(stripped) * 0.3:
        return True
    
    return False


def is_eos_token(token: str) -> bool:
    """Check if token is an end-of-sequence marker."""
    stripped = token.strip()
    return stripped in STOP_TOKENS


async def generate_streaming_response(
    content: str,
    chat_id: Optional[str],
    include_thinking: bool,
    max_tokens: int,
    user_id: int,
    file_ids: Optional[List[str]] = None
):
    """
    TRUE streaming using the EXISTING agent's model.
    NOW WITH MESSAGE PERSISTENCE TO DATABASE.
    FIXED: Early stopping on EOS tokens, no trash padding.
    FIXED: File context support.
    """
    
    # Build file context if files provided
    file_context = ""
    if file_ids:
        for file_id in file_ids:
            file_meta = file_service.get_metadata(file_id)
            if file_meta and file_meta.extracted_content:
                file_context += f"""
<uploaded_file name="{file_meta.original_name}" type="{file_meta.content_type}">
{file_meta.extracted_content[:15000]}
</uploaded_file>
"""
    
    # Prepend file context to user message
    original_content = content
    if file_context:
        content = f"{file_context}\n\nUser message: {content}"
        
    from app.agent.integration import get_agent
    from app.core.interfaces import Context
    
    # =========================================================================
    # SAVE USER MESSAGE TO DATABASE FIRST
    # =========================================================================
    if chat_id and user_id:
        try:
            storage.add_message(chat_id, "user", original_content)
            logger.info(f"💬 Saved user message to chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to save user message: {e}")
    
    # Collect all response tokens for saving at the end
    full_response_tokens = []
    
    try:
        # Phase 1: Acknowledge
        yield f"data: {json.dumps({'type': 'status', 'data': '🔍 Αναζήτηση...'}, ensure_ascii=False)}\n\n"
        
        agent = get_agent()
        
        # Build context with chat history from database
        chat_history = []
        if chat_id:
            try:
                messages = storage.get_messages(chat_id, limit=10)
                # Exclude the message we just added (last one)
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in messages[:-1] if messages
                ]
                logger.info(f"📜 Loaded {len(chat_history)} messages from history")
            except Exception as e:
                logger.warning(f"Could not load chat history: {e}")
        
        context = Context(
            query=content,
            chat_history=chat_history,
            metadata={"chat_id": chat_id, "user_id": user_id},
            debug_info=[]
        )
        
        # Phase 2: Preprocessing (RAG, intent)
        logger.info(f"📥 Query: {original_content[:100]}...")
        
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
        
        # Add instruction for thinking language
        thinking_instruction = """

<instructions>
THINKING LANGUAGE: When using <think> tags, think in English or Greek - NEVER Chinese.
RESPONSE LANGUAGE: Always respond in Greek.
BREVITY: Stop when the answer is complete. Short questions = short answers.
</instructions>

"""
        prompt = thinking_instruction + prompt
        
        logger.info(f"📝 Prompt prepared ({len(prompt)} chars)")
        yield f"data: {json.dumps({'type': 'status', 'data': '💭 Σκέψη...'}, ensure_ascii=False)}\n\n"
        
        # Phase 3: Token-by-token streaming
        # Get the model/tokenizer from the LLM provider
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
            llm._ensure_initialized()
            model = llm.model
            tokenizer = llm.tokenizer
            logger.info("Using model from direct provider")
        elif hasattr(llm, 'model') and llm.model is not None:
            model = llm.model
            tokenizer = llm.tokenizer
            logger.info("Using already-initialized model")
        else:
            raise RuntimeError(f"Cannot get model from provider type: {type(llm).__name__}")
        
        token_queue = Queue()
        generation_done = {"done": False, "eos_hit": False}
        
        def generate_with_streamer():
            try:
                import torch
                from transformers import TextIteratorStreamer
                
                logger.info(f"🔧 Starting generation thread...")
                
                inputs = tokenizer(prompt, return_tensors="pt")
                input_len = inputs["input_ids"].shape[1]
                logger.info(f"   Input tokens: {input_len}")
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=False,  # Don't skip so we can detect EOS
                    timeout=300
                )
                
                gen_kwargs = {
                    **inputs,
                    "streamer": streamer,
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                
                def run_generate():
                    try:
                        model.generate(**gen_kwargs)
                    except Exception as e:
                        logger.error(f"   model.generate() failed: {e}")
                
                gen_thread = Thread(target=run_generate)
                gen_thread.start()
                
                token_count = 0
                consecutive_eos = 0
                
                for token in streamer:
                    token_count += 1
                    
                    # Check for EOS token - signal early stop
                    if is_eos_token(token):
                        consecutive_eos += 1
                        if consecutive_eos >= 1:  # Stop on first real EOS
                            logger.info(f"   EOS detected at token {token_count}, stopping early")
                            generation_done["eos_hit"] = True
                            break
                    else:
                        consecutive_eos = 0
                        token_queue.put(("token", token))
                
                logger.info(f"   Streamer finished. Total tokens: {token_count}, EOS hit: {generation_done['eos_hit']}")
                gen_thread.join(timeout=10)
                token_queue.put(("done", None))
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                token_queue.put(("error", str(e)))
            finally:
                generation_done["done"] = True
        
        # Start generation thread
        gen_thread = Thread(target=generate_with_streamer, daemon=True)
        gen_thread.start()
        
        # Stream tokens as they arrive
        in_thinking = False
        thinking_start_time = 0
        token_count = 0
        
        while True:
            try:
                event_type, data = token_queue.get(timeout=0.5)
                
                if event_type == "token":
                    token_count += 1
                    
                    # Skip junk tokens
                    if is_junk_token(data):
                        continue
                    
                    # Clean the token
                    data = clean_token_for_display(data)
                    if not data:
                        continue
                    
                    # Collect for saving (including thinking)
                    full_response_tokens.append(data)
                    
                    # Detect thinking tags
                    if "<think>" in data or data.strip().startswith("<think"):
                        in_thinking = True
                        thinking_start_time = time.time()
                        yield f"data: {json.dumps({'type': 'thinking_start', 'data': ''}, ensure_ascii=False)}\n\n"
                        data = data.replace("<think>", "").replace("<think", "")
                        if not data.strip():
                            continue
                    
                    if "</think>" in data or data.strip().endswith("</think"):
                        in_thinking = False
                        thinking_duration = int(time.time() - thinking_start_time) if thinking_start_time else 0
                        yield f"data: {json.dumps({'type': 'thinking_end', 'data': str(thinking_duration)}, ensure_ascii=False)}\n\n"
                        data = data.replace("</think>", "").replace("</think", "")
                        if not data.strip():
                            continue
                    
                    # Stream token with appropriate type
                    if in_thinking:
                        # Send thinking tokens with special type so frontend can handle them
                        yield f"data: {json.dumps({'type': 'thinking_token', 'data': data}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'token', 'data': data}, ensure_ascii=False)}\n\n"
                    
                    await asyncio.sleep(0.01)
                
                elif event_type == "done":
                    break
                
                elif event_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'data': data}, ensure_ascii=False)}\n\n"
                    break
                    
            except Empty:
                if generation_done["done"]:
                    break
                yield f"data: {json.dumps({'type': 'heartbeat', 'data': ''}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.1)
        
        gen_thread.join(timeout=5)
        
        # =====================================================================
        # SAVE ASSISTANT RESPONSE TO DATABASE
        # =====================================================================
        if chat_id and user_id and full_response_tokens:
            try:
                # Join all tokens
                complete_response = ''.join(full_response_tokens)
                
                # Remove thinking content for saved response
                complete_response = re.sub(r'<think>.*?</think>', '', complete_response, flags=re.DOTALL)
                complete_response = re.sub(r'<think.*', '', complete_response, flags=re.DOTALL)
                
                # Clean up language markers and stop tokens
                complete_response = re.sub(r'/(zh|en|el|think|no_think)\b', '', complete_response)
                for stop in STOP_TOKENS:
                    complete_response = complete_response.replace(stop, '')
                
                complete_response = complete_response.strip()
                
                if complete_response:
                    storage.add_message(chat_id, "assistant", complete_response)
                    logger.info(f"💬 Saved assistant response to chat {chat_id} ({len(complete_response)} chars)")
                else:
                    logger.warning(f"⚠️ Empty response, not saving to database")
            except Exception as e:
                logger.error(f"Failed to save assistant response: {e}")
        
        logger.info(f"✅ Streamed {token_count} tokens")
        
        yield f"data: {json.dumps({'type': 'response_end', 'data': ''}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'data': ''}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        logger.error(f"Stream error: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"


@app.post("/stream/chat")
async def stream_chat_post(
    request: StreamingChatRequest,
    current_user: dict = Depends(get_current_user_dep)
):
    """Stream chat with message persistence and file support."""
    # Verify chat belongs to user
    if request.chat_id:
        chat = storage.get_chat(request.chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        if chat["user_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
    
    return StreamingResponse(
        generate_streaming_response(
            content=request.content,
            chat_id=request.chat_id,
            include_thinking=request.include_thinking,
            max_tokens=request.max_tokens or 4096,  # FIXED: Higher default
            user_id=current_user["id"],
            file_ids=request.file_ids  # FIXED: Pass file_ids
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
        "name": "Prometheus AI API",
        "version": "2.2.0",
        "language": "Greek (Ελληνικά)",
        "status": "running",
        "features": {
            "authentication": True,
            "chat_persistence": True,
            "rag": True,
            "streaming": True,
            "thinking_tags": True,
            "greek_language": True,
            "network_filesystem": True,
            "file_upload": True,
            "file_generation": True
        },
        "endpoints": {
            "auth": "/auth",
            "chats": "/chats",
            "stream": "/stream/chat",
            "files": "/files",
            "config": "/config",
            "rag_status": "/rag/status",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
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
        "file_storage": file_service.is_available,
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
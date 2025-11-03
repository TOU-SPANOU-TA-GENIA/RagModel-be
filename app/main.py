from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .models import (
    NewChatRequest, MessageRequest, ChatResponse, UploadResponse,
    ErrorResponse, HealthResponse, StatsResponse, IngestionResponse,
    ChatSummary, ChatDetail, SourceDocument
)
from .chat_manager import (
    create_chat, list_chats, append_message, get_history, get_chat,
    update_chat_title, delete_chat, get_stats
)
from .vectorstore import retrieve, init_vectorstore
from .ingestion import save_upload, ingest_single_file, ingest_directory
from .llm import generate_answer, build_prompt
from .config import CORS_ORIGINS, SYSTEM_INSTRUCTION, KNOWLEDGE_DIR
from .exceptions import (
    ChatNotFoundException, VectorStoreNotInitializedException,
    ModelLoadException, RAGException
)
from .logger import setup_logger

logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG application...")
    
    try:
        init_vectorstore()
        logger.info("Vector store initialized")
    except VectorStoreNotInitializedException:
        logger.warning("Vector store not found. Please run ingestion.")
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
    
    yield
    
    logger.info("Shutting down RAG application...")

app = FastAPI(
    title="Assistant RAG API",
    description="Retrieval-Augmented Generation system for support",
    version="1.0.0",
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

@app.exception_handler(ChatNotFoundException)
async def chat_not_found_handler(request, exc: ChatNotFoundException):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Chat not found", "detail": str(exc)}
    )

@app.exception_handler(VectorStoreNotInitializedException)
async def vector_store_not_initialized_handler(request, exc: VectorStoreNotInitializedException):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Vector store not initialized",
            "detail": str(exc),
            "hint": "Please upload documents and run ingestion first"
        }
    )

@app.exception_handler(RAGException)
async def rag_exception_handler(request, exc: RAGException):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "RAG system error", "detail": str(exc)}
    )

@app.get("/stats", response_model=StatsResponse)
async def get_system_stats():
    stats = get_stats()
    
    vector_store_initialized = False
    try:
        init_vectorstore()
        vector_store_initialized = True
    except:
        pass
    
    return {
        **stats,
        "vector_store_initialized": vector_store_initialized
    }

@app.get("/ping")
async def ping():
    return {"ok": True, "message": "ping"}

@app.post("/chats", response_model=str, status_code=status.HTTP_201_CREATED)
async def create_new_chat(req: NewChatRequest):
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
    try:
        chat = get_chat(chat_id)
        return chat
    except ChatNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@app.post("/chats/{chat_id}/message", response_model=ChatResponse)
async def post_message_to_chat(chat_id: str, msg: MessageRequest):
    try:
        # Validate chat exists
        _ = get_chat(chat_id)
        
        # Store user message
        append_message(chat_id, msg.role, msg.content)
        logger.info(f"User message received for chat {chat_id}")
        
        # Retrieve relevant documents
        retrieved_docs = retrieve(msg.content, k=3)
        
        # Format context
        context_parts = []
        sources = []
        
        for content, metadata, score in retrieved_docs:
            source_name = metadata.get("source", "Unknown")
            context_parts.append(f"[{source_name}]\n{content}")
            
            sources.append(SourceDocument(
                content=content[:200] + "..." if len(content) > 200 else content,
                source=source_name,
                relevance_score=round(score, 3)
            ))
        
        context_text = "\n\n".join(context_parts)
        
        # Get conversation history (exclude the just-added user message)
        history = get_history(chat_id)[:-1]
        
        # Build prompt
        prompt = build_prompt(
            system_instruction=SYSTEM_INSTRUCTION,
            context=context_text,
            history=history,
            user_query=msg.content
        )
        
        logger.debug(f"Prompt built ({len(prompt)} chars)")
        
        # Generate answer
        answer = generate_answer(prompt)
        
        # Store assistant response
        append_message(chat_id, "assistant", answer)
        
        logger.info(f"Response generated for chat {chat_id}")
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            metadata={
                "chat_id": chat_id,
                "num_sources": len(sources),
                "has_context": len(context_text) > 0
            }
        )
    
    except ChatNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    except VectorStoreNotInitializedException:
        logger.warning("Vector store not available, generating without context")
        
        history = get_history(chat_id)[:-1]
        prompt = build_prompt(
            system_instruction=SYSTEM_INSTRUCTION,
            context="",
            history=history,
            user_query=msg.content
        )
        
        answer = generate_answer(prompt)
        append_message(chat_id, "assistant", answer)
        
        return ChatResponse(
            answer=answer,
            sources=[],
            metadata={"warning": "No knowledge base available"}
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )


@app.get("/chats/{chat_id}/history")
async def get_chat_history(chat_id: str):
    try:
        history = get_history(chat_id)
        return {"chat_id": chat_id, "messages": history}
    except ChatNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )

@app.patch("/chats/{chat_id}/title")
async def update_title(chat_id: str, new_title: str):
    try:
        update_chat_title(chat_id, new_title)
        return {"success": True, "message": "Title updated"}
    except ChatNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )

@app.delete("/chats/{chat_id}")
async def delete_chat_endpoint(chat_id: str):
    try:
        delete_chat(chat_id)
        return {"success": True, "message": "Chat deleted"}
    except ChatNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )

@app.post("/chat", response_model=ChatResponse)
async def simple_chat(msg: MessageRequest):
    try:
        retrieved_docs = retrieve(msg.content, k=3)
        
        context_parts = []
        sources = []
        
        for content, metadata, score in retrieved_docs:
            source_name = metadata.get("source", "Unknown")
            context_parts.append(f"[{source_name}]\n{content}")
            
            sources.append(SourceDocument(
                content=content[:200] + "..." if len(content) > 200 else content,
                source=source_name,
                relevance_score=round(score, 3)
            ))
        
        context_text = "\n\n".join(context_parts)
        
        prompt = build_prompt(
            system_instruction=SYSTEM_INSTRUCTION,
            context=context_text,
            history=[],
            user_query=msg.content
        )
        
        answer = generate_answer(prompt)
        
        return ChatResponse(
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        logger.error(f"Error in simple chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )


# Upload a new file to the knowledge base.
# Note: File is saved but not automatically indexed.
# Call /ingest/file/{filename} to index it.
@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)):
    try:
        filepath = await save_upload(file)
        logger.info(f"File uploaded: {file.filename}")
        
        return UploadResponse(
            filename=file.filename,
            status="uploaded",
            message=f"File saved to {filepath}. Call /ingest/file/{file.filename} to index it."
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )


@app.post("/ingest/file/{filename}", response_model=IngestionResponse)
async def ingest_file(filename: str):
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

# Ingest all documents from the knowledge directory.
@app.post("/ingest/all", response_model=IngestionResponse)
async def ingest_all_documents(rebuild: bool = False):
    try:
        logger.info(f"Starting full ingestion (rebuild={rebuild})")
        result = ingest_directory(KNOWLEDGE_DIR, rebuild=rebuild)
        
        if result["success"]:
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

# Search the vector store directly without generating an answer.
# Useful for testing retrieval quality.
@app.get("/search")
async def search_documents(q: str, k: int = 3):
    try:
        if not q or not q.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query parameter 'q' is required"
            )
        
        results = retrieve(q, k=k)
        
        return {
            "query": q,
            "num_results": len(results),
            "results": [
                {
                    "content": content,
                    "source": metadata.get("source", "Unknown"),
                    "relevance_score": round(score, 3)
                }
                for content, metadata, score in results
            ]
        }
    except VectorStoreNotInitializedException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized"
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )
        

@app.get("/health", response_model=HealthResponse)
async def health_check():
    from datetime import datetime
    from pathlib import Path
    from .gpu_utils import get_gpu_info
    
    vector_store_ok = False
    try:
        init_vectorstore()
        vector_store_ok = True
    except:
        pass
    
    knowledge_dir_ok = KNOWLEDGE_DIR.exists()
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    services = {
        "vector_store": vector_store_ok,
        "knowledge_directory": knowledge_dir_ok,
        "llm": True
    }
    
    if gpu_info:
        services["gpu"] = True
        services["gpu_info"] = gpu_info
    else:
        services["gpu"] = False
    
    return {
        "status": "healthy" if vector_store_ok else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "services": services
    }
# app/main.py
"""
Modified to include Agent capabilities.

Changes from original:
1. Added agent import and initialization
2. Modified /chats/{chat_id}/message to use agent
3. Modified /chat (simple endpoint) to use agent
4. Added /agent/capabilities endpoint for debugging
5. Preserved all existing functionality
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .models import (
    NewChatRequest, MessageRequest, ChatResponse, UploadResponse,
    ErrorResponse, HealthResponse, StatsResponse, IngestionResponse,
    ChatSummary, ChatDetail, SourceDocument,
    # New model for agent response
    AgentChatResponse
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

# NEW: Import agent
from .agent import create_agent, Agent

logger = setup_logger(__name__)

# Global agent instance
_agent: Agent = None


def get_agent() -> Agent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


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
    
    # NEW: Initialize agent
    try:
        get_agent()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
    
    yield
    
    logger.info("Shutting down RAG application...")

app = FastAPI(
    title="AI Agent RAG API",  # Updated title
    description="Retrieval-Augmented Generation system with AI Agent capabilities",
    version="2.0.0",  # Bumped version
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
# Exception Handlers (unchanged)
# ============================================================================

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

# ============================================================================
# System Endpoints (unchanged)
# ============================================================================

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

# ============================================================================
# NEW: Agent Capabilities Endpoint
# ============================================================================

@app.get("/agent/capabilities")
async def get_agent_capabilities():
    """
    Get information about agent capabilities and available tools.
    Useful for debugging and understanding what the agent can do.
    """
    try:
        agent = get_agent()
        tool_registry = agent.tool_registry
        
        tools_info = []
        for tool_name in tool_registry.list_tools():
            tool = tool_registry.get_tool(tool_name)
            if tool:
                tools_info.append(tool.get_schema())
        
        return {
            "agent_version": "2.0.0",
            "capabilities": {
                "rag_retrieval": True,
                "tool_usage": True,
                "multi_turn_conversation": True,
                "intent_classification": True
            },
            "available_tools": tools_info,
            "intents": ["question", "action", "conversation"]
        }
    except Exception as e:
        logger.error(f"Error getting agent capabilities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get agent capabilities"
        )

# ============================================================================
# Chat Management Endpoints (unchanged)
# ============================================================================

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

# ============================================================================
# MODIFIED: Chat Message Endpoint with Agent
# ============================================================================

@app.post("/chats/{chat_id}/message", response_model=AgentChatResponse)
async def post_message_to_chat(chat_id: str, msg: MessageRequest):
    """
    Send a message to a chat and get an agent response.
    
    The agent will:
    1. Analyze the message intent
    2. Use tools if an action is requested
    3. Retrieve relevant knowledge from RAG
    4. Generate a contextual response
    """
    try:
        # Validate chat exists
        _ = get_chat(chat_id)
        
        # Store user message
        append_message(chat_id, msg.role, msg.content)
        logger.info(f"User message received for chat {chat_id}")
        
        # Get conversation history (exclude the just-added user message)
        history = get_history(chat_id)[:-1]
        
        # NEW: Use agent to process the query
        agent = get_agent()
        agent_response = agent.process_query(
            user_query=msg.content,
            chat_history=history,
            use_rag=True
        )
        
        # Extract response components
        answer = agent_response["answer"]
        sources = agent_response.get("sources", [])
        tool_used = agent_response.get("tool_used")
        tool_result = agent_response.get("tool_result")
        decision = agent_response.get("decision", {})
        metadata = agent_response.get("metadata", {})
        
        # Store assistant response
        append_message(chat_id, "assistant", answer)
        
        logger.info(f"Agent response generated for chat {chat_id}")
        logger.info(f"  Intent: {metadata.get('intent')}")
        logger.info(f"  Tool used: {tool_used}")
        logger.info(f"  RAG sources: {len(sources)}")
        
        # Convert sources to SourceDocument format
        source_docs = [
            SourceDocument(
                content=src["content"],
                source=src["source"],
                relevance_score=src["relevance_score"]
            )
            for src in sources
        ]
        
        return AgentChatResponse(
            answer=answer,
            sources=source_docs,
            tool_used=tool_used,
            tool_result=tool_result,
            metadata={
                "chat_id": chat_id,
                "intent": metadata.get("intent"),
                "used_rag": metadata.get("used_rag", False),
                "used_tool": metadata.get("used_tool", False),
                "num_sources": len(sources),
                "decision": decision
            }
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

# ============================================================================
# MODIFIED: Simple Chat Endpoint with Agent
# ============================================================================

@app.post("/chat", response_model=AgentChatResponse)
async def simple_chat(msg: MessageRequest):
    """
    Stateless chat endpoint using agent.
    No conversation history maintained.
    """
    try:
        agent = get_agent()
        agent_response = agent.process_query(
            user_query=msg.content,
            chat_history=[],
            use_rag=True
        )
        
        answer = agent_response["answer"]
        sources = agent_response.get("sources", [])
        tool_used = agent_response.get("tool_used")
        tool_result = agent_response.get("tool_result")
        metadata = agent_response.get("metadata", {})
        
        source_docs = [
            SourceDocument(
                content=src["content"],
                source=src["source"],
                relevance_score=src["relevance_score"]
            )
            for src in sources
        ]
        
        return AgentChatResponse(
            answer=answer,
            sources=source_docs,
            tool_used=tool_used,
            tool_result=tool_result,
            metadata={
                "intent": metadata.get("intent"),
                "used_rag": metadata.get("used_rag", False),
                "used_tool": metadata.get("used_tool", False)
            }
        )
    
    except Exception as e:
        logger.error(f"Error in simple chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )

# ============================================================================
# History and Management Endpoints (unchanged)
# ============================================================================

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

# ============================================================================
# Document Upload and Ingestion (unchanged)
# ============================================================================

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

# ============================================================================
# Health Check (updated with agent info)
# ============================================================================

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
    
    # Check agent
    agent_ok = False
    try:
        get_agent()
        agent_ok = True
    except:
        pass
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    services = {
        "vector_store": vector_store_ok,
        "knowledge_directory": knowledge_dir_ok,
        "llm": True,
        "agent": agent_ok  # NEW
    }
    
    if gpu_info:
        services["gpu"] = True
        services["gpu_info"] = gpu_info
    else:
        services["gpu"] = False
    
    return {
        "status": "healthy" if (vector_store_ok and agent_ok) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "services": services
    }
# app/api/streaming_routes.py
"""
Streaming API endpoints for real-time response generation.
Uses Server-Sent Events (SSE) for token-by-token delivery.
FIXED: Now saves messages to database for persistence.
"""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.auth_routes import get_current_user_dep
from app.db.storage import storage
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/stream", tags=["streaming"])


class StreamingChatRequest(BaseModel):
    """Request body for streaming chat."""
    content: str
    chat_id: Optional[str] = None
    include_thinking: bool = False


async def event_generator(
    content: str,
    chat_id: Optional[str],
    user_id: int,
    include_thinking: bool = False
):
    """
    Async generator that yields SSE-formatted events.
    Collects full response and saves to database at the end.
    """
    from app.llm.streaming_provider import create_streaming_provider
    from app.core.interfaces import Context
    
    # Collect the full response for saving
    full_response = []
    
    try:
        # Save user message to database FIRST
        if chat_id:
            storage.add_message(chat_id, "user", content)
            logger.info(f"💬 Saved user message to chat {chat_id}")
        
        # Build context
        context = Context(query=content)
        context.metadata["user_id"] = user_id
        context.metadata["chat_id"] = chat_id
        
        # Get chat history for context
        if chat_id:
            messages = storage.get_messages(chat_id, limit=10)
            context.chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages[:-1]  # Exclude the message we just added
            ]
        
        # Get orchestrator for preprocessing (RAG, intent, etc.)
        orchestrator = _get_orchestrator()
        
        # Run pre-generation steps
        context = await asyncio.to_thread(
            orchestrator.run_preprocessing,
            context
        )
        
        # Get the prepared prompt
        prompt = context.metadata.get("prompt", content)
        
        # Create streaming provider
        streaming_llm = create_streaming_provider()
        
        # Stream tokens and collect response
        async for event in streaming_llm.generate_stream(
            prompt, 
            include_thinking=include_thinking
        ):
            # Collect tokens for the full response
            if hasattr(event, 'event_type'):
                from app.llm.streaming.events import StreamEventType
                if event.event_type == StreamEventType.TOKEN:
                    full_response.append(event.data)
            
            yield event.to_sse()
            await asyncio.sleep(0.01)
        
        # Save assistant response to database AFTER streaming completes
        if chat_id and full_response:
            complete_response = ''.join(full_response)
            storage.add_message(chat_id, "assistant", complete_response)
            logger.info(f"💬 Saved assistant response to chat {chat_id} ({len(complete_response)} chars)")
        
        # Send done event
        done_event = {"type": "done", "data": ""}
        yield f"data: {json.dumps(done_event)}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_event = {"type": "error", "data": str(e)}
        yield f"data: {json.dumps(error_event)}\n\n"


def _get_orchestrator():
    """Get or create orchestrator instance."""
    from app.agent.orchestrator import SimpleAgentOrchestrator
    return SimpleAgentOrchestrator()


@router.post("/chat")
async def stream_chat(
    request: StreamingChatRequest,
    current_user: dict = Depends(get_current_user_dep)
):
    """
    Stream chat response token-by-token with message persistence.
    
    Messages are saved to the database:
    - User message: saved immediately before streaming starts
    - Assistant response: saved after streaming completes
    """
    # Verify chat belongs to user if chat_id provided
    if request.chat_id:
        chat = storage.get_chat(request.chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        if chat["user_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="Access denied")
    
    return StreamingResponse(
        event_generator(
            content=request.content,
            chat_id=request.chat_id,
            user_id=current_user["id"],
            include_thinking=request.include_thinking
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
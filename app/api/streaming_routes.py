# app/api/streaming_routes.py
"""
Streaming API endpoints for real-time response generation.
Uses Server-Sent Events (SSE) for token-by-token delivery.
"""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/stream", tags=["streaming"])


class StreamingChatRequest(BaseModel):
    """Request body for streaming chat."""
    content: str
    chat_id: Optional[str] = None
    include_thinking: bool = False  # Whether to stream thinking tokens


async def event_generator(
    content: str,
    chat_id: Optional[str],
    user_id: str,
    include_thinking: bool = False
):
    """
    Async generator that yields SSE-formatted events.
    
    Event format:
    data: {"type": "token|thinking_start|thinking_end|response_start|response_end|done|error", "data": "..."}
    
    """
    from app.llm.streaming_provider import create_streaming_provider, StreamEventType
    from app.agent.orchestrator import SimpleAgentOrchestrator
    from app.core.interfaces import Context
    
    try:
        # Build context
        context = Context(query=content)
        context.metadata["user_id"] = user_id
        context.metadata["chat_id"] = chat_id
        
        # Get orchestrator for preprocessing (RAG, intent, etc.)
        # This runs synchronously before streaming
        orchestrator = _get_orchestrator()
        
        # Run pre-generation steps (intent classification, RAG retrieval, etc.)
        context = await asyncio.to_thread(
            orchestrator.run_preprocessing,
            context
        )
        
        # Get the prepared prompt
        prompt = context.metadata.get("prompt", content)
        
        # Create streaming provider
        streaming_llm = create_streaming_provider()
        
        # Stream tokens
        async for event in streaming_llm.generate_stream(
            prompt, 
            include_thinking=include_thinking
        ):
            yield event.to_sse()
            
            # Small delay to prevent overwhelming client
            await asyncio.sleep(0.01)
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_event = {
            "type": "error",
            "data": str(e)
        }
        yield f"data: {json.dumps(error_event)}\n\n"


@router.post("/chat")
async def stream_chat(
    request: StreamingChatRequest,
    # current_user = Depends(get_current_user)  # Add auth when ready
):
    """
    Stream chat response token-by-token.
    
    Uses Server-Sent Events (SSE) format:
    - Each token is sent as: `data: {"type": "token", "data": "..."}`
    - Special events: thinking_start, thinking_end, response_start, response_end, done, error
    
    **Client Usage (JavaScript):**
    ```javascript
    const eventSource = new EventSource('/stream/chat');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
            // Append to response display
            responseDiv.textContent += data.data;
        } else if (data.type === 'done') {
            eventSource.close();
        }
    };
    ```
    
    **Or with fetch:**
    ```javascript
    const response = await fetch('/stream/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({content: 'Hello!'})
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        // Parse SSE format and handle tokens
    }
    ```
    """
    # Temporary user ID until auth is integrated
    user_id = "anonymous"  # Replace with: current_user.id
    
    return StreamingResponse(
        event_generator(
            content=request.content,
            chat_id=request.chat_id,
            user_id=user_id,
            include_thinking=request.include_thinking
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/chat")
async def stream_chat_get(
    content: str = Query(..., description="Message content"),
    chat_id: Optional[str] = Query(None, description="Chat ID"),
    include_thinking: bool = Query(False, description="Include thinking in stream")
):
    """
    GET endpoint for SSE-based streaming.
    
    Useful for EventSource API which only supports GET:
    ```javascript
    const eventSource = new EventSource(
        '/stream/chat?content=Hello&chat_id=abc123'
    );
    ```
    """
    user_id = "anonymous"
    
    return StreamingResponse(
        event_generator(
            content=content,
            chat_id=chat_id,
            user_id=user_id,
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
# Helper to get orchestrator with preprocessing support
# =============================================================================

def _get_orchestrator():
    """Get or create orchestrator instance."""
    from app.agent.orchestrator import SimpleAgentOrchestrator
    
    # Use global instance if available
    if not hasattr(_get_orchestrator, '_instance'):
        _get_orchestrator._instance = SimpleAgentOrchestrator()
    
    return _get_orchestrator._instance


# =============================================================================
# Extension to SimpleAgentOrchestrator for preprocessing
# =============================================================================

def add_preprocessing_to_orchestrator():
    """
    Monkey-patch to add preprocessing method to orchestrator.
    Call this during app startup.
    """
    from app.agent.orchestrator import SimpleAgentOrchestrator
    
    def run_preprocessing(self, context):
        """
        Run preprocessing steps (everything except LLM generation).
        Returns context with prompt ready for streaming.
        """
        from app.core.interfaces import Context
        
        # Run each step except LLM generation
        for step in self.pipeline.steps:
            if step.name == "LLM Generation":
                continue  # Skip - we'll stream this
            context = step.process(context)
        
        return context
    
    # Add method to class
    SimpleAgentOrchestrator.run_preprocessing = run_preprocessing
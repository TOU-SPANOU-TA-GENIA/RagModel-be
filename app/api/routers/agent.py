import json
import asyncio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.auth.service import get_current_user
from app.db.models import User, Message # Ensure Message model is imported
from app.llm import llm_service
from app.rag.service import rag_service

router = APIRouter()

async def event_generator(chat_id: str, messages: list, db: Session):
    # 1. RAG Context Injection
    user_query = messages[-1]['content']
    context_snippets = rag_service.search(user_query, limit=3)
    
    if context_snippets:
        context_str = "\n".join(context_snippets)
        messages.insert(0, {
            "role": "system", 
            "content": f"Use this document context to help answer:\n{context_str}"
        })

    # Save the User's Message to DB immediately
    user_msg = Message(chat_id=chat_id, role="user", content=user_query)
    db.add(user_msg)
    db.commit()

    yield f"data: {json.dumps({'type': 'status', 'data': 'connected'})}\n\n"

    is_thinking = False
    full_buffer = ""
    full_assistant_response = ""

    try:
        async for chunk in llm_service.stream_response(messages):
            if not chunk: continue
            full_buffer += chunk

            if "<think>" in full_buffer and not is_thinking:
                is_thinking = True
                yield f"data: {json.dumps({'type': 'thinking_start', 'data': ''})}\n\n"
                full_buffer = full_buffer.replace("<think>", "")

            if "</think>" in full_buffer:
                parts = full_buffer.split("</think>")
                if parts[0] and is_thinking:
                    yield f"data: {json.dumps({'type': 'thinking_token', 'data': parts[0]})}\n\n"
                yield f"data: {json.dumps({'type': 'thinking_end', 'data': ''})}\n\n"
                is_thinking = False
                full_buffer = parts[1] if len(parts) > 1 else ""
                continue

            if is_thinking:
                yield f"data: {json.dumps({'type': 'thinking_token', 'data': full_buffer})}\n\n"
                full_buffer = ""
            else:
                if full_buffer:
                    full_assistant_response += full_buffer # Track for DB saving
                    yield f"data: {json.dumps({'type': 'token', 'data': full_buffer})}\n\n"
                    full_buffer = ""
            
            await asyncio.sleep(0.001) # Faster flush

        # 2. SAVE ASSISTANT RESPONSE TO DB
        if full_assistant_response:
            db_assistant_msg = Message(
                chat_id=chat_id, 
                role="assistant", 
                content=full_assistant_response
            )
            db.add(db_assistant_msg)
            db.commit()

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    yield f"data: {json.dumps({'type': 'done', 'data': ''})}\n\n"

@router.post("/chat")
async def stream_chat(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    body = await request.json()
    content = body.get("content") or body.get("message")
    chat_id = body.get("chat_id") or body.get("chatId")
    
    # NEW: General approach to support automation tools
    stream_mode = body.get("stream", True) 

    formatted_messages = [{"role": "user", "content": content}]

    if stream_mode:
        return StreamingResponse(
            event_generator(str(chat_id), formatted_messages, db),
            media_type="text/event-stream"
        )
    else:
        # Automation Mode: Consume the generator and return full JSON
        full_response = ""
        # We manually iterate the generator to build the response
        async for event in event_generator(str(chat_id), formatted_messages, db):
            # Parse the SSE format: "data: {json}\n\n"
            clean_line = event.strip().replace("data: ", "")
            try:
                data_obj = json.loads(clean_line)
                if data_obj['type'] == 'token':
                    full_response += data_obj['data']
            except:
                continue
        
        return JSONResponse(content={"response": full_response})
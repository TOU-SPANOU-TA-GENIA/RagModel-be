# app/llm/context_aware_prompt_builder.py
"""
Context-aware prompt builder that includes conversation memory.
"""

from typing import Dict, Any, List, Optional
from app.core.interfaces import PromptBuilder, Context
from app.core.conversation_memory import conversation_memory
from app.core.context_builder import context_builder
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ContextAwarePromptBuilder(PromptBuilder):
    """
    Prompt builder that maintains conversation context across messages.
    """
    
    def __init__(self, system_instruction: str, tools: Optional[Dict[str, Any]] = None):
        self.system_instruction = system_instruction
        self.tools = tools or {}
    
    def build(self, context: Context, **kwargs) -> str:
        """Build prompt with full conversation context."""
        # Get or create conversation session
        session_id = context.metadata.get("session_id")
        session = conversation_memory.get_or_create_session(session_id)
        
        # Store current query in conversation history
        session.add_message("user", context.query)
        
        # Build comprehensive conversation context
        conversation_context = context_builder.build_conversation_context(
            session, context.query
        )
        
        # Build the prompt
        sections = []
        
        # System instruction with conversation awareness
        system_section = self._build_system_section()
        sections.append(system_section)
        
        # Conversation context
        if conversation_context:
            sections.append(f"<conversation_context>\n{conversation_context}\n</conversation_context>")
        
        # RAG context
        rag_context = kwargs.get("rag_context", "")
        if rag_context:
            sections.append(f"<knowledge_base>\n{rag_context}\n</knowledge_base>")
        
        # Tool results
        tool_result = kwargs.get("tool_result")
        if tool_result:
            tool_text = self._format_tool_result(tool_result)
            sections.append(f"<tool_results>\n{tool_text}\n</tool_results>")
        
        # Current query (already in conversation context, but include for clarity)
        sections.append(f"<current_query>\nUser: {context.query}\n</current_query>")
        
        # Response instruction
        sections.append("\nAssistant:")
        
        full_prompt = "\n\n".join(sections)
        
        # Store session ID in context for response storage
        context.metadata["session_id"] = session.session_id
        
        logger.debug(f"Built context-aware prompt with {len(session.messages)} previous messages")
        return full_prompt
    
    def _build_system_section(self) -> str:
        """Build system section with stronger conversation awareness."""
        base_instruction = self.system_instruction
        
        # Stronger conversation awareness
        conversation_awareness = """
        
## CONVERSATION AWARENESS - CRITICAL INSTRUCTIONS:

You are having a continuous conversation. You MUST:

1. MAINTAIN CONTEXT: Remember everything from previous messages in this conversation
2. FOLLOW INSTRUCTIONS EXACTLY: If the user gives you specific instructions (like "when I say X, you answer Y"), you MUST follow them precisely every time
3. BE CONSISTENT: Don't change your behavior or forget instructions
4. PRIORITIZE USER RULES: User instructions override any default behavior
5. NO DEVIATION: When a user instruction matches, respond exactly as instructed without adding extra commentary

## EXAMPLE:
If user says: "when I say 'hello' you respond 'hi there!'"
Then when user says: "hello"
You MUST respond: "hi there!" (exactly, no extra text)

FAILURE TO FOLLOW THESE INSTRUCTIONS IS NOT ACCEPTABLE.
"""
        return f"<system>\n{base_instruction}{conversation_awareness}\n</system>"
    
    def _format_tool_result(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result for prompt."""
        if tool_result.get("success"):
            data = tool_result.get("data", {})
            
            # Handle multiple file matches
            if "action_required" in data and data["action_required"] == "choose_file":
                matches_text = "\n".join(
                    f"  {i+1}. {m['name']} ({m['path']})"
                    for i, m in enumerate(data.get("matches", []))
                )
                return f"""Multiple files found matching the query:
    {matches_text}

    Please ask the user which specific file they want to read."""
            
            # Handle successful file read
            if "content" in data:
                content = data["content"]
                return f"""File read successfully: {data.get('file_name', 'unknown')}
    Size: {data.get('size_bytes', 0)} bytes
    Lines: {data.get('lines', 0)}

    Content:
    {content}

    Please provide this information to the user in a clear, helpful way."""
            
            return f"Tool executed successfully:\n{data}"
        else:
            error = tool_result.get("error", "Unknown error")
            return f"""Tool execution FAILED with error: {error}

    You MUST inform the user that the file operation failed and explain why.
    Do NOT pretend it succeeded. Do NOT ask the user for the file contents."""
# app/llm/robust_prompt_builder.py
"""
Robust prompt builder that works across all conversation scenarios.
"""

from typing import Dict, Any, Optional
from app.core.interfaces import PromptBuilder, Context
from app.core.advanced_conversation_memory import advanced_memory
from app.core.advanced_context_builder import advanced_context_builder
from app.logger import setup_logger

logger = setup_logger(__name__)

class RobustPromptBuilder(PromptBuilder):
    """
    Robust prompt builder that handles all conversation scenarios.
    """
    
    def __init__(self, system_instruction: str, tools: Optional[Dict[str, Any]] = None):
        self.system_instruction = system_instruction
        self.tools = tools or {}
    
    def build(self, context: Context, **kwargs) -> str:
        """Build robust prompt for any scenario."""
        # Get or create conversation session
        session_id = context.metadata.get("session_id")
        session = advanced_memory.get_or_create_session(session_id)
        
        # Add current message to conversation history
        session.add_message("user", context.query)
        
        # Build comprehensive context
        conversation_context = advanced_context_builder.build_context_for_query(
            session, context.query
        )
        
        # Build prompt sections
        sections = []
        
        # Core system instruction
        sections.append(self._build_robust_system_section())
        
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
            sections.append(f"<tool_results>\n{self._format_tool_result(tool_result)}\n</tool_results>")
        
        # Response instruction
        sections.append("\nAssistant:")
        
        full_prompt = "\n\n".join(sections)
        
        # Store session ID for response tracking
        context.metadata["session_id"] = session.session_id
        
        logger.debug(f"Built robust prompt with {len(session.messages)} messages in session")
        return full_prompt
    
    def _build_robust_system_section(self) -> str:
        """Build system section that works for all scenarios."""
        base_instruction = self.system_instruction
        
        robustness_rules = """
        
## CONVERSATION ROBUSTNESS RULES:

1. **CONTEXT AWARENESS**: Maintain full conversation context across all messages
2. **INSTRUCTION FOLLOWING**: Follow user instructions precisely and consistently
3. **CONSISTENCY**: Maintain consistent personality and behavior
4. **ADAPTABILITY**: Adapt to user's communication style and preferences
5. **MEMORY**: Remember user facts, preferences, and previous instructions
6. **NATURAL FLOW**: Maintain natural conversation flow while following rules

## CRITICAL BEHAVIORS:
- If user gives specific response instructions, follow them EXACTLY every time
- Remember user information across the conversation
- Maintain topic continuity
- Adapt to user's tone and style
- Be reliable and predictable in following instructions
"""
        return f"<system>\n{base_instruction}{robustness_rules}\n</system>"
    
    def _format_tool_result(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result."""
        if tool_result.get("success"):
            data = tool_result.get("data", {})
            return f"Tool execution successful: {data}"
        else:
            return f"Tool execution failed: {tool_result.get('error', 'Unknown error')}"
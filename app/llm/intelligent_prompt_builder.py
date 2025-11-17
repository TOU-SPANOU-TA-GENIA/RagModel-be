# app/llm/intelligent_prompt_builder.py
"""
Intelligent prompt builder with context relevance filtering.
Uses dynamic context selection instead of including everything.
"""

from typing import Dict, Any, Optional, List
from app.core.interfaces import PromptBuilder, Context
from app.core.conversation_memory import conversation_memory
from app.core.context_relevance_filter import context_filter
from app.logger import setup_logger

logger = setup_logger(__name__)

class IntelligentPromptBuilder(PromptBuilder):
    """
    Prompt builder that intelligently selects relevant context.
    Prevents context bleeding and improves response quality.
    """
    
    def __init__(self, system_instruction: str, tools: Optional[Dict[str, Any]] = None):
        self.system_instruction = system_instruction
        self.tools = tools or {}
    
    def build(self, context: Context, **kwargs) -> str:
        """Build prompt with intelligently filtered context."""
        # Get or create conversation session
        session_id = context.metadata.get("session_id")
        session = conversation_memory.get_or_create_session(session_id)
        
        # Don't add current message here - it's already in context.query
        # session.add_message("user", context.query)  # REMOVED
        
        # Filter context based on query relevance
        filtered = context_filter.filter_context(
            query=context.query,
            messages=session.messages,
            user_facts=session.metadata.get("user_facts", {}),
            instructions=session.user_instructions
        )
        
        # Build prompt sections
        sections = []
        
        # 1. System instruction
        sections.append(f"<system>\n{self.system_instruction}\n</system>")
        
        # 2. Active instructions (if any)
        if filtered["active_instructions"]:
            instructions_text = self._format_instructions(filtered["active_instructions"])
            sections.append(f"<active_instructions>\n{instructions_text}\n</active_instructions>")
        
        # 3. Relevant user facts (if any)
        if filtered["relevant_facts"]:
            facts_text = self._format_facts(filtered["relevant_facts"])
            sections.append(f"<user_info>\n{facts_text}\n</user_info>")
        
        # 4. Relevant conversation history (if any)
        if filtered["should_use_history"] and filtered["relevant_messages"]:
            history_text = self._format_history(filtered["relevant_messages"])
            sections.append(f"<conversation_history>\n{history_text}\n</conversation_history>")
        
        # 5. RAG context (if provided)
        rag_context = kwargs.get("rag_context", "")
        if rag_context:
            sections.append(f"<knowledge_base>\n{rag_context}\n</knowledge_base>")
        
        # 6. Tool results (if any)
        tool_result = kwargs.get("tool_result")
        if tool_result:
            tool_text = self._format_tool_result(tool_result)
            sections.append(f"<tool_result>\n{tool_text}\n</tool_result>")
        
        # 7. Current query
        sections.append(f"<current_query>\nUser: {context.query}\n</current_query>")
        
        # 8. Response instruction
        sections.append("\nAssistant:")
        
        # Build final prompt
        full_prompt = "\n\n".join(sections)
        
        # Store session ID for response tracking
        context.metadata["session_id"] = session.session_id
        
        logger.debug(f"Built intelligent prompt - Query type: {filtered['query_type']}, "
                    f"History: {len(filtered['relevant_messages'])}, "
                    f"Facts: {len(filtered['relevant_facts'])}")
        
        return full_prompt
    
    def _format_instructions(self, instructions: Dict[str, Any]) -> str:
        """Format active instructions."""
        lines = []
        
        for key, instruction in instructions.items():
            # New format: instructions are dicts with 'description'
            if isinstance(instruction, dict):
                desc = instruction.get('description', '')
                if desc:
                    lines.append(f"- {desc}")
            # Old format: plain strings
            elif isinstance(instruction, str):
                lines.append(f"- {instruction}")
        
        return "\n".join(lines) if lines else "No active instructions"
    
    def _format_facts(self, facts: Dict[str, Any]) -> str:
        """Format relevant user facts."""
        lines = []
        for key, value in facts.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines) if lines else "No user information"
    
    def _format_history(self, messages: List[Dict[str, str]]) -> str:
        """Format relevant conversation history."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "No conversation history"
    
    def _format_tool_result(self, tool_result: Dict[str, Any]) -> str:
        """Format tool execution result."""
        if tool_result.get("success"):
            data = tool_result.get("data", {})
            if isinstance(data, dict) and "content" in data:
                return f"Tool execution successful:\n{data['content']}"
            else:
                return f"Tool execution successful:\n{data}"
        else:
            return f"Tool execution failed: {tool_result.get('error', 'Unknown error')}"
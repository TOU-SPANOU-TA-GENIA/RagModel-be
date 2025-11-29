# app/llm/intelligent_prompt_builder.py
"""
Intelligent prompt builder with context relevance filtering.
FIXED: Now uses chat_history from Context (database messages) instead of separate session memory.
"""

from typing import Dict, Any, Optional, List

from app.core.interfaces import PromptBuilder, Context
from app.core.conversation_memory import conversation_memory
from app.core.context_relevance_filter import context_filter
from app.llm.tool_result_formatter import tool_result_formatter
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class IntelligentPromptBuilder(PromptBuilder):
    """
    Prompt builder with intelligent context selection.
    """
    
    def __init__(self, system_instruction: str, tools: Optional[Dict[str, Any]] = None):
        self.system_instruction = system_instruction
        self.tools = tools or {}
    
    def build(self, context: Context, **kwargs) -> str:
        # FIXED: Use chat_history from context (from database) instead of separate session
        chat_history = context.chat_history or []
        
        # For backward compatibility, also check session memory
        # but prioritize chat_history from context (database messages)
        session_id = context.metadata.get("session_id")
        session = conversation_memory.get_or_create_session(session_id) if session_id else None
        
        # Use database chat_history if available, otherwise fall back to session
        if chat_history:
            messages_to_use = chat_history
            logger.info(f"🔍 Using {len(chat_history)} messages from database chat_history")
        elif session:
            messages_to_use = session.messages
            logger.info(f"⚠️ Fallback: Using {len(session.messages)} messages from session memory")
        else:
            messages_to_use = []
            logger.info("❌ No history available")
        
        # BYPASS aggressive context_filter - it filters out ALL messages
        # Instead, just use all messages directly
        filtered = {
            "query_type": "conversational",
            "should_use_history": len(messages_to_use) > 0,
            "relevant_messages": messages_to_use,  # Use ALL messages, don't filter
            "active_instructions": [],
            "relevant_facts": {}
        }
        
        sections = self._build_sections(context, filtered, **kwargs)
        full_prompt = "\n\n".join(sections)
        
        if session:
            context.metadata["session_id"] = session.session_id
        
        logger.info(f"✅ Built prompt with {len(filtered['relevant_messages'])} messages in history")
        
        return full_prompt
    
    def _build_sections(self, context: Context, filtered: Dict, **kwargs) -> List[str]:
        sections = []
        
        sections.append(f"<s>\n{self.system_instruction}\n</s>")
        
        if filtered["active_instructions"]:
            instructions_text = self._format_instructions(filtered["active_instructions"])
            sections.append(f"<active_instructions>\n{instructions_text}\n</active_instructions>")
        
        if filtered["relevant_facts"]:
            facts_text = self._format_facts(filtered["relevant_facts"])
            sections.append(f"<user_info>\n{facts_text}\n</user_info>")
        
        if filtered["should_use_history"] and filtered["relevant_messages"]:
            history_text = self._format_history(filtered["relevant_messages"])
            sections.append(f"<conversation_history>\n{history_text}\n</conversation_history>")
        
        rag_context = kwargs.get("rag_context", "")
        if rag_context:
            sections.append(f"<knowledge_base>\n{rag_context}\n</knowledge_base>")
        
        tool_result = kwargs.get("tool_result")
        if tool_result:
            tool_text = tool_result_formatter.format(tool_result)
            sections.append(f"<tool_result>\n{tool_text}\n</tool_result>")
        
        sections.append(f"<current_query>\nUser: {context.query}\n</current_query>")
        sections.append("\nAssistant:")
        
        return sections
    
    def _format_instructions(self, instructions: List[str]) -> str:
        return "\n".join(f"- {inst}" for inst in instructions)
    
    def _format_facts(self, facts: Dict[str, Any]) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in facts.items())
    
    def _format_history(self, messages: List[Dict[str, str]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
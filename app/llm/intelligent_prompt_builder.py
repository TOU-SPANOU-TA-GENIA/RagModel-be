# app/llm/intelligent_prompt_builder.py
"""
Intelligent prompt builder with context relevance filtering.
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
        session_id = context.metadata.get("session_id")
        session = conversation_memory.get_or_create_session(session_id)
        
        filtered = context_filter.filter_context(
            query=context.query,
            messages=session.messages,
            user_facts=session.metadata.get("user_facts", {}),
            instructions=session.user_instructions
        )
        
        sections = self._build_sections(context, filtered, **kwargs)
        full_prompt = "\n\n".join(sections)
        
        context.metadata["session_id"] = session.session_id
        
        logger.debug(f"Built prompt - Query type: {filtered['query_type']}, "
                    f"History: {len(filtered['relevant_messages'])}")
        
        return full_prompt
    
    def _build_sections(self, context: Context, filtered: Dict, **kwargs) -> List[str]:
        sections = []
        
        sections.append(f"<system>\n{self.system_instruction}\n</system>")
        
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
    
    def _format_instructions(self, instructions: Dict[str, Any]) -> str:
        lines = []
        for key, instruction in instructions.items():
            if isinstance(instruction, dict):
                desc = instruction.get('description', '')
                if desc:
                    lines.append(f"- {desc}")
            elif isinstance(instruction, str):
                lines.append(f"- {instruction}")
        return "\n".join(lines) if lines else "No active instructions"
    
    def _format_facts(self, facts: Dict[str, Any]) -> str:
        lines = [f"- {key}: {value}" for key, value in facts.items()]
        return "\n".join(lines) if lines else "No user information"
    
    def _format_history(self, messages: List[Dict[str, str]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "No conversation history"
# app/llm/intelligent_prompt_builder.py
"""
Intelligent prompt builder with context relevance filtering.
OPTIMIZED FOR 6GB VRAM - Aggressive context limiting.
"""

from typing import Dict, Any, Optional, List
import re

from app.core.interfaces import PromptBuilder, Context
from app.core.conversation_memory import conversation_memory
from app.llm.tool_result_formatter import tool_result_formatter
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# MEMORY LIMITS - Tuned for 6GB VRAM with Qwen3-4B
# =============================================================================
MAX_RAG_CONTEXT_CHARS = 1500      # Was 2000
MAX_PROMPT_CHARS = 4000           # Was 6000 - keep under 3000 tokens
MAX_SYSTEM_CHARS = 1000           # Limit system instruction
MAX_HISTORY_MESSAGES = 3          # Only last 3 messages
MAX_MESSAGE_LENGTH = 200          # Truncate long messages in history


class IntelligentPromptBuilder(PromptBuilder):
    """
    Prompt builder with intelligent context selection and memory management.
    Optimized for limited VRAM environments.
    """
    
    # Patterns that indicate simple queries not needing RAG
    SIMPLE_QUERY_PATTERNS = [
        r'^(γεια|γειά|χαίρε|καλημέρα|καλησπέρα|καληνύχτα|ευχαριστώ|ναι|όχι|οκ|εντάξει)[\s!.?]*$',
        r'^(hi|hello|hey|thanks|yes|no|ok|okay|bye|goodbye)[\s!.?]*$',
        r'^(τι κάνεις|πώς είσαι|how are you)[\s!?.]*$',
        r'^\s*$',
    ]
    
    def __init__(self, system_instruction: str, tools: Optional[Dict[str, Any]] = None):
        self.system_instruction = system_instruction
        self.tools = tools or {}
        self._simple_patterns = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_QUERY_PATTERNS]
    
    def _is_simple_query(self, query: str) -> bool:
        """Check if query is a simple greeting that doesn't need RAG."""
        query_stripped = query.strip()
        if len(query_stripped) < 20:
            for pattern in self._simple_patterns:
                if pattern.match(query_stripped):
                    return True
        return False
    
    def _truncate_rag_context(self, rag_context: str) -> str:
        """Truncate RAG context to fit memory limits."""
        if not rag_context or len(rag_context) <= MAX_RAG_CONTEXT_CHARS:
            return rag_context
        
        truncated = rag_context[:MAX_RAG_CONTEXT_CHARS]
        
        # Cut at sentence boundary if possible
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        cut_point = max(last_period, last_newline)
        
        if cut_point > MAX_RAG_CONTEXT_CHARS * 0.6:
            truncated = truncated[:cut_point + 1]
        
        logger.info(f"Truncated RAG: {len(rag_context)} → {len(truncated)} chars")
        return truncated + "\n[...]"
    
    def build(self, context: Context, **kwargs) -> str:
        # Get chat history
        chat_history = context.chat_history or []
        
        # Fallback to session memory
        session_id = context.metadata.get("session_id")
        session = conversation_memory.get_or_create_session(session_id) if session_id else None
        
        if chat_history:
            messages_to_use = chat_history
        elif session:
            messages_to_use = session.messages
        else:
            messages_to_use = []
        
        # AGGRESSIVE: Only last N messages
        messages_to_use = messages_to_use[-MAX_HISTORY_MESSAGES:]
        
        filtered = {
            "should_use_history": len(messages_to_use) > 0,
            "relevant_messages": messages_to_use,
        }
        
        sections = self._build_sections(context, filtered, **kwargs)
        full_prompt = "\n\n".join(sections)
        
        # Final check
        if len(full_prompt) > MAX_PROMPT_CHARS:
            logger.warning(f"Prompt too large ({len(full_prompt)}), emergency truncate")
            full_prompt = self._emergency_truncate(full_prompt, context.query)
        
        if session:
            context.metadata["session_id"] = session.session_id
        
        logger.info(f"✅ Prompt: {len(full_prompt)} chars")
        return full_prompt
    
    def _build_sections(self, context: Context, filtered: Dict, **kwargs) -> List[str]:
        sections = []
        
        # System instruction (truncated)
        system_text = self.system_instruction[:MAX_SYSTEM_CHARS]
        if len(self.system_instruction) > MAX_SYSTEM_CHARS:
            system_text += "..."
        sections.append(f"<s>\n{system_text}\n</s>")
        
        # History (limited)
        if filtered["should_use_history"] and filtered["relevant_messages"]:
            history_text = self._format_history(filtered["relevant_messages"])
            sections.append(f"<history>\n{history_text}\n</history>")
        
        # RAG context - skip for simple queries
        rag_context = kwargs.get("rag_context", "")
        if rag_context and not self._is_simple_query(context.query):
            truncated_rag = self._truncate_rag_context(rag_context)
            if truncated_rag:
                sections.append(f"<context>\n{truncated_rag}\n</context>")
        elif rag_context:
            logger.info(f"⏭️ Skip RAG for simple query")
        
        # Tool result (limited)
        tool_result = kwargs.get("tool_result")
        if tool_result:
            tool_text = tool_result_formatter.format(tool_result)
            if len(tool_text) > 1000:
                tool_text = tool_text[:1000] + "\n[...]"
            sections.append(f"<tool>\n{tool_text}\n</tool>")
        
        # Query
        sections.append(f"<query>\nUser: {context.query}\n</query>")
        sections.append("\nAssistant:")
        
        return sections
    
    def _emergency_truncate(self, prompt: str, query: str) -> str:
        """Emergency truncation when prompt is still too large."""
        # Keep system + query only
        system_end = prompt.find("</s>")
        query_start = prompt.find("<query>")
        
        if system_end > 0 and query_start > 0:
            system_part = prompt[:min(system_end + 4, MAX_SYSTEM_CHARS)]
            query_part = prompt[query_start:]
            return system_part + "\n\n" + query_part
        
        return prompt[:MAX_PROMPT_CHARS]
    
    def _format_history(self, messages: List[Dict[str, str]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > MAX_MESSAGE_LENGTH:
                content = content[:MAX_MESSAGE_LENGTH] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
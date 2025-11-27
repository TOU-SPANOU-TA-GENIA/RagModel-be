# app/core/advanced_context_builder.py
"""
Advanced context builder for robust conversation handling.
"""

from typing import Optional
from app.core.advanced_conversation_memory import ConversationContext, advanced_memory
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AdvancedContextBuilder:
    """
    Builds robust conversation context that works across all scenarios.
    """
    
    def __init__(self, max_history_messages: int = 8):
        self.max_history_messages = max_history_messages
    
    def build_context_for_query(self, session: ConversationContext, current_query: str) -> str:
        """Build comprehensive context for any scenario."""
        context_parts = []
        
        # 1. Check for instruction matches first
        instruction_response = session.check_instruction_match(current_query)
        if instruction_response:
            logger.info(f"🎯 Instruction matched: responding with '{instruction_response}'")
            context_parts.append(self._build_instruction_context(current_query, instruction_response))
        
        # 2. General conversation context
        conversation_context = session.get_conversation_context()
        if conversation_context:
            context_parts.append(conversation_context)
        
        # 3. Current query context
        context_parts.append(self._build_current_query_context(current_query))
        
        return "\n\n".join(context_parts)
    
    def _build_instruction_context(self, query: str, response: str) -> str:
        """Build context for instruction following."""
        return f"""## INSTRUCTION FOLLOWING - RESPOND EXACTLY:

User query: "{query}"
Matched instruction response: "{response}"

RESPOND WITH THIS EXACT PHRASE: "{response}"

Do not modify, explain, or add to this response. The user expects exactly this response.
"""
    
    def _build_current_query_context(self, query: str) -> str:
        """Build context for current query."""
        return f"""## CURRENT QUERY:

User: {query}

Respond naturally while maintaining conversation context and following any active instructions.
"""

# Global advanced context builder
advanced_context_builder = AdvancedContextBuilder()
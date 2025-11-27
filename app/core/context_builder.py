# app/core/context_builder.py
"""
Builds comprehensive context from conversation history.
"""

from typing import List, Dict, Any, Optional
from app.core.conversation_memory import ConversationContext, conversation_memory
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class ContextBuilder:
    """
    Builds intelligent context from conversation history.
    """
    
    def __init__(self, max_history_messages: int = 10):
        self.max_history_messages = max_history_messages
    
    def build_conversation_context(self, session: ConversationContext, current_query: str) -> str:
        """Build comprehensive conversation context."""
        context_parts = []
        
        # 1. Add conversation instructions
        instructions = session.get_conversation_summary()
        if instructions:
            context_parts.append(instructions)
        
        # 2. Add recent conversation history
        recent_messages = session.get_recent_messages(self.max_history_messages)
        if recent_messages:
            context_parts.append(self._format_conversation_history(recent_messages))
        
        # 3. Add current query context
        context_parts.append(self._build_current_context(current_query, session))
        
        return "\n\n".join(filter(None, context_parts))
    
    def _format_conversation_history(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation history for context."""
        if not messages:
            return ""
        
        history_lines = ["## Recent Conversation:"]
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        
        return "\n".join(history_lines)
    
    def _build_current_context(self, current_query: str, session: ConversationContext) -> str:
        """Build context for current query with instruction checking."""
        context_parts = ["## Current Query:"]
        context_parts.append(f"User: {current_query}")
        
        # Check if current query matches any stored instructions
        instruction_response = self._check_instructions(current_query, session)
        if instruction_response:
            context_parts.append(f"\n## Instruction Match:")
            context_parts.append(f"User said: '{current_query}'")
            context_parts.append(f"Following instruction, you should respond: '{instruction_response}'")
        
        return "\n".join(context_parts)
    
    def _check_instructions(self, query: str, session: ConversationContext) -> Optional[str]:
        """Check if query matches any stored instructions with better matching."""
        query_lower = query.lower().strip()
        
        for trigger, response in session.user_instructions.items():
            trigger_lower = trigger.lower().strip()
            
            # More flexible matching
            if (query_lower == trigger_lower or 
                query_lower.startswith(trigger_lower) or
                trigger_lower in query_lower or
                self._fuzzy_match(query_lower, trigger_lower)):
                
                logger.info(f"🎯 Instruction matched: '{trigger}' -> '{response}'")
                return response
        
        return None
    
    def _fuzzy_match(self, query: str, trigger: str) -> bool:
        """Simple fuzzy matching for instructions."""
        # Remove punctuation and extra spaces
        import re
        query_clean = re.sub(r'[^\w\s]', '', query)
        trigger_clean = re.sub(r'[^\w\s]', '', trigger)
        
        return trigger_clean in query_clean

# Global context builder
context_builder = ContextBuilder()
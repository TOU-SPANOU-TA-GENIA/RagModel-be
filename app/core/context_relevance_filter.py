# app/core/context_relevance_filter.py
"""
Context relevance filter - decides what conversation context is relevant to current query.
This prevents context bleeding and ensures appropriate context usage.
"""

from typing import List, Dict, Any, Set
import re
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class QueryType:
    """Classification of query types for context relevance."""
    FACTUAL = "factual"  # Math, geography, definitions, science
    PERSONAL = "personal"  # About the user
    TECHNICAL = "technical"  # Code, how-to
    CONVERSATIONAL = "conversational"  # Greetings, chat
    INSTRUCTION = "instruction"  # Setting rules

class ContextRelevanceFilter:
    """
    Filters conversation context to include only relevant information.
    Prevents context bleeding and improves response quality.
    """
    
    def __init__(self):
        # Patterns for detecting query types
        self.factual_patterns = [
            r'\bwhat\s+is\s+\d+\s*[\+\-\*\/]',  # Math questions
            r'\bcapital\s+of\b',  # Geography
            r'\bboiling\s+point\b',  # Science
            r'\bwhat\s+is\s+the\b.*\?',  # General "what is X?"
            r'\bhow\s+many\b',  # Counting
            r'\bwhen\s+did\b',  # Historical dates
        ]
        
        self.personal_patterns = [
            r'\bmy\s+name\b',
            r'\bwhat\s+(do\s+)?i\b',
            r'\bwhere\s+am\s+i\b',
            r'\bwhat\s+did\s+i\b',
            r'\bremind\s+me\b',
            r'\babout\s+me\b',
        ]
        
        self.technical_patterns = [
            r'\bhow\s+(do\s+i|to)\s+(write|code|program|implement)',
            r'\bpython\b.*\bcode\b',
            r'\breverse\s+a\s+string\b',
            r'\bfunction\s+to\b',
            r'\balgorithm\b',
        ]
        
        self.instruction_patterns = [
            r'\bwhen\s+i\s+say\b',
            r'\balways\s+(respond|answer|be)\b',
            r'\bnever\s+',
            r'\bend\s+with\b',
            r'\bremember\s+to\b',
        ]
    
    def classify_query(self, query: str) -> str:
        """Classify the query type."""
        query_lower = query.lower()
        
        # Check instruction patterns first
        for pattern in self.instruction_patterns:
            if re.search(pattern, query_lower):
                return QueryType.INSTRUCTION
        
        # Check personal patterns
        for pattern in self.personal_patterns:
            if re.search(pattern, query_lower):
                return QueryType.PERSONAL
        
        # Check factual patterns
        for pattern in self.factual_patterns:
            if re.search(pattern, query_lower):
                return QueryType.FACTUAL
        
        # Check technical patterns
        for pattern in self.technical_patterns:
            if re.search(pattern, query_lower):
                return QueryType.TECHNICAL
        
        # Default to conversational
        return QueryType.CONVERSATIONAL
    
    def filter_context(
        self, 
        query: str,
        messages: List[Dict[str, str]],
        user_facts: Dict[str, Any],
        instructions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter conversation context based on query relevance.
        
        Returns:
            Dict with filtered context components
        """
        query_type = self.classify_query(query)
        
        filtered_context = {
            "query_type": query_type,
            "relevant_messages": [],
            "relevant_facts": {},
            "active_instructions": instructions.copy(),  # Always include instructions
            "should_use_history": True
        }
        
        # Apply filtering rules based on query type
        if query_type == QueryType.FACTUAL:
            # For factual questions, minimal context
            filtered_context["should_use_history"] = False
            filtered_context["relevant_messages"] = []
            filtered_context["relevant_facts"] = {}
            # Keep instructions though
            
        elif query_type == QueryType.PERSONAL:
            # For personal questions, include all relevant facts
            filtered_context["relevant_facts"] = user_facts.copy()
            filtered_context["relevant_messages"] = self._filter_personal_messages(
                messages, query
            )
            
        elif query_type == QueryType.TECHNICAL:
            # For technical questions, include recent context only
            filtered_context["relevant_messages"] = messages[-3:] if messages else []
            # Include user role if relevant (e.g., "I'm a developer")
            if "job" in user_facts or "role" in user_facts:
                filtered_context["relevant_facts"] = {
                    k: v for k, v in user_facts.items() 
                    if k in ["job", "role", "experience"]
                }
            
        elif query_type == QueryType.INSTRUCTION:
            # For instructions, include minimal history
            filtered_context["relevant_messages"] = messages[-2:] if messages else []
            
        else:  # CONVERSATIONAL
            # For conversational, include recent history and some facts
            filtered_context["relevant_messages"] = messages[-5:] if messages else []
            filtered_context["relevant_facts"] = user_facts.copy()
        
        logger.debug(f"Query type: {query_type}, Filtered context: "
                    f"{len(filtered_context['relevant_messages'])} messages, "
                    f"{len(filtered_context['relevant_facts'])} facts")
        
        return filtered_context
    
    def _filter_personal_messages(
        self, 
        messages: List[Dict[str, str]], 
        query: str
    ) -> List[Dict[str, str]]:
        """Filter messages for personal queries."""
        query_lower = query.lower()
        relevant_messages = []
        
        # Look for messages that might be relevant
        keywords = self._extract_keywords(query_lower)
        
        for msg in messages[-10:]:  # Check last 10 messages
            content_lower = msg.get("content", "").lower()
            
            # Include if it mentions any query keywords
            if any(keyword in content_lower for keyword in keywords):
                relevant_messages.append(msg)
            
            # Include if it's a recent instruction
            if any(word in content_lower for word in ["when i say", "always", "never"]):
                relevant_messages.append(msg)
        
        return relevant_messages[-5:]  # Keep max 5 relevant messages
    
    def _extract_keywords(self, query: str) -> Set[str]:
        """Extract key terms from query."""
        # Remove common words
        stop_words = {
            "what", "is", "the", "my", "do", "i", "am", "are", "did",
            "can", "you", "tell", "me", "about", "where", "when", "how"
        }
        
        words = query.lower().split()
        keywords = {word for word in words if word not in stop_words and len(word) > 2}
        
        return keywords

# Global instance
context_filter = ContextRelevanceFilter()
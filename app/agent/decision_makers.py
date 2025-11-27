# app/agent/decision_makers.py
"""
Decision making implementations.
Determines what actions to take based on intent.
"""

import re
from typing import Dict, Any, Optional

from app.core.interfaces import Intent, Context, Decision, DecisionMaker, Tool
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SimpleDecisionMaker(DecisionMaker):
    """
    Rule-based decision maker for tool selection.
    """
    
    def __init__(self, tools: Optional[Dict[str, Tool]] = None):
        self.tools = tools or {}
        self._extractors = ParameterExtractors()
    
    def decide(self, context: Context, intent: Intent) -> Decision:
        """Make decision based on intent."""
        decision = Decision(
            intent=intent,
            confidence=1.0,
            use_tool=False,
            tool_name=None,
            tool_params=None,
            use_rag=True,
            reasoning=""
        )
        
        if intent == Intent.CONVERSATION:
            decision.use_rag = False
            decision.reasoning = "Casual conversation"
            
        elif intent == Intent.QUESTION:
            decision.use_rag = True
            decision.reasoning = "Information query, using RAG"
            
        elif intent == Intent.ACTION:
            tool_info = self._identify_tool(context.query)
            if tool_info:
                decision.use_tool = True
                decision.tool_name = tool_info['name']
                decision.tool_params = tool_info['params']
                decision.reasoning = f"Using {tool_info['name']} tool"
                logger.info(f"Tool: {tool_info['name']} params: {tool_info['params']}")
        
        context.add_debug(f"Decision: {decision.reasoning}")
        return decision
    
    def _identify_tool(self, query: str) -> Optional[Dict[str, Any]]:
        """Identify which tool to use."""
        query_lower = query.lower()
        
        # Document generation (check first)
        if self._is_document_request(query_lower):
            doc_info = self._extractors.extract_document_request(query)
            if doc_info[0]:
                return {
                    'name': 'generate_document',
                    'params': {
                        'doc_type': doc_info[0],
                        'title': doc_info[1] or 'Generated Document',
                        'content': doc_info[2] or query
                    }
                }
        
        # File reading
        if self._is_file_read_request(query_lower):
            file_ref = self._extractors.extract_file_reference(query)
            if file_ref and self._validate_file_reference(file_ref):
                return {
                    'name': 'read_file',
                    'params': {'file_identifier': file_ref}
                }
        
        # File search
        if self._is_file_search_request(query_lower):
            search_term = self._extractors.extract_search_term(query)
            if search_term:
                return {'name': 'search_files', 'params': {'query': search_term}}
        
        # List directory
        if self._is_list_request(query_lower):
            directory = self._extractors.extract_directory(query)
            return {'name': 'list_files', 'params': {'directory': directory or '.'}}
        
        return None
    
    def _is_document_request(self, query: str) -> bool:
        action_words = ['create', 'build', 'generate', 'make']
        doc_words = ['document', 'word', 'powerpoint', 'presentation', 'pdf', 'ppt', 'docx']
        return any(w in query for w in action_words) and any(w in query for w in doc_words)
    
    def _is_file_read_request(self, query: str) -> bool:
        read_words = ['read', 'show', 'open', 'view', 'display', 'cat']
        return any(w in query for w in read_words) and ('file' in query or self._has_file_extension(query))
    
    def _is_file_search_request(self, query: str) -> bool:
        return any(p in query for p in ['find file', 'search file', 'locate file', 'look for'])
    
    def _is_list_request(self, query: str) -> bool:
        return any(p in query for p in ['list files', 'show files', 'files in'])
    
    def _has_file_extension(self, query: str) -> bool:
        return bool(re.search(r'\.\w{2,4}\b', query))
    
    def _validate_file_reference(self, file_ref: str) -> bool:
        if not file_ref:
            return False
        stop_words = {'me', 'the', 'of', 'file', 'contents', 'content'}
        if file_ref.lower() in stop_words:
            return False
        return 2 <= len(file_ref) <= 500


class ParameterExtractors:
    """Utility class for extracting parameters from queries."""
    
    def extract_document_request(self, query: str) -> tuple:
        """Extract document type, title, and description."""
        query_lower = query.lower()
        
        doc_types = {
            'word': 'docx', 'doc': 'docx', 'docx': 'docx',
            'powerpoint': 'pptx', 'ppt': 'pptx', 'pptx': 'pptx',
            'presentation': 'pptx', 'pdf': 'pdf',
            'text': 'txt', 'markdown': 'md'
        }
        
        doc_type = None
        for keyword, file_type in doc_types.items():
            if keyword in query_lower:
                doc_type = file_type
                break
        
        if not doc_type:
            return None, None, None
        
        title = self._extract_title(query_lower)
        return doc_type, title, query
    
    def _extract_title(self, query: str) -> Optional[str]:
        patterns = [
            r'(?:about|regarding|on)\s+(.+?)(?:\s+with|\s+that|\s*$)',
            r'(?:build|create|generate|make)\s+(?:a|an)?\s+\w+\s+(.+?)(?:\s+with|\s*$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        return None
    
    def extract_file_reference(self, query: str) -> Optional[str]:
        """Extract file path or name from query."""
        query_clean = re.sub(
            r'^(please\s+|could\s+you\s+|can\s+you\s+|would\s+you\s+)', 
            '', query, flags=re.IGNORECASE
        )
        
        patterns = [
            (r'(?:show|display)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:contents?\s+)?'
             r'(?:of\s+)?(?:the\s+)?(?:file\s+)?(?:called\s+)?(?:named\s+)?'
             r'([^\s,]+(?:\.\w+)?)', 1),
            (r'(?:read|open|view|cat)\s+(?:the\s+)?(?:file\s+)?([^\s,]+(?:\.\w+)?)', 1),
            (r'contents?\s+(?:of\s+)?([^\s,]+(?:\.\w+)?)', 1),
            (r'([^\s/\\]+\.\w{2,4})\s*$', 1),
        ]
        
        stop_words = {'all', 'the', 'of', 'me', 'contents', 'content', 'file'}
        
        for pattern, group in patterns:
            match = re.search(pattern, query_clean, re.IGNORECASE)
            if match:
                candidate = match.group(group).strip('.,!?')
                if candidate.lower() not in stop_words:
                    return candidate
        
        return None
    
    def extract_search_term(self, query: str) -> Optional[str]:
        match = re.search(r'(?:find|search|locate|look for)\s+(?:file\s+)?(.+)', 
                         query, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def extract_directory(self, query: str) -> Optional[str]:
        match = re.search(r'(?:in|from)\s+([^\s]+)', query, re.IGNORECASE)
        return match.group(1) if match else None


class LLMDecisionMaker(DecisionMaker):
    """Decision maker with LLM fallback for complex cases."""
    
    def __init__(self, simple_maker: SimpleDecisionMaker, llm_provider=None):
        self.simple_maker = simple_maker
        self.llm_provider = llm_provider
    
    def decide(self, context: Context, intent: Intent) -> Decision:
        decision = self.simple_maker.decide(context, intent)
        
        if intent == Intent.ACTION and not decision.use_tool and self.llm_provider:
            decision = self._decide_with_llm(context, intent, decision)
        
        return decision
    
    def _decide_with_llm(self, context: Context, intent: Intent, 
                        initial: Decision) -> Decision:
        return initial
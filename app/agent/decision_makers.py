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


class ParameterExtractors:
    """Extract parameters from natural language queries."""
    
    def extract_filename(self, query: str) -> Optional[str]:
        """Extract filename from query."""
        # Look for quoted filenames
        quoted = re.search(r'["\']([^"\']+\.\w+)["\']', query)
        if quoted:
            return quoted.group(1)
        
        # Look for filenames with extensions
        file_match = re.search(r'\b(\S+\.\w{2,4})\b', query)
        if file_match:
            return file_match.group(1)
        
        return None
    
    def extract_directory(self, query: str) -> Optional[str]:
        """Extract directory path from query."""
        # Look for explicit paths
        path_match = re.search(r'(?:in|from|at)\s+["\']?([/\\]?\S+)["\']?', query.lower())
        if path_match:
            return path_match.group(1)
        return None
    
    def extract_extension_pattern(self, query: str) -> Optional[str]:
        """Extract file extension pattern (e.g., *.txt)."""
        # Look for extension mentions
        ext_match = re.search(r'\.(\w{2,4})\s+files?|(\w{2,4})\s+files?', query.lower())
        if ext_match:
            ext = ext_match.group(1) or ext_match.group(2)
            return f"*.{ext}"
        return None
    
    def extract_search_query(self, query: str) -> Optional[str]:
        """Extract search query for file search."""
        # Remove action words
        cleaned = re.sub(
            r'^(?:search|find|look)\s+(?:for\s+)?(?:files?\s+)?(?:named|called|matching)?\s*',
            '', 
            query.lower()
        ).strip()
        return cleaned if cleaned else None
    
    def extract_write_params(self, query: str) -> Dict[str, Any]:
        """Extract parameters for file writing."""
        params = {}
        
        # Extract filename
        filename = self.extract_filename(query)
        if filename:
            params['file_path'] = filename
        
        # Content would typically come from context, not query
        params['content'] = ''
        
        return params
    
    def extract_document_params(self, query: str) -> Dict[str, Any]:
        """Extract parameters for document generation."""
        params = {'format': 'docx', 'title': 'Document'}
        
        # Detect format
        if re.search(r'\bpdf\b', query.lower()):
            params['format'] = 'pdf'
        elif re.search(r'\b(?:ppt|powerpoint|presentation)\b', query.lower()):
            params['format'] = 'pptx'
        elif re.search(r'\b(?:word|doc)\b', query.lower()):
            params['format'] = 'docx'
        
        return params
    
    def extract_command(self, query: str) -> Optional[str]:
        """Extract shell command from query."""
        # Look for backtick-enclosed commands
        backtick = re.search(r'`([^`]+)`', query)
        if backtick:
            return backtick.group(1)
        
        # Look for "run X" pattern
        run_match = re.search(r'run\s+(?:command\s+)?["\']?(.+?)["\']?$', query.lower())
        if run_match:
            return run_match.group(1)
        
        return None


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
        """
        Identify which tool to use based on query patterns.
        Order matters: more specific patterns should come first.
        """
        query_lower = query.lower()
        
        # =========================================================================
        # 1. LIST/SHOW FILES PATTERNS (check first - most likely to be misclassified)
        # =========================================================================
        list_patterns = [
            # Explicit list requests
            r'list\s+(?:all\s+)?(?:the\s+)?files',
            r'show\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?files',
            r'what\s+files\s+(?:do\s+you\s+)?(?:have|see|can)',
            r'which\s+files\s+(?:do\s+you\s+)?(?:have|see|can)',
            r'display\s+(?:all\s+)?(?:the\s+)?files',
            
            # Access/availability questions
            r'(?:files|documents)\s+(?:do\s+)?you\s+(?:have|see)\s+access',
            r'(?:what|which)\s+(?:files|documents)\s+(?:are\s+)?available',
            r'files\s+you\s+(?:can\s+)?(?:see|access|read)',
            
            # Directory listing
            r'(?:show|list|display)\s+(?:the\s+)?directory',
            r'(?:what\'?s?|show)\s+in\s+(?:the\s+)?(?:folder|directory)',
            
            # Generic "all files" patterns
            r'all\s+(?:the\s+)?(?:txt|pdf|doc|files)',
            r'(?:every|all)\s+file',
        ]
        
        for pattern in list_patterns:
            if re.search(pattern, query_lower):
                # Extract optional directory or pattern
                directory = self._extractors.extract_directory(query) or "."
                file_pattern = self._extractors.extract_extension_pattern(query) or "*"
                
                return {
                    'name': 'list_files',
                    'params': {
                        'directory': directory,
                        'pattern': file_pattern
                    }
                }
        
        # =========================================================================
        # 2. SEARCH FILES PATTERNS
        # =========================================================================
        search_patterns = [
            r'search\s+(?:for\s+)?(?:files?|documents?)',
            r'find\s+(?:files?|documents?)\s+(?:named|called|matching)',
            r'look\s+for\s+(?:files?|documents?)',
        ]
        
        for pattern in search_patterns:
            if re.search(pattern, query_lower):
                search_query = self._extractors.extract_search_query(query)
                if search_query:
                    return {
                        'name': 'search_files',
                        'params': {'query': search_query}
                    }
        
        # =========================================================================
        # 3. READ FILE PATTERNS (must have specific file reference)
        # =========================================================================
        read_patterns = [
            r'read\s+(?:the\s+)?(?:file\s+)?["\']?(\S+\.\w+)["\']?',
            r'open\s+(?:the\s+)?(?:file\s+)?["\']?(\S+\.\w+)["\']?',
            r'show\s+(?:me\s+)?(?:the\s+)?(?:contents?\s+of\s+)?["\']?(\S+\.\w+)["\']?',
            r'(?:what\'?s?\s+in|display)\s+["\']?(\S+\.\w+)["\']?',
        ]
        
        for pattern in read_patterns:
            match = re.search(pattern, query_lower)
            if match:
                file_identifier = match.group(1) if match.groups() else None
                if not file_identifier:
                    file_identifier = self._extractors.extract_filename(query)
                
                if file_identifier:
                    return {
                        'name': 'read_file',
                        'params': {'file_identifier': file_identifier}
                    }
        
        # =========================================================================
        # 4. WRITE FILE PATTERNS
        # =========================================================================
        write_patterns = [
            r'(?:write|save|create)\s+(?:a\s+)?(?:new\s+)?file',
            r'save\s+(?:this\s+)?(?:to|as)\s+',
        ]
        
        for pattern in write_patterns:
            if re.search(pattern, query_lower):
                return {
                    'name': 'write_file',
                    'params': self._extractors.extract_write_params(query)
                }
        
        # =========================================================================
        # 5. DOCUMENT GENERATION PATTERNS
        # =========================================================================
        doc_patterns = [
            r'(?:generate|create|make)\s+(?:a\s+)?(?:word|pdf|powerpoint|pptx?|docx?)',
            r'(?:generate|create|make)\s+(?:a\s+)?document',
            r'(?:generate|create|make)\s+(?:a\s+)?presentation',
            r'(?:generate|create|make)\s+(?:a\s+)?report',
        ]
        
        for pattern in doc_patterns:
            if re.search(pattern, query_lower):
                return {
                    'name': 'generate_document',
                    'params': self._extractors.extract_document_params(query)
                }
        
        # =========================================================================
        # 6. COMMAND EXECUTION PATTERNS
        # =========================================================================
        command_patterns = [
            r'run\s+(?:the\s+)?command',
            r'execute\s+',
            r'shell\s+command',
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, query_lower):
                command = self._extractors.extract_command(query)
                if command:
                    return {
                        'name': 'execute_command', 
                        'params': {'command': command}
                    }
        
        return None


class LLMDecisionMaker(DecisionMaker):
    """
    LLM-based decision maker for complex queries.
    Falls back to SimpleDecisionMaker for efficiency.
    """
    
    def __init__(self, llm_provider, tools: Optional[Dict[str, Tool]] = None):
        self.llm = llm_provider
        self.tools = tools or {}
        self._simple_maker = SimpleDecisionMaker(tools)
    
    def decide(self, context: Context, intent: Intent) -> Decision:
        """Make decision, using LLM for complex cases."""
        # First try simple rules
        simple_decision = self._simple_maker.decide(context, intent)
        
        # If simple rules found a tool, use that
        if simple_decision.use_tool:
            return simple_decision
        
        # For complex queries, could use LLM to determine tool
        # For now, return simple decision
        return simple_decision
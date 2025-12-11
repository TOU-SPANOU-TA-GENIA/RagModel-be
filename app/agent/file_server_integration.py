# app/agent/file_server_integration.py
"""
Integration layer connecting file server access with analysis tools.
Routes queries to appropriate tools based on intent detection.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from app.services.file_server import FileServerService, create_file_server
from app.tools.file_server_tool import (
    FileServerTool, 
    detect_file_server_intent,
    extract_folder_from_query
)
from app.analysis.logistics_analyzer import LogisticsAnalyzer
from app.config.file_server_config import get_settings


@dataclass
class QueryAnalysis:
    """Result of analyzing a user query."""
    uses_file_server: bool
    folder_reference: Optional[str]
    action_type: str  # analyze, compare, browse, search, generate_report
    file_types: List[str]
    original_query: str


class FileServerAgent:
    """
    Agent that handles file server queries and routes to appropriate tools.
    """
    
    def __init__(self):
        settings = get_settings()
        self.file_server = create_file_server(
            mount_point=settings.mount_point,
            base_path=settings.smb_path,
            aliases=settings.folder_aliases
        )
        self.file_tool = FileServerTool(self.file_server)
        self.analyzer = LogisticsAnalyzer()
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query to determine intent and required tools.
        
        Args:
            query: User's natural language query
            
        Returns:
            QueryAnalysis with detected intent
        """
        query_lower = query.lower()
        
        # Check for file server reference
        uses_file_server = detect_file_server_intent(query)
        folder = extract_folder_from_query(query) if uses_file_server else None
        
        # Detect action type
        action = self._detect_action(query_lower)
        
        # Detect file types
        file_types = self._detect_file_types(query_lower)
        
        return QueryAnalysis(
            uses_file_server=uses_file_server,
            folder_reference=folder,
            action_type=action,
            file_types=file_types,
            original_query=query
        )
    
    def _detect_action(self, query: str) -> str:
        """Detect the primary action requested."""
        action_patterns = {
            'analyze': [
                r'αναλ[υύ]', r'εντόπισ[εέ]', r'έλεγ[χξ]', r'εξέτασ',
                r'ανωμαλ[ίι]', r'αποκλ[ιί]σ', r'analyze', r'anomal'
            ],
            'compare': [
                r'σ[υύ]γκρι', r'compare', r'διαφορ', r'αντιπαραβολ'
            ],
            'generate_report': [
                r'δημιούργησε.*αναφορ', r'φτιάξε.*αναφορ', r'generate.*report',
                r'αναφορά ελέγχου', r'audit report'
            ],
            'browse': [
                r'δε[ίι]ξ', r'εμφάνισ', r'list', r'show', r'τι υπάρχει'
            ],
            'search': [
                r'ψά[χξ]', r'βρ[εέ]ς?', r'search', r'αναζήτησ'
            ],
        }
        
        for action, patterns in action_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return action
        
        return 'analyze'  # default
    
    def _detect_file_types(self, query: str) -> List[str]:
        """Detect types of files/documents mentioned."""
        type_patterns = {
            'inventory': [r'απογραφ', r'inventory', r'αποθέμα', r'στοκ'],
            'maintenance': [r'συντήρηση', r'maintenance', r'επισκευ'],
            'requisition': [r'αίτη', r'παραγγελ', r'requisition'],
            'budget': [r'προϋπολογ', r'budget', r'οικονομικ', r'δαπάν'],
        }
        
        found = []
        for ftype, patterns in type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    found.append(ftype)
                    break
        
        return found
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query end-to-end.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary with results
        """
        analysis = self.analyze_query(query)
        
        if not analysis.uses_file_server:
            return {
                "success": False,
                "error": "Η ερώτηση δεν αναφέρεται σε αρχεία server",
                "suggestion": "Χρησιμοποιήστε 'από το φάκελο X' για αναφορά σε αρχεία"
            }
        
        # Get files from server
        if analysis.folder_reference:
            success, file_paths, error = self.file_tool.get_files_by_folder(
                analysis.folder_reference
            )
            
            if not success:
                return {"success": False, "error": error}
        else:
            return {
                "success": False,
                "error": "Δεν αναγνωρίστηκε φάκελος",
                "detected": analysis.__dict__
            }
        
        # Route to appropriate handler
        if analysis.action_type == 'browse':
            return await self._handle_browse(analysis.folder_reference)
        
        elif analysis.action_type == 'search':
            return await self._handle_search(query, analysis.folder_reference)
        
        elif analysis.action_type in ['analyze', 'compare', 'generate_report']:
            return await self._handle_analysis(
                file_paths,
                analysis.action_type,
                analysis.original_query
            )
        
        return {"success": False, "error": "Άγνωστη ενέργεια"}
    
    async def _handle_browse(self, folder: str) -> Dict[str, Any]:
        """Handle folder browsing request."""
        result = self.file_tool._browse_folder(folder)
        return result.data if result.success else {"error": result.error}
    
    async def _handle_search(self, query: str, folder: str) -> Dict[str, Any]:
        """Handle file search request."""
        result = self.file_tool._search_files(folder, query)
        return result.data if result.success else {"error": result.error}
    
    async def _handle_analysis(
        self,
        file_paths: List[str],
        action: str,
        query: str
    ) -> Dict[str, Any]:
        """Handle logistics analysis request."""
        # Read file contents
        documents = []
        for path in file_paths:
            try:
                content = self._read_file(path)
                documents.append({
                    "path": path,
                    "content": content
                })
            except Exception as e:
                continue
        
        if not documents:
            return {"success": False, "error": "Δεν ήταν δυνατή η ανάγνωση αρχείων"}
        
        # Run analysis
        result = self.analyzer.analyze_documents(documents)
        
        # Generate report if requested
        if action == 'generate_report':
            result['report_requested'] = True
        
        return {
            "success": True,
            "action": action,
            "files_analyzed": len(documents),
            "result": result
        }
    
    def _read_file(self, path: str) -> str:
        """Read file content based on type."""
        import os
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.txt' or ext == '.md' or ext == '.csv':
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif ext == '.xlsx' or ext == '.xls':
            import pandas as pd
            df = pd.read_excel(path)
            return df.to_string()
        
        elif ext == '.docx':
            from docx import Document
            doc = Document(path)
            return '\n'.join(p.text for p in doc.paragraphs)
        
        elif ext == '.pdf':
            # Use pdfplumber or similar
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    text = ''
                    for page in pdf.pages:
                        text += page.extract_text() or ''
                    return text
            except ImportError:
                return f"[PDF file: {path}]"
        
        return ""


# Singleton instance
_agent_instance: Optional[FileServerAgent] = None


def get_file_server_agent() -> FileServerAgent:
    """Get or create the file server agent singleton."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FileServerAgent()
    return _agent_instance


# ==========================================================================
# QUERY ROUTING FOR MAIN AGENT
# ==========================================================================

def should_use_file_server(query: str) -> bool:
    """
    Check if a query should be routed to file server.
    Use this in main agent decision maker.
    """
    return detect_file_server_intent(query)


async def handle_file_server_query(query: str) -> Dict[str, Any]:
    """
    Handle a file server query.
    Use this in main agent tool execution.
    """
    agent = get_file_server_agent()
    return await agent.process_query(query)
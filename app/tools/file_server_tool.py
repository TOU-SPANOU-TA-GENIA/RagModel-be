# app/tools/file_server_tool.py
"""
File Server Tool - Access files from Windows SMB shares.
"""

import re
from typing import Dict, Any, Optional, List, Tuple

from app.tools.models import BaseTool, ToolResult
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class FileServerTool(BaseTool):
    """
    Tool for accessing files from a Windows file server via SMB.
    """
    
    name = "file_server"
    description = "Πρόσβαση σε αρχεία από Windows file server. Υποστηρίζει φυσική γλώσσα."
    
    def __init__(self, file_server: "FileServerService"):
        super().__init__()
        self.file_server = file_server
    
    def _execute_impl(self, **kwargs) -> ToolResult:
        """Execute file server operation."""
        folder = kwargs.get('folder', '')
        action = kwargs.get('action', 'analyze')
        query = kwargs.get('query', '')
        
        logger.info(f"FileServerTool: folder={folder}, action={action}")
        
        try:
            if action == 'browse':
                return self._browse_folder(folder)
            elif action == 'search':
                return self._search_files(folder, query)
            else:  # analyze, compare
                return self._get_files_for_analysis(folder, query)
        except Exception as e:
            logger.error(f"FileServerTool error: {e}")
            return ToolResult(success=False, error=str(e))
    
    def _browse_folder(self, folder: str) -> ToolResult:
        """List contents of a folder."""
        files = self.file_server.list_folder(folder)
        
        if not files:
            return ToolResult(
                success=False,
                error=f"Δεν βρέθηκε ο φάκελος: {folder}"
            )
        
        return ToolResult(
            success=True,
            data={
                "action": "browse",
                "folder": folder,
                "file_count": len(files),
                "contents": [
                    {
                        "name": f.name,
                        "path": f.path,
                        "size": f.size,
                        "is_directory": f.is_directory,
                    }
                    for f in files
                ]
            }
        )
    
    def _search_files(self, folder: str, query: str) -> ToolResult:
        """Search for files matching query."""
        result = self.file_server.search_files(
            query=query,
            folder=folder,
            recursive=True
        )
        
        return ToolResult(
            success=True,
            data={
                "action": "search",
                "query": query,
                "search_path": result.search_path,
                "total_found": result.total_count,
                "files": [{"name": f.name, "path": f.path, "size": f.size} for f in result.files]
            }
        )
    
    def _get_files_for_analysis(self, folder: str, query: str) -> ToolResult:
        """Get files from folder for analysis."""
        files = self.file_server.get_files(folder)
        
        if not files:
            # Try fuzzy search
            search_result = self.file_server.search_files(
                query=folder,
                recursive=True,
                max_results=20
            )
            files = [f.path for f in search_result.files]
        
        if not files:
            return ToolResult(
                success=False,
                error=f"Δεν βρέθηκαν αρχεία στο: {folder}"
            )
        
        # Copy to temp for processing
        temp_paths = self.file_server.copy_to_temp(files)
        
        return ToolResult(
            success=True,
            data={
                "action": "retrieve",
                "folder": folder,
                "file_count": len(temp_paths),
                "file_paths": temp_paths,
                "original_paths": files,
                "ready_for_analysis": True
            }
        )
    
    def get_files_by_folder(self, folder_name: str) -> Tuple[bool, List[str], str]:
        """Simplified method to get files from a folder."""
        result = self._get_files_for_analysis(folder_name, "")
        
        if result.success:
            return True, result.data.get('file_paths', []), ""
        return False, [], result.error or "Unknown error"


def detect_file_server_intent(query: str) -> bool:
    """Check if a query references file server folders."""
    query_lower = query.lower()
    
    patterns = [
        r'(?:από|μέσα σ?τ[οα]ν?|στ[οα]ν?)\s+φάκελο',
        r'φάκελο[ςσ]?\s+',
        r'αρχεία\s+(?:του?|της?|στ)',
    ]
    
    return any(re.search(p, query_lower) for p in patterns)


def extract_folder_from_query(query: str) -> Optional[str]:
    """Extract folder name from a query."""
    patterns = [
        r'(?:από|μέσα σ?τ[οα]ν?|στ[οα]ν?)\s+φάκελο\s+[«"]?([^»".,]+)[»"]?',
        r'φάκελος\s+[«"]?([^»".,]+)[»"]?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1).strip()
    
    return None
# app/tools/enhanced_file_tools.py
"""
Enhanced file tools with smart search and complete content display.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from app.tools.base import BaseTool, ToolResult
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class FileMatch:
    """Represents a found file."""
    path: Path
    name: str
    size_bytes: int
    relevance_score: float  # How well it matches the query
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "size_mb": round(self.size_bytes / (1024 * 1024), 2),
            "relevance": round(self.relevance_score, 2)
        }


class SmartFileSearchTool(BaseTool):
    """
    Searches for files by name/pattern across allowed directories.
    Returns matches ranked by relevance.
    """
    
    def __init__(self, search_dirs: List[Path] = None, max_results: int = 5):
        super().__init__()
        self.search_dirs = search_dirs or []
        self.max_results = max_results
    
    @property
    def name(self) -> str:
        return "search_files"
    
    @property
    def description(self) -> str:
        return "Search for files by name or pattern. Returns list of matching files."
    
    def _execute_impl(self, query: str, **kwargs) -> ToolResult:
        """Search for files matching query."""
        query_lower = query.lower().strip()
        matches: List[FileMatch] = []
        
        # Search all allowed directories
        for search_dir in self.search_dirs:
            if not search_dir.exists():
                continue
                
            try:
                # Search recursively
                for file_path in search_dir.rglob("*"):
                    if not file_path.is_file():
                        continue
                    
                    # Calculate relevance score
                    score = self._calculate_relevance(file_path.name, query_lower)
                    
                    if score > 0:
                        matches.append(FileMatch(
                            path=file_path,
                            name=file_path.name,
                            size_bytes=file_path.stat().st_size,
                            relevance_score=score
                        ))
            except Exception as e:
                logger.warning(f"Error searching {search_dir}: {e}")
        
        # Sort by relevance
        matches.sort(key=lambda m: m.relevance_score, reverse=True)
        
        # Limit results
        top_matches = matches[:self.max_results]
        
        if not top_matches:
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "matches": [],
                    "message": f"No files found matching '{query}'"
                }
            )
        
        return ToolResult(
            success=True,
            data={
                "query": query,
                "matches": [m.to_dict() for m in top_matches],
                "total_found": len(matches),
                "showing": len(top_matches)
            }
        )
    
    def _calculate_relevance(self, filename: str, query: str) -> float:
        """
        Calculate how well filename matches query.
        Returns score 0-1, higher is better.
        """
        filename_lower = filename.lower()
        
        # Exact match
        if filename_lower == query:
            return 1.0
        
        # Exact match without extension
        name_without_ext = Path(filename).stem.lower()
        if name_without_ext == query:
            return 0.95
        
        # Contains exact query
        if query in filename_lower:
            return 0.8
        
        # Contains query words
        query_words = query.split()
        word_matches = sum(1 for word in query_words if word in filename_lower)
        if word_matches > 0:
            return 0.5 + (0.2 * word_matches / len(query_words))
        
        # Fuzzy match (partial word matches)
        partial_score = 0
        for i in range(len(query) - 2):
            trigram = query[i:i+3]
            if trigram in filename_lower:
                partial_score += 0.1
        
        return min(partial_score, 0.5)


class EnhancedReadFileTool(BaseTool):
    """
    Enhanced file reading with NO content truncation.
    Can read by full path or search result index.
    """
    
    def __init__(self, allowed_dirs: List[Path] = None, max_file_size_mb: int = 50):
        super().__init__()
        self.allowed_dirs = allowed_dirs or []
        self.max_file_size_mb = max_file_size_mb
    
    @property
    def name(self) -> str:
        return "read_file_complete"
    
    @property
    def description(self) -> str:
        return "Read complete file content with NO truncation. Provide file path or name."
    
    def _execute_impl(self, file_path: str, **kwargs) -> ToolResult:
        """Read complete file content."""
        path = Path(file_path)
        
        # If not absolute path, search for it
        if not path.is_absolute():
            search_result = self._find_file(file_path)
            if not search_result:
                return ToolResult(
                    success=False,
                    error=f"File not found: {file_path}. Try using search_files tool first.",
                    data=None
                )
            path = search_result
        
        # Validate file
        if not path.exists():
            return ToolResult(
                success=False,
                error=f"File does not exist: {path}",
                data=None
            )
        
        # Check size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            return ToolResult(
                success=False,
                error=f"File too large: {size_mb:.2f}MB (max: {self.max_file_size_mb}MB)",
                data=None
            )
        
        # Read complete content
        try:
            content = self._read_with_encoding_detection(path)
            
            return ToolResult(
                success=True,
                data={
                    "content": content,  # COMPLETE content, NO truncation
                    "file_path": str(path),
                    "file_name": path.name,
                    "size_bytes": path.stat().st_size,
                    "lines": content.count('\n') + 1,
                    "characters": len(content)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to read file: {str(e)}",
                data=None
            )
    
    def _find_file(self, filename: str) -> Optional[Path]:
        """Search for file in allowed directories."""
        filename_lower = filename.lower()
        
        for search_dir in self.allowed_dirs:
            if not search_dir.exists():
                continue
            
            for file_path in search_dir.rglob("*"):
                if file_path.is_file() and file_path.name.lower() == filename_lower:
                    return file_path
        
        return None
    
    def _read_with_encoding_detection(self, path: Path) -> str:
        """Read file with automatic encoding detection."""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file with encodings: {encodings}")


class InteractiveFileReadTool(BaseTool):
    """
    Combines search and read in one tool.
    If multiple files match, returns list for user to choose.
    If single match, reads it automatically.
    """
    
    def __init__(self, search_dirs: List[Path] = None, max_file_size_mb: int = 50):
        super().__init__()
        self.search_tool = SmartFileSearchTool(search_dirs)
        self.read_tool = EnhancedReadFileTool(search_dirs, max_file_size_mb)
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read file by name or path. Searches automatically if needed."
    
    # Complete InteractiveFileReadTool._execute_impl() for app/tools/enhanced_file_tools.py

    def _execute_impl(self, file_identifier: str, **kwargs) -> ToolResult:
        """
        Smart file reading with automatic best-match selection.
        
        Flow:
        1. If exact path exists -> read it
        2. Otherwise search for file by name
        3. If single match -> read it
        4. If multiple matches -> auto-select best one (prefers knowledge dir)
        5. If no matches -> return error
        
        Args:
            file_identifier: File name, pattern, or full path
            
        Returns:
            ToolResult with file content or error
        """
        path = Path(file_identifier)
        
        # ========================================================================
        # Case 1: Exact path provided and exists
        # ========================================================================
        if path.exists() and path.is_file():
            logger.info(f"Reading file from exact path: {path}")
            return self.read_tool._execute_impl(str(path))
        
        # ========================================================================
        # Case 2: Search for file by name/pattern
        # ========================================================================
        logger.info(f"Searching for file matching: {file_identifier}")
        search_result = self.search_tool._execute_impl(file_identifier)
        
        if not search_result.success:
            logger.warning(f"Search failed: {search_result.error}")
            return search_result
        
        matches = search_result.data.get("matches", [])
        
        # ========================================================================
        # Case 2a: No matches found
        # ========================================================================
        if not matches:
            logger.info(f"No files found matching: {file_identifier}")
            return ToolResult(
                success=False,
                error=f"No files found matching '{file_identifier}'",
                data={
                    "searched_query": file_identifier,
                    "suggestion": "Try using 'search_files' tool to see all available files"
                }
            )
        
        # ========================================================================
        # Case 2b: Single match - read it automatically
        # ========================================================================
        if len(matches) == 1:
            file_path = matches[0]["path"]
            logger.info(f"Single match found, reading: {file_path}")
            
            read_result = self.read_tool._execute_impl(file_path)
            
            # Add search metadata to result
            if read_result.success:
                read_result.data["matched_from_search"] = True
                read_result.data["search_query"] = file_identifier
                read_result.data["relevance_score"] = matches[0].get("relevance", 1.0)
            
            return read_result
        
        # ========================================================================
        # Case 2c: Multiple matches - AUTO-SELECT best one
        # ========================================================================
        logger.info(f"Found {len(matches)} files matching '{file_identifier}'")
        
        # Strategy: Prefer knowledge directory, then highest relevance
        from app.config import KNOWLEDGE_DIR
        
        best_match = None
        selection_reason = ""
        
        # Strategy 1: Look for file in knowledge directory
        for match in matches:
            match_path = Path(match["path"])
            try:
                # Check if path is from network share (preferred)
                if "network" in str(match_path).lower() or str(match_path).startswith(("Z:", "z:")):
                    best_match = match
                    selection_reason = "from network share"
                    logger.info(f"Selected file from network share: {match['path']}")
                    break
            except Exception:
                continue
        
        # Strategy 2: If no knowledge dir match, use highest relevance
        if not best_match:
            best_match = matches[0]  # Already sorted by relevance
            selection_reason = f"highest relevance ({best_match.get('relevance', 0):.2f})"
            logger.info(f"Selected file with highest relevance: {best_match['path']}")
        
        # Read the selected file
        selected_path = best_match["path"]
        read_result = self.read_tool._execute_impl(selected_path)
        
        # Add metadata about the selection
        if read_result.success:
            read_result.data["auto_selected"] = True
            read_result.data["selected_path"] = selected_path
            read_result.data["selection_reason"] = selection_reason
            read_result.data["total_matches"] = len(matches)
            
            # Include paths of other matches (max 5 for readability)
            other_matches = [
                m["path"] for m in matches 
                if m["path"] != selected_path
            ]
            if other_matches:
                read_result.data["other_versions"] = other_matches[:5]
            
            logger.info(f"Successfully read auto-selected file. {len(other_matches)} other versions available.")
        else:
            # Read failed, but add context about what we tried
            read_result.data["attempted_path"] = selected_path
            read_result.data["total_matches"] = len(matches)
            logger.error(f"Failed to read selected file: {selected_path}")
        
        return read_result


    # ============================================================================
    # ALTERNATIVE: If you want to keep the "ask user" behavior
    # ============================================================================

    def _execute_impl_with_user_choice(self, file_identifier: str, **kwargs) -> ToolResult:
        """
        Alternative implementation that asks user to choose when multiple matches.
        Use this if you prefer explicit user selection over auto-selection.
        """
        path = Path(file_identifier)
        
        # Case 1: Exact path exists
        if path.exists() and path.is_file():
            return self.read_tool._execute_impl(str(path))
        
        # Case 2: Search for file
        search_result = self.search_tool._execute_impl(file_identifier)
        
        if not search_result.success:
            return search_result
        
        matches = search_result.data.get("matches", [])
        
        # No matches
        if not matches:
            return ToolResult(
                success=False,
                error=f"No files found matching '{file_identifier}'",
                data={"searched_query": file_identifier}
            )
        
        # Single match - read it
        if len(matches) == 1:
            file_path = matches[0]["path"]
            logger.info(f"Single file match found, reading: {file_path}")
            
            read_result = self.read_tool._execute_impl(file_path)
            
            if read_result.success:
                read_result.data["matched_from_search"] = True
                read_result.data["search_query"] = file_identifier
            
            return read_result
        
        # Multiple matches - ask user to clarify
        logger.info(f"Multiple matches ({len(matches)}), asking user to choose")
        
        return ToolResult(
            success=True,  # Not an error, just needs clarification
            data={
                "action_required": "choose_file",
                "message": f"Found {len(matches)} files matching '{file_identifier}'. Which one would you like to read?",
                "query": file_identifier,
                "matches": matches,
                "instruction": "Please specify the exact file name or provide the full path from the list above."
            }
        )
    
def create_enhanced_file_tools(search_dirs: List[Path]) -> Dict[str, BaseTool]:
    """Factory function to create all enhanced file tools."""
    return {
        "search_files": SmartFileSearchTool(search_dirs),
        "read_file_complete": EnhancedReadFileTool(search_dirs),
        "read_file": InteractiveFileReadTool(search_dirs)
    }
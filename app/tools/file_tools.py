# app/tools/file_tools.py
"""
File system tools for reading, writing, and listing files.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from app.tools.models import BaseTool, ToolResult
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ReadFileTool(BaseTool):
    """Tool for reading file contents."""
    
    def __init__(self, allowed_dirs: List[Path] = None, max_file_size_mb: int = 10):
        super().__init__()
        self.allowed_dirs = allowed_dirs or [Path.cwd()]
        self.max_file_size_mb = max_file_size_mb
        self.add_validator(self._validate_file_path)
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read contents of a text file"
    
    def _validate_file_path(self, params: Dict[str, Any]) -> Optional[str]:
        if "file_path" not in params:
            return "Missing required parameter: file_path"
        
        file_path = Path(params["file_path"])
        
        if not self._is_path_allowed(file_path):
            return f"Access denied: {file_path}"
        
        if not file_path.exists():
            return f"File not found: {file_path}"
        
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            return f"File too large: {size_mb:.2f}MB (max: {self.max_file_size_mb}MB)"
        
        return None
    
    def _is_path_allowed(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            for allowed_dir in self.allowed_dirs:
                try:
                    resolved.relative_to(allowed_dir.resolve())
                    return True
                except ValueError:
                    continue
            return False
        except Exception:
            return False
    
    def _execute_impl(self, file_path: str, **kwargs) -> ToolResult:
        path = Path(file_path)
        
        try:
            content = self._read_with_encoding_fallback(path)
            
            if content is None:
                return ToolResult(
                    success=False,
                    error="Could not decode file",
                    data=None
                )
            
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "file_path": str(path),
                    "file_size_bytes": path.stat().st_size,
                    "lines": content.count('\n') + 1
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), data=None)
    
    def _read_with_encoding_fallback(self, path: Path) -> Optional[str]:
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return None


class WriteFileTool(BaseTool):
    """Tool for writing files."""
    
    def __init__(self, allowed_dirs: List[Path] = None):
        super().__init__()
        self.allowed_dirs = allowed_dirs or [Path.cwd() / "outputs"]
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "Write content to a file"
    
    def _execute_impl(self, file_path: str, content: str, **kwargs) -> ToolResult:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            path.write_text(content, encoding='utf-8')
            return ToolResult(
                success=True,
                data={
                    "file_path": str(path),
                    "bytes_written": len(content.encode('utf-8'))
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), data=None)


class ListFilesTool(BaseTool):
    """Tool for listing directory contents."""
    
    def __init__(self, allowed_dirs: List[Path] = None):
        super().__init__()
        self.allowed_dirs = allowed_dirs or [Path.cwd()]
    
    @property
    def name(self) -> str:
        return "list_files"
    
    @property
    def description(self) -> str:
        return "List files in a directory"
    
    def _execute_impl(self, directory: str = ".", pattern: str = "*", **kwargs) -> ToolResult:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return ToolResult(
                success=False,
                error=f"Directory not found: {directory}",
                data=None
            )
        
        try:
            files = []
            for item in dir_path.glob(pattern):
                if item.is_file():
                    files.append({
                        "name": item.name,
                        "path": str(item),
                        "size_bytes": item.stat().st_size,
                        "modified": item.stat().st_mtime
                    })
            
            return ToolResult(
                success=True,
                data={
                    "directory": str(dir_path),
                    "pattern": pattern,
                    "files": files,
                    "count": len(files)
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), data=None)
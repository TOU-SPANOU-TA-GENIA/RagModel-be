# app/tools/base.py
"""
Simplified Tool System.
Clean, modular tools that are easy to create, test, and debug.
"""
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
import json
from app.config import KNOWLEDGE_DIR, DATA_DIR, BASE_DIR
from app.core.interfaces import Tool
from app.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Tool Result Model
# ============================================================================

@dataclass
class ToolResult:
    """Standardized tool result."""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time
        }


# ============================================================================
# Base Tool Implementation
# ============================================================================

class BaseTool(Tool):
    """
    Base class for all tools with common functionality.
    Much simpler than the original Tool class!
    """
    
    def __init__(self):
        self._validators: List[Callable] = []
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with timing and error handling."""
        start_time = time.time()
        
        try:
            # Validate parameters
            validation_error = self._validate_params(kwargs)
            if validation_error:
                logger.warning(f"Tool {self.name} validation failed: {validation_error}")
                return ToolResult(
                    success=False,
                    error=validation_error,
                    execution_time=time.time() - start_time
                ).to_dict()
            
            # Execute tool logic
            logger.info(f"Executing tool: {self.name}")
            result = self._execute_impl(**kwargs)
            
            # Ensure result is a ToolResult
            if not isinstance(result, ToolResult):
                result = ToolResult(success=True, data=result)
            
            result.execution_time = time.time() - start_time
            logger.info(f"Tool {self.name} executed in {result.execution_time:.2f}s")
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            ).to_dict()
    
    @abstractmethod
    def _execute_impl(self, **kwargs) -> ToolResult:
        """Actual tool implementation."""
        pass
    
    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate parameters."""
        # Run custom validators
        for validator in self._validators:
            error = validator(params)
            if error:
                return error
        return None
    
    def add_validator(self, validator: Callable) -> None:
        """Add a parameter validator."""
        self._validators.append(validator)


# ============================================================================
# File System Tools
# ============================================================================

class ReadFileTool(BaseTool):
    """
    Simplified file reading tool.
    Much cleaner than the original implementation!
    """
    
    def __init__(self, allowed_dirs: List[Path] = None, max_file_size_mb: int = 10):
        super().__init__()
        self.allowed_dirs = allowed_dirs or [Path.cwd()]
        self.max_file_size_mb = max_file_size_mb
        
        # Add validators
        self.add_validator(self._validate_file_path)
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read contents of a text file from the filesystem"
    
    def _validate_file_path(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate file path parameter."""
        if "file_path" not in params:
            return "Missing required parameter: file_path"
        
        file_path = Path(params["file_path"])
        
        # Check if path is allowed
        if not self._is_path_allowed(file_path):
            return f"Access denied: {file_path} is not in allowed directories"
        
        # Check if file exists
        if not file_path.exists():
            return f"File not found: {file_path}"
        
        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            return f"File too large: {size_mb:.2f}MB (max: {self.max_file_size_mb}MB)"
        
        return None
    
    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is within allowed directories."""
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
        """Read the file."""
        path = Path(file_path)
        
        try:
            # Try different encodings
            content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
                try:
                    content = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return ToolResult(
                    success=False,
                    error="Could not decode file with any supported encoding",
                    data=None
                )
            
            # Prepare result data
            data = {
                "content": content,
                "file_path": str(path),
                "file_size_bytes": path.stat().st_size,
                "lines": content.count('\n') + 1
            }
            
            return ToolResult(success=True, data=data)
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to read file: {str(e)}",
                data=None
            )


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
        """Write content to file."""
        path = Path(file_path)
        
        # Ensure parent directory exists
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
            return ToolResult(
                success=False,
                error=f"Failed to write file: {str(e)}",
                data=None
            )


class ListFilesTool(BaseTool):
    """Tool for listing files in a directory."""
    
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
        """List files in directory."""
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
            return ToolResult(
                success=False,
                error=f"Failed to list files: {str(e)}",
                data=None
            )


# ============================================================================
# System Tools
# ============================================================================

class ExecuteCommandTool(BaseTool):
    """Tool for executing system commands (use with caution!)."""
    
    def __init__(self, allowed_commands: List[str] = None):
        super().__init__()
        self.allowed_commands = allowed_commands or ["ls", "pwd", "echo", "date"]
    
    @property
    def name(self) -> str:
        return "execute_command"
    
    @property
    def description(self) -> str:
        return "Execute a system command (limited commands allowed)"
    
    def _execute_impl(self, command: str, **kwargs) -> ToolResult:
        """Execute command."""
        # Check if command is allowed
        base_command = command.split()[0] if command else ""
        if base_command not in self.allowed_commands:
            return ToolResult(
                success=False,
                error=f"Command not allowed: {base_command}",
                data=None
            )
        
        try:
            import subprocess
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            return ToolResult(
                success=result.returncode == 0,
                data={
                    "command": command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                },
                error=result.stderr if result.returncode != 0 else None
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error="Command timed out",
                data=None
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to execute command: {str(e)}",
                data=None
            )


# ============================================================================
# Database Tools (placeholder for future)
# ============================================================================

class DatabaseQueryTool(BaseTool):
    """Tool for querying databases (placeholder)."""
    
    def __init__(self, connection_string: str = None):
        super().__init__()
        self.connection_string = connection_string
    
    @property
    def name(self) -> str:
        return "database_query"
    
    @property
    def description(self) -> str:
        return "Query a database"
    
    def _execute_impl(self, query: str, **kwargs) -> ToolResult:
        """Execute database query."""
        # This is a placeholder
        # In production, you'd implement actual database connection
        return ToolResult(
            success=False,
            error="Database tool not implemented yet",
            data=None
        )


# ============================================================================
# Tool Registry
# ============================================================================

class SimpleToolRegistry:
    """
    Simple tool registry for managing tools.
    Much cleaner than the original ToolRegistry!
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Unregistered tool: {name}")
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all tool names."""
        return list(self.tools.keys())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get all tool descriptions."""
        return {
            name: tool.description 
            for name, tool in self.tools.items()
        }
    
    def execute(self, tool_name: str, **params) -> Dict[str, Any]:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        
        if not tool:
            logger.error(f"Tool not found: {tool_name}")
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}",
                data=None
            ).to_dict()
        
        return tool.execute(**params)


# ============================================================================
# Tool Chains (for complex operations)
# ============================================================================

class ToolChain:
    """
    Chain multiple tools together for complex operations.
    This is for future enhancement.
    """
    
    def __init__(self, registry: SimpleToolRegistry):
        self.registry = registry
        self.steps: List[Dict[str, Any]] = []
    
    def add_step(self, tool_name: str, params: Dict[str, Any]) -> 'ToolChain':
        """Add a step to the chain."""
        self.steps.append({"tool": tool_name, "params": params})
        return self
    
    def execute(self) -> List[Dict[str, Any]]:
        """Execute all steps in order."""
        results = []
        
        for step in self.steps:
            result = self.registry.execute(
                step["tool"],
                **step["params"]
            )
            results.append(result)
            
            # Stop if a step fails
            if not result.get("success", False):
                break
        
        return results


# ============================================================================
# Factory Functions
# ============================================================================

# app/tools/base.py - UPDATED create_default_tools() function
# Replace the existing function with this version
def create_default_tools() -> SimpleToolRegistry:
    """Create a registry with enhanced file tools."""
    registry = SimpleToolRegistry()
    
    # Import enhanced tools
    try:
        from app.tools.enhanced_file_tools import create_enhanced_file_tools
        
        # Determine search directories
        search_dirs = [KNOWLEDGE_DIR, DATA_DIR, BASE_DIR / "logs"]
        
        # Add enhanced file tools
        enhanced_tools = create_enhanced_file_tools(search_dirs)
        for tool in enhanced_tools.values():
            registry.register(tool)
        
        logger.info("Registered enhanced file tools with smart search")
    
    except ImportError as e:
        logger.warning(f"Enhanced file tools not available: {e}, using basic tools")
        # Fallback to basic read tool
        registry.register(ReadFileTool(allowed_dirs=[KNOWLEDGE_DIR, DATA_DIR]))
    
    # Add other file system tools
    registry.register(WriteFileTool())
    registry.register(ListFilesTool())
    
    # Add system tools (with restrictions)
    registry.register(ExecuteCommandTool())
    
    # Add document generator tool (LAZY IMPORT HERE) ✅
    try:
        from app.tools.document_generator import DocumentGeneratorTool
        registry.register(DocumentGeneratorTool())
        logger.info("Registered document generator tool")
    except ImportError as e:
        logger.warning(f"Document generator not available: {e}")
    
    logger.info(f"Created tool registry with {len(registry.tools)} tools")
    return registry


def create_tool_registry_for_military() -> SimpleToolRegistry:
    """Enhanced military registry with smart file tools."""
    registry = SimpleToolRegistry()
    
    # Configure allowed directories for military network
    military_dirs = [
        KNOWLEDGE_DIR,
        DATA_DIR,
        BASE_DIR / "logs",
    ]
    
    # Import and use enhanced tools
    try:
        from app.tools.enhanced_file_tools import create_enhanced_file_tools
        
        enhanced_tools = create_enhanced_file_tools(military_dirs)
        for tool in enhanced_tools.values():
            registry.register(tool)
    
    except ImportError:
        # Fallback
        read_tool = ReadFileTool(allowed_dirs=military_dirs)
        registry.register(read_tool)
    
    # Add write tool with restricted output directory
    write_tool = WriteFileTool(allowed_dirs=[Path("/data/outputs")])
    registry.register(write_tool)
    
    # Add restricted command tool
    safe_commands = ["ls", "pwd", "date", "whoami", "df", "free"]
    registry.register(ExecuteCommandTool(allowed_commands=safe_commands))
    
    logger.info("Created military-configured tool registry with enhanced file tools")
    return registry




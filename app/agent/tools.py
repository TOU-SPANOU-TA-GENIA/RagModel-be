# app/agent/tools.py
"""
Tool System for AI Agent

This module defines the base Tool class and implements specific tools
that the agent can use. Each tool is self-contained and follows a 
consistent interface for easy debugging and extension.

Design Philosophy:
- Each tool is independent and testable
- Tools return structured results (success, data, error)
- Tools validate their inputs before execution
- Tools log their actions for debugging
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
import json

from ..logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ToolResult:
    """
    Standardized result from tool execution.
    
    Attributes:
        success: Whether the tool executed successfully
        data: The result data (structure depends on tool)
        error: Error message if success=False
        tool_name: Name of the tool that produced this result
    """
    success: bool
    data: Any
    error: Optional[str] = None
    tool_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "tool_name": self.tool_name
        }


class Tool(ABC):
    """
    Base class for all agent tools.
    
    Every tool must:
    1. Have a unique name
    2. Provide a description for the LLM
    3. Define required parameters
    4. Implement the execute method
    5. Validate inputs before execution
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """What this tool does - shown to the LLM."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Parameter schema for this tool.
        
        Format:
        {
            "param_name": {
                "type": "string|number|boolean",
                "description": "what this parameter does",
                "required": True|False,
                "default": value  # optional
            }
        }
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Parameters as defined in self.parameters
            
        Returns:
            ToolResult with success status and data/error
        """
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate that provided parameters match the schema.
        
        Returns:
            (is_valid, error_message)
        """
        for param_name, param_schema in self.parameters.items():
            if param_schema.get("required", False):
                if param_name not in params:
                    return False, f"Missing required parameter: {param_name}"
        
        return True, None
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the complete tool schema for the LLM prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class ReadFileTool(Tool):
    """
    Tool for reading file contents from the local filesystem.
    
    Security considerations:
    - Only reads from whitelisted directories
    - Validates file paths to prevent directory traversal
    - Limits file size to prevent memory issues
    - Only reads text files (configurable extensions)
    
    Example usage by agent:
        "I need to read the config file at /data/config.txt"
        -> Tool reads and returns file content
    """
    
    # Configuration
    ALLOWED_DIRECTORIES = [
        Path("/data"),
        Path("/app/data"),
        Path("/home/user/documents"),  # Adjust to your server paths
    ]
    
    ALLOWED_EXTENSIONS = {'.txt', '.md', '.json', '.yaml', '.yml', '.conf', '.log', '.py'}
    MAX_FILE_SIZE_MB = 10
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return (
            "Read the contents of a text file from the local filesystem. "
            "Use this when the user asks to read, view, or check the contents of a file. "
            "Supports text files, markdown, JSON, YAML, config files, and logs."
        )
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the file to read",
                "required": True
            },
            "max_lines": {
                "type": "number",
                "description": "Maximum number of lines to read (optional, for large files)",
                "required": False,
                "default": None
            }
        }
    
    def _is_path_allowed(self, file_path: Path) -> bool:
        """Check if the file path is within allowed directories."""
        try:
            # Resolve to absolute path to prevent traversal attacks
            resolved_path = file_path.resolve()
            
            # Check if it's within any allowed directory
            for allowed_dir in self.ALLOWED_DIRECTORIES:
                try:
                    resolved_path.relative_to(allowed_dir.resolve())
                    return True
                except ValueError:
                    continue
            
            return False
        except Exception as e:
            logger.error(f"Error validating path: {e}")
            return False
    
    def _is_extension_allowed(self, file_path: Path) -> bool:
        """Check if the file extension is allowed."""
        return file_path.suffix.lower() in self.ALLOWED_EXTENSIONS
    
    def execute(self, file_path: str, max_lines: Optional[int] = None, **kwargs) -> ToolResult:
        """
        Read and return the contents of a file.
        
        Args:
            file_path: Path to the file to read
            max_lines: Optional limit on number of lines to read
            
        Returns:
            ToolResult with file contents or error
        """
        try:
            logger.info(f"ReadFileTool: Attempting to read {file_path}")
            
            # Convert to Path object
            path = Path(file_path)
            
            # Security checks
            if not self._is_path_allowed(path):
                error_msg = f"Access denied: {file_path} is not in an allowed directory"
                logger.warning(error_msg)
                return ToolResult(
                    success=False,
                    data=None,
                    error=error_msg,
                    tool_name=self.name
                )
            
            if not self._is_extension_allowed(path):
                error_msg = f"File type not supported: {path.suffix}"
                logger.warning(error_msg)
                return ToolResult(
                    success=False,
                    data=None,
                    error=error_msg,
                    tool_name=self.name
                )
            
            # Check if file exists
            if not path.exists():
                error_msg = f"File not found: {file_path}"
                logger.warning(error_msg)
                return ToolResult(
                    success=False,
                    data=None,
                    error=error_msg,
                    tool_name=self.name
                )
            
            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.MAX_FILE_SIZE_MB:
                error_msg = f"File too large: {file_size_mb:.2f}MB (max: {self.MAX_FILE_SIZE_MB}MB)"
                logger.warning(error_msg)
                return ToolResult(
                    success=False,
                    data=None,
                    error=error_msg,
                    tool_name=self.name
                )
            
            # Read the file
            encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        if max_lines:
                            lines = []
                            for i, line in enumerate(f):
                                if i >= max_lines:
                                    lines.append(f"\n... (truncated at {max_lines} lines)")
                                    break
                                lines.append(line)
                            content = ''.join(lines)
                        else:
                            content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                error_msg = f"Could not decode file with any supported encoding"
                logger.error(error_msg)
                return ToolResult(
                    success=False,
                    data=None,
                    error=error_msg,
                    tool_name=self.name
                )
            
            logger.info(f"ReadFileTool: Successfully read {len(content)} characters from {file_path}")
            
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "file_path": str(path),
                    "file_size_bytes": path.stat().st_size,
                    "lines": content.count('\n') + 1 if content else 0
                },
                error=None,
                tool_name=self.name
            )
            
        except Exception as e:
            error_msg = f"Unexpected error reading file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                tool_name=self.name
            )


class ToolRegistry:
    """
    Central registry for all available tools.
    
    The agent queries this registry to:
    1. Get available tools for prompt building
    2. Execute tools by name
    3. Validate tool availability
    
    Usage:
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        result = registry.execute_tool("read_file", file_path="/data/test.txt")
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        logger.info("ToolRegistry initialized")
    
    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        if tool.name in self._tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")
        
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools (used in agent prompt)."""
        return [tool.get_schema() for tool in self._tools.values()]
    
    def execute_tool(self, tool_name: str, **params) -> ToolResult:
        """
        Execute a tool by name with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **params: Parameters to pass to the tool
            
        Returns:
            ToolResult from tool execution
        """
        tool = self.get_tool(tool_name)
        
        if tool is None:
            logger.error(f"Tool not found: {tool_name}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}",
                tool_name=tool_name
            )
        
        # Validate parameters
        is_valid, error_msg = tool.validate_parameters(params)
        if not is_valid:
            logger.error(f"Invalid parameters for {tool_name}: {error_msg}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid parameters: {error_msg}",
                tool_name=tool_name
            )
        
        # Execute tool
        logger.info(f"Executing tool: {tool_name} with params: {list(params.keys())}")
        try:
            result = tool.execute(**params)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool execution error: {str(e)}",
                tool_name=tool_name
            )


# Global registry instance
_registry = None

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        # Register default tools
        _registry.register(ReadFileTool())
    return _registry
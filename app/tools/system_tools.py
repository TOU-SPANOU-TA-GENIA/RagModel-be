# app/tools/system_tools.py
"""
System command tools.
"""

from typing import Dict, Any, List, Optional
import subprocess

from app.tools.models import BaseTool, ToolResult
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ExecuteCommandTool(BaseTool):
    """Tool for executing allowed system commands."""
    
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
        base_command = command.split()[0] if command else ""
        
        if base_command not in self.allowed_commands:
            return ToolResult(
                success=False,
                error=f"Command not allowed: {base_command}",
                data=None
            )
        
        try:
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
            return ToolResult(success=False, error="Command timed out", data=None)
        except Exception as e:
            return ToolResult(success=False, error=str(e), data=None)


class DatabaseQueryTool(BaseTool):
    """Placeholder for database queries."""
    
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
        return ToolResult(
            success=False,
            error="Database tool not implemented",
            data=None
        )
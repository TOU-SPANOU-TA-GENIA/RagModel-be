from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Type
from pydantic import BaseModel

class ToolResult(BaseModel):
    """Standard output for any tool execution."""
    success: bool
    output: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BaseTool(ABC):
    """
    Abstract base class for all tools.
    Implements the Core interface but adds Pydantic schema validation for arguments.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name used by the LLM to call this tool (e.g., 'read_file')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Instruction for the LLM on when/how to use this tool."""
        pass
    
    @property
    @abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        """Pydantic model describing the input arguments."""
        pass

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """The implementation logic."""
        pass

    # Bridge to the Core interface which expects 'execute'
    def execute(self, **kwargs) -> Any:
        try:
            # Validate args against schema
            validated_args = self.args_schema(**kwargs)
            return self.run(**validated_args.dict())
        except Exception as e:
            return ToolResult(
                success=False, 
                output=f"Tool execution failed: {str(e)}", 
                error=str(e)
            )
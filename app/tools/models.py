# app/tools/models.py
"""
Tool data models and base classes.
"""

from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

from app.core.interfaces import Tool
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ToolResult:
    """Standardized tool execution result."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time
        }


class BaseTool(Tool):
    """
    Base class for all tools with common functionality.
    """
    
    def __init__(self):
        self._validators: List[Callable] = []
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with timing and error handling."""
        start_time = time.time()
        
        try:
            validation_error = self._validate_params(kwargs)
            if validation_error:
                logger.warning(f"Tool {self.name} validation failed: {validation_error}")
                return ToolResult(
                    success=False,
                    error=validation_error,
                    execution_time=time.time() - start_time
                ).to_dict()
            
            logger.info(f"Executing tool: {self.name}")
            result = self._execute_impl(**kwargs)
            
            if not isinstance(result, ToolResult):
                result = ToolResult(success=True, data=result)
            
            result.execution_time = time.time() - start_time
            logger.info(f"Tool {self.name} completed in {result.execution_time:.2f}s")
            
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
        """Tool implementation - override in subclasses."""
        pass
    
    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Run parameter validators."""
        for validator in self._validators:
            error = validator(params)
            if error:
                return error
        return None
    
    def add_validator(self, validator: Callable) -> None:
        """Add a parameter validator."""
        self._validators.append(validator)


class SimpleToolRegistry:
    """
    Registry for managing tools.
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> None:
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Unregistered tool: {name}")
    
    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        return list(self.tools.keys())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self.tools.items()}
    
    def execute(self, tool_name: str, **params) -> Dict[str, Any]:
        tool = self.get(tool_name)
        
        if not tool:
            logger.error(f"Tool not found: {tool_name}")
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}",
                data=None
            ).to_dict()
        
        return tool.execute(**params)


class ToolChain:
    """Chain multiple tools for complex operations."""
    
    def __init__(self, registry: SimpleToolRegistry):
        self.registry = registry
        self.steps: List[Dict[str, Any]] = []
    
    def add_step(self, tool_name: str, params: Dict[str, Any]) -> 'ToolChain':
        self.steps.append({"tool": tool_name, "params": params})
        return self
    
    def execute(self) -> List[Dict[str, Any]]:
        results = []
        
        for step in self.steps:
            result = self.registry.execute(step["tool"], **step["params"])
            results.append(result)
            
            if not result.get("success", False):
                break
        
        return results
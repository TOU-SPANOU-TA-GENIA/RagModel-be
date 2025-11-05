# app/agent/__init__.py
"""
Agent Module

This module provides AI agent capabilities with tool usage.

Public API:
    - Agent: Main agent class
    - create_agent: Factory function to create configured agent
    - ToolRegistry: Registry for managing tools
    - Tool: Base class for creating new tools
    - ToolResult: Standardized tool execution result

Quick Start:
    from app.agent import create_agent
    
    agent = create_agent()
    response = agent.process_query("Read file at /data/config.txt")
    print(response['answer'])
"""

from .core import (
    Agent,
    create_agent,
    AgentIntent,
    AgentDecision
)

from .tools import (
    Tool,
    ToolResult,
    ToolRegistry,
    get_tool_registry,
    ReadFileTool
)

__all__ = [
    # Core agent
    "Agent",
    "create_agent",
    "AgentIntent",
    "AgentDecision",
    
    # Tools
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "get_tool_registry",
    "ReadFileTool",
]
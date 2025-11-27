# app/__init__.py
"""
AI Agent RAG Application

A modular AI agent system with RAG capabilities.
"""

__version__ = "2.0.0"
__author__ = "Panos Kafantaris"

# Core exports for convenience
from app.config import (
    config,
    LLM, RAG, AGENT, TOOLS, PATHS,
    SYSTEM_INSTRUCTION
)
from app.core import (
    Context, Intent, Decision,
    RAGException,
    ChatNotFoundException,
    startup_manager
)
from app.agent.integration import create_agent, get_agent

__all__ = [
    "__version__",
    "__author__",
    
    # Config
    "config",
    "LLM", "RAG", "AGENT", "TOOLS", "PATHS",
    "SYSTEM_INSTRUCTION",
    
    # Core
    "Context", "Intent", "Decision",
    "RAGException", "ChatNotFoundException",
    "startup_manager",
    
    # Agent
    "create_agent", "get_agent",
]
__version__ = "2.0.0"
__author__ = "Panos Kafantaris"

from .config import (
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    KNOWLEDGE_DIR,
    INDEX_DIR,
    SYSTEM_INSTRUCTION,
    AGENT_MODE,
    AGENT_USE_CACHE,
    AGENT_DEBUG_MODE
)

from .exceptions import (
    RAGException,
    ChatNotFoundException,
    VectorStoreNotInitializedException,
    ModelLoadException,
    IngestionException
)

# New imports
from .agent.integration import create_agent, get_agent
from .core.interfaces import Context, Intent, Decision

__all__ = [
    "__version__",
    "__author__",
    "LLM_MODEL_NAME",
    "EMBEDDING_MODEL_NAME",
    "KNOWLEDGE_DIR",
    "INDEX_DIR",
    "SYSTEM_INSTRUCTION",
    "AGENT_MODE",
    "AGENT_USE_CACHE",
    "AGENT_DEBUG_MODE",
    "RAGException",
    "ChatNotFoundException",
    "VectorStoreNotInitializedException",
    "ModelLoadException",
    "IngestionException",
    # New exports
    "create_agent",
    "get_agent",
    "Context",
    "Intent",
    "Decision",
]
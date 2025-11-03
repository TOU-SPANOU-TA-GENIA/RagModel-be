__version__ = "1.0.0"
__author__ = "Panos Kafantaris"

from .config import (
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    KNOWLEDGE_DIR,
    INDEX_DIR,
    SYSTEM_INSTRUCTION
)

from .exceptions import (
    RAGException,
    ChatNotFoundException,
    VectorStoreNotInitializedException,
    ModelLoadException,
    IngestionException
)

__all__ = [
    "__version__",
    "__author__",
    "LLM_MODEL_NAME",
    "EMBEDDING_MODEL_NAME",
    "KNOWLEDGE_DIR",
    "INDEX_DIR",
    "SYSTEM_INSTRUCTION",
    "RAGException",
    "ChatNotFoundException",
    "VectorStoreNotInitializedException",
    "ModelLoadException",
    "IngestionException",
]
# app/core/exceptions.py
"""
Application exceptions.
"""


class RAGException(Exception):
    """Base exception for RAG system."""
    pass


class ChatNotFoundException(RAGException):
    """Raised when a chat session is not found."""
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        super().__init__(f"Chat with ID '{chat_id}' not found")


class VectorStoreNotInitializedException(RAGException):
    """Raised when vector store is accessed before initialization."""
    def __init__(self):
        super().__init__(
            "Vector store not initialized. Please run ingestion first."
        )


class ModelLoadException(RAGException):
    """Raised when a model fails to load."""
    def __init__(self, model_name: str, original_error: Exception):
        self.model_name = model_name
        self.original_error = original_error
        super().__init__(
            f"Failed to load model '{model_name}': {str(original_error)}"
        )


class IngestionException(RAGException):
    """Raised when document ingestion fails."""
    def __init__(self, message: str):
        super().__init__(f"Ingestion failed: {message}")


class ConfigurationException(RAGException):
    """Raised for configuration errors."""
    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")


class ToolExecutionException(RAGException):
    """Raised when a tool fails to execute."""
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")
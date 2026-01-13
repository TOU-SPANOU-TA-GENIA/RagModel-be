class AppError(Exception):
    """Base exception for the application."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)

class ConfigurationError(AppError):
    """Raised when config is invalid or missing."""
    pass

class ProviderError(AppError):
    """Raised when an LLM or Embedding provider fails."""
    pass

class ToolError(AppError):
    """Raised when a tool execution fails."""
    pass
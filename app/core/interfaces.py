from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union, Generator
from dataclasses import dataclass

# We avoid importing specific schemas here to prevent circular imports.
# Components implementation will import the actual schemas.

class LLMProvider(ABC):
    """Abstract base class for Language Model providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response synchronously."""
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream a response token by token."""
        pass

class Retriever(ABC):
    """Abstract base class for RAG retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents.
        Returns: List of dicts with 'content' and 'metadata'.
        """
        pass

class Tool(ABC):
    """Abstract base class for Tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool logic."""
        pass

class MemoryStore(ABC):
    """Abstract base class for conversation persistence."""
    
    @abstractmethod
    def load_history(self, session_id: str) -> List[Dict[str, str]]:
        pass
        
    @abstractmethod
    def save_message(self, session_id: str, role: str, content: str):
        pass
        
    @abstractmethod
    def clear_history(self, session_id: str):
        pass
# app/rag/embedding_providers.py
"""
Embedding provider implementations with thread-safe GPU loading.
"""

import threading
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np
import torch

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# Global lock for model loading - prevents GPU contention
# =============================================================================
_model_loading_lock = threading.Lock()


def get_model_loading_lock() -> threading.Lock:
    """Get the global model loading lock for use by other modules."""
    return _model_loading_lock


# =============================================================================
# Base Class
# =============================================================================

class EmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass


# =============================================================================
# Singleton Registry for Models
# =============================================================================

class EmbeddingModelRegistry:
    """
    Singleton registry for embedding models.
    Ensures each model is loaded only once with thread-safe GPU access.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models = {}
                    cls._instance._model_lock = threading.Lock()
        return cls._instance
    
    def get_model(self, model_name: str):
        """Get or load model by name (thread-safe)."""
        with self._model_lock:
            if model_name not in self._models:
                self._models[model_name] = self._load_model(model_name)
            return self._models[model_name]
    
    def _load_model(self, model_name: str):
        """
        Load model with global lock to prevent GPU contention.
        
        Key fix: Load on CPU first, then move to GPU to avoid
        meta tensor issues from concurrent loading.
        """
        # Acquire global lock - only one model loads at a time
        with _model_loading_lock:
            logger.info(f"Loading embedding model (singleton): {model_name}")
            
            try:
                from sentence_transformers import SentenceTransformer
                
                # KEY FIX: Load on CPU first to avoid meta tensor issues
                # This prevents "Cannot copy out of meta tensor" errors
                model = SentenceTransformer(model_name, device='cpu')
                
                # Now safely move to GPU if available
                if torch.cuda.is_available():
                    model = model.to('cuda')
                    logger.info(f"Embedding model loaded on cuda")
                else:
                    logger.info(f"Embedding model loaded on cpu")
                
                return model
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def is_loaded(self, model_name: str) -> bool:
        """Check if model is already loaded."""
        return model_name in self._models
    
    def clear(self, model_name: Optional[str] = None):
        """Clear cached model(s)."""
        with self._model_lock:
            if model_name:
                self._models.pop(model_name, None)
                logger.info(f"Cleared embedding model: {model_name}")
            else:
                self._models.clear()
                logger.info("Cleared all embedding models")


# Global registry instance
_model_registry = EmbeddingModelRegistry()


# =============================================================================
# Local Embedding Provider
# =============================================================================

class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    Uses singleton registry to avoid duplicate model loading.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self._model = None  # Lazy loaded from registry
    
    @property
    def model(self):
        """Get model from singleton registry (lazy load)."""
        if self._model is None:
            self._model = _model_registry.get_model(self.model_name)
        return self._model
    
    def embed_text(self, text: str) -> List[float]:
        """Embed single text."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch of texts."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


# =============================================================================
# Mock Provider for Testing
# =============================================================================

class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_text(self, text: str) -> List[float]:
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        embedding = np.random.randn(self.dimension)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]


# =============================================================================
# Utility Functions
# =============================================================================

def preload_embedding_model(model_name: str = None):
    """
    Pre-load embedding model into registry.
    Call this during startup BEFORE LLM loading to avoid GPU contention.
    """
    model_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    if not _model_registry.is_loaded(model_name):
        logger.info(f"Pre-loading embedding model: {model_name}")
        _model_registry.get_model(model_name)
    else:
        logger.info(f"Embedding model already loaded: {model_name}")



_global_embedding_provider = None

def get_embedding_provider() -> LocalEmbeddingProvider:
    """Get the global embedding provider (ensures consistent model)."""
    global _global_embedding_provider
    
    if _global_embedding_provider is None:
        from app.config import EMBEDDING_MODEL_NAME
        _global_embedding_provider = LocalEmbeddingProvider(EMBEDDING_MODEL_NAME)
    
    return _global_embedding_provider
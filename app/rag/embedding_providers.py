# app/rag/embedding_providers.py
"""
Embedding provider implementations.
"""

from typing import List
from abc import ABC, abstractmethod
import numpy as np

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model = None
        self._initialized = False
    
    def _ensure_initialized(self):
        if not self._initialized:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = SentenceTransformer(self.model_name, device=device)
                self._initialized = True
                logger.info(f"Embedding model loaded on {device}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def embed_text(self, text: str) -> List[float]:
        self._ensure_initialized()
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._ensure_initialized()
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


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
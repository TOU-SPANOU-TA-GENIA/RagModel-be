# app/rag/vector_stores.py
"""
Vector store implementations.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Document:
    """Document model."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "metadata": self.metadata}


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    content: str
    metadata: Dict[str, Any]
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "metadata": self.metadata, "score": self.score}


class VectorStore(ABC):
    """Base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 3) -> List[RetrievalResult]:
        pass
    
    @abstractmethod
    def delete_all(self) -> None:
        pass


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using numpy."""
    
    def __init__(self):
        self.documents: List[Document] = []
        logger.info("InMemoryVectorStore initialized")
    
    def add_documents(self, documents: List[Document]) -> None:
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[RetrievalResult]:
        if not self.documents:
            return []
        
        similarities = []
        for doc in self.documents:
            if doc.embedding:
                score = self._cosine_similarity(query_embedding, doc.embedding)
                similarities.append((doc, score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        return [
            RetrievalResult(content=doc.content, metadata=doc.metadata, score=score)
            for doc, score in top_k
        ]
    
    def delete_all(self) -> None:
        self.documents = []
        logger.info("Vector store cleared")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1, vec2 = np.array(vec1), np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for production."""
    
    def __init__(self, index_path: Optional[str] = None, dimension: int = 384):
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.documents = []
        self._initialized = False
    
    def _ensure_initialized(self):
        if not self._initialized:
            try:
                import faiss
                
                if self.index_path and Path(self.index_path).exists():
                    self.index = faiss.read_index(str(self.index_path))
                    logger.info(f"Loaded FAISS index from {self.index_path}")
                else:
                    self.index = faiss.IndexFlatL2(self.dimension)
                    logger.info(f"Created new FAISS index")
                
                self._initialized = True
            except ImportError:
                logger.error("FAISS not installed")
                raise
    
    def add_documents(self, documents: List[Document]) -> None:
        self._ensure_initialized()
        
        embeddings = [doc.embedding for doc in documents if doc.embedding]
        if not embeddings:
            logger.warning("No documents with embeddings")
            return
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_array)
        self.documents.extend(documents)
        
        if self.index_path:
            import faiss
            index_path = Path(self.index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved index to {index_path}")
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[RetrievalResult]:
        self._ensure_initialized()
        
        if self.index.ntotal == 0:
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.documents):
                doc = self.documents[idx]
                score = 1.0 / (1.0 + distances[0][i])
                results.append(RetrievalResult(
                    content=doc.content,
                    metadata=doc.metadata,
                    score=score
                ))
        
        return results
    
    def delete_all(self) -> None:
        self._ensure_initialized()
        self.index.reset()
        self.documents = []
        logger.info("FAISS index cleared")
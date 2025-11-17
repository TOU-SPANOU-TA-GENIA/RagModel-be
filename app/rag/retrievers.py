# app/rag/retrievers.py
"""
Improved RAG and Retriever implementations.
Modular, testable, and easy to extend with different retrieval strategies.
"""

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np

from app.core.interfaces import Retriever
from app.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Document:
    """Simple document model."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    content: str
    metadata: Dict[str, Any]
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score
        }


# ============================================================================
# Embedding Providers
# ============================================================================

class EmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    Replaces the complex embedding logic in vectorstore.py
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy load the model."""
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
        """Embed a single text."""
        self._ensure_initialized()
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        self._ensure_initialized()
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Generate mock embedding based on text hash."""
        # Generate deterministic embedding from text
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        embedding = np.random.randn(self.dimension)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for batch."""
        return [self.embed_text(text) for text in texts]


# ============================================================================
# Vector Stores
# ============================================================================

class VectorStore(ABC):
    """Base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 3) -> List[RetrievalResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete_all(self) -> None:
        """Clear the store."""
        pass


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store for testing and small datasets.
    Much simpler than FAISS for understanding and debugging.
    """
    
    def __init__(self):
        self.documents: List[Document] = []
        logger.info("InMemoryVectorStore initialized")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to store."""
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[RetrievalResult]:
        """Search using cosine similarity."""
        if not self.documents:
            return []
        
        # Calculate similarities
        similarities = []
        for doc in self.documents:
            if doc.embedding:
                score = self._cosine_similarity(query_embedding, doc.embedding)
                similarities.append((doc, score))
        
        # Sort by score and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        # Convert to results
        results = [
            RetrievalResult(
                content=doc.content,
                metadata=doc.metadata,
                score=score
            )
            for doc, score in top_k
        ]
        
        return results
    
    def delete_all(self) -> None:
        """Clear all documents."""
        self.documents = []
        logger.info("Vector store cleared")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store for production use.
    Simplified wrapper around FAISS.
    """
    
    def __init__(self, index_path: Optional[str] = None, dimension: int = 384):
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.documents = []  # Store documents separately
        self._initialized = False
    
    def _ensure_initialized(self):
        """Initialize FAISS index."""
        if not self._initialized:
            try:
                import faiss
                
                if self.index_path and Path(self.index_path).exists():
                    # Load existing index
                    self.index = faiss.read_index(str(self.index_path))
                    logger.info(f"Loaded FAISS index from {self.index_path}")
                else:
                    # Create new index
                    self.index = faiss.IndexFlatL2(self.dimension)
                    logger.info(f"Created new FAISS index with dimension {self.dimension}")
                
                self._initialized = True
                
            except ImportError:
                logger.error("FAISS not installed. Falling back to in-memory store.")
                raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to FAISS index."""
        self._ensure_initialized()
        
        # Extract embeddings
        embeddings = [doc.embedding for doc in documents if doc.embedding]
        if not embeddings:
            logger.warning("No documents with embeddings to add")
            return
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        self.index.add(embeddings_array)
        self.documents.extend(documents)
        
        # Save index if path specified
        if self.index_path:
            import faiss
            # Ensure directory exists
            index_path = Path(self.index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved index to {index_path}")
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[RetrievalResult]:
        """Search FAISS index."""
        self._ensure_initialized()
        
        if self.index.ntotal == 0:
            return []
        
        # Convert to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        # Convert to results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]
                # Convert distance to similarity score (1 / (1 + distance))
                score = 1.0 / (1.0 + distances[0][i])
                
                results.append(RetrievalResult(
                    content=doc.content,
                    metadata=doc.metadata,
                    score=score
                ))
        
        return results
    
    def delete_all(self) -> None:
        """Clear the index."""
        self._ensure_initialized()
        self.index.reset()
        self.documents = []
        logger.info("FAISS index cleared")


# ============================================================================
# Retriever Implementations
# ============================================================================

class SimpleRetriever(Retriever):
    """
    Simple retriever that combines embedding and vector store.
    Much cleaner than the original retrieve function!
    """
    
    def __init__(self, 
                 embedding_provider: EmbeddingProvider,
                 vector_store: VectorStore,
                 min_score: float = 0.3):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.min_score = min_score
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for query."""
        start_time = time.time()
        
        try:
            # Embed query
            if hasattr(self.embedding_provider, 'embed_text'):
                query_embedding = self.embedding_provider.embed_text(query)
            else:
                # Handle list return from some providers
                query_embedding = self.embedding_provider.embed_text(query)
                if isinstance(query_embedding, list):
                    import numpy as np
                    query_embedding = np.array(query_embedding)
            
            # Search vector store
            if hasattr(self.vector_store, 'search'):
                results = self.vector_store.search(query_embedding, k=k)
            else:
                # Fallback for different store interfaces
                results = []
            
            # Filter by minimum score if results have scores
            filtered_results = []
            for r in results:
                if isinstance(r, dict):
                    score = r.get('score', 1.0)
                    if score >= self.min_score:
                        filtered_results.append(r)
                elif hasattr(r, 'score'):
                    if r.score >= self.min_score:
                        filtered_results.append(r.to_dict())
            
            elapsed = time.time() - start_time
            logger.info(f"Retrieved {len(filtered_results)} documents in {elapsed:.2f}s")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []


class HybridRetriever(Retriever):
    """
    Advanced retriever that combines multiple retrieval strategies.
    For future enhancement.
    """
    
    def __init__(self, 
                 semantic_retriever: SimpleRetriever,
                 keyword_retriever: Optional[Retriever] = None):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search."""
        # Get semantic results
        semantic_results = self.semantic_retriever.retrieve(query, k=k)
        
        # If keyword retriever available, merge results
        if self.keyword_retriever:
            keyword_results = self.keyword_retriever.retrieve(query, k=k)
            # Merge and deduplicate (simplified)
            all_results = semantic_results + keyword_results
            # In production, you'd want smarter merging
            return all_results[:k]
        
        return semantic_results


# ============================================================================
# Document Processing
# ============================================================================

class DocumentProcessor:
    """
    Process documents for ingestion.
    Replaces the complex ingestion logic.
    """
    
    def __init__(self, 
                 embedding_provider: EmbeddingProvider,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process text into chunks with embeddings."""
        # Split into chunks
        chunks = self._split_text(text)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                **metadata,
                "chunk_id": i,
                "chunk_size": len(chunk)
            }
            
            # Embed chunk
            embedding = self.embedding_provider.embed_text(chunk)
            
            # Ensure embedding is numpy array
            if isinstance(embedding, list):
                import numpy as np
                embedding = np.array(embedding)
            
            doc = Document(
                content=chunk,
                metadata=doc_metadata,
                embedding=embedding
            )
            documents.append(doc)
        
        logger.info(f"Processed text into {len(documents)} chunks")
        return documents
    
    def _split_text(self, text: str) -> List[str]:
        """Simple text splitting."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


# ============================================================================
# Factory Functions
# ============================================================================

def create_simple_retriever(
    use_faiss: bool = False,
    index_path: Optional[str] = None,
    model_name: Optional[str] = None
) -> SimpleRetriever:
    """Create a simple retriever with appropriate components."""
    
    # Create embedding provider
    embedding_provider = LocalEmbeddingProvider(model_name)
    
    # Create vector store
    if use_faiss:
        try:
            from pathlib import Path
            vector_store = FAISSVectorStore(index_path)
        except ImportError:
            logger.warning("FAISS not available, using in-memory store")
            vector_store = InMemoryVectorStore()
    else:
        vector_store = InMemoryVectorStore()
    
    # Create retriever
    retriever = SimpleRetriever(embedding_provider, vector_store)
    
    logger.info(f"Created retriever with {type(vector_store).__name__}")
    return retriever


def create_mock_retriever() -> SimpleRetriever:
    """Create a mock retriever for testing."""
    embedding_provider = MockEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    
    # Add some mock documents
    mock_docs = [
        Document(
            content="Panos's favorite food is souvlaki.",
            metadata={"source": "test.txt"},
            embedding=embedding_provider.embed_text("favorite food souvlaki")
        ),
        Document(
            content="The system runs on Linux.",
            metadata={"source": "system.txt"},
            embedding=embedding_provider.embed_text("system linux operating")
        )
    ]
    vector_store.add_documents(mock_docs)
    
    return SimpleRetriever(embedding_provider, vector_store)
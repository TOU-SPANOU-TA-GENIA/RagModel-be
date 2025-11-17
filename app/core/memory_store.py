# app/core/memory_store.py
"""
In-Memory Storage System
Keeps everything in RAM/GPU memory for maximum performance.
No disk I/O during operation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pickle
import io
import numpy as np
from threading import Lock

from app.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Global In-Memory Stores (Singleton Pattern)
# ============================================================================

class InMemoryDatabase:
    """
    Central in-memory database for all application data.
    Everything stays in RAM for fast access.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize all in-memory stores."""
        # Document store
        self.documents: Dict[str, Any] = {}
        self.document_embeddings: Dict[str, np.ndarray] = {}
        
        # Vector index (numpy arrays for fast similarity)
        self.vector_index: Optional[np.ndarray] = None
        self.vector_metadata: List[Dict[str, Any]] = []
        
        # Chat store
        self.chats: Dict[str, Dict] = {}
        
        # Model cache
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Any] = {}
        
        # Embedding cache (text -> embedding)
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.max_cache_size = 10000
        
        # Tool results cache
        self.tool_cache: Dict[str, Any] = {}
        
        logger.info("In-memory database initialized")
    
    # ========================================================================
    # Document Management
    # ========================================================================
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any], 
                     embedding: Optional[np.ndarray] = None):
        """Add document to memory."""
        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata
        }
        if embedding is not None:
            self.document_embeddings[doc_id] = embedding
        logger.debug(f"Added document {doc_id} to memory")
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document from memory."""
        doc = self.documents.get(doc_id)
        if doc and doc_id in self.document_embeddings:
            doc["embedding"] = self.document_embeddings[doc_id]
        return doc
    
    def list_documents(self) -> List[str]:
        """List all document IDs."""
        return list(self.documents.keys())
    
    # ========================================================================
    # Vector Index Management (No FAISS, pure NumPy)
    # ========================================================================
    
    def build_vector_index(self):
        """Build vector index from document embeddings."""
        if not self.document_embeddings:
            logger.warning("No embeddings to build index from")
            return
        
        # Stack all embeddings into a single matrix
        doc_ids = list(self.document_embeddings.keys())
        embeddings = [self.document_embeddings[doc_id] for doc_id in doc_ids]
        
        self.vector_index = np.vstack(embeddings)
        self.vector_metadata = [
            {"doc_id": doc_id, **self.documents[doc_id]["metadata"]} 
            for doc_id in doc_ids
        ]
        
        logger.info(f"Built vector index with {len(doc_ids)} vectors")
    
    def search_vectors(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """
        Fast vector similarity search using NumPy.
        No disk I/O, pure memory operations.
        """
        if self.vector_index is None or len(self.vector_index) == 0:
            return []
        
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Normalize index (do this once during build for efficiency)
        index_norms = np.linalg.norm(self.vector_index, axis=1, keepdims=True)
        index_normalized = self.vector_index / (index_norms + 1e-10)
        
        # Compute cosine similarities (dot product of normalized vectors)
        similarities = np.dot(index_normalized, query_norm)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Build results
        results = []
        for idx in top_k_indices:
            doc_id = self.vector_metadata[idx]["doc_id"]
            results.append({
                "doc_id": doc_id,
                "content": self.documents[doc_id]["content"],
                "metadata": self.vector_metadata[idx],
                "score": float(similarities[idx])
            })
        
        return results
    
    # ========================================================================
    # Embedding Cache
    # ========================================================================
    
    def cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache an embedding for reuse."""
        # Simple FIFO eviction if cache is full
        if len(self.embedding_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest]
        
        self.embedding_cache[text] = embedding
    
    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding if exists."""
        return self.embedding_cache.get(text)
    
    # ========================================================================
    # Model Cache
    # ========================================================================
    
    def cache_model(self, model_name: str, model: Any, config: Any = None):
        """Keep model in memory."""
        self.models[model_name] = model
        if config:
            self.model_configs[model_name] = config
        logger.info(f"Cached model {model_name} in memory")
    
    def get_cached_model(self, model_name: str) -> Optional[Any]:
        """Get model from memory cache."""
        return self.models.get(model_name)
    
    # ========================================================================
    # Chat Management (replaces file-based chat storage)
    # ========================================================================
    
    def save_chat(self, chat_id: str, chat_data: Dict):
        """Save chat in memory."""
        self.chats[chat_id] = chat_data
    
    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get chat from memory."""
        return self.chats.get(chat_id)
    
    def list_chats(self) -> List[str]:
        """List all chat IDs."""
        return list(self.chats.keys())
    
    # ========================================================================
    # Tool Cache
    # ========================================================================
    
    def cache_tool_result(self, key: str, result: Any):
        """Cache tool execution result."""
        self.tool_cache[key] = result
    
    def get_cached_tool_result(self, key: str) -> Optional[Any]:
        """Get cached tool result."""
        return self.tool_cache.get(key)
    
    # ========================================================================
    # Memory Management
    # ========================================================================
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        def get_size(obj):
            """Get approximate size of object in bytes."""
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, dict):
                return sum(get_size(v) for v in obj.values())
            elif isinstance(obj, (list, tuple)):
                return sum(get_size(item) for item in obj)
            else:
                # Rough estimate for other objects
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        
        return {
            "documents": get_size(self.documents),
            "embeddings": get_size(self.document_embeddings),
            "vector_index": get_size(self.vector_index) if self.vector_index is not None else 0,
            "embedding_cache": get_size(self.embedding_cache),
            "model_cache": sum(get_size(m) for m in self.models.values()),
            "chats": get_size(self.chats),
            "tool_cache": get_size(self.tool_cache)
        }
    
    def clear_caches(self, keep_models: bool = True):
        """Clear caches to free memory."""
        self.embedding_cache.clear()
        self.tool_cache.clear()
        
        if not keep_models:
            self.models.clear()
            self.model_configs.clear()
        
        logger.info("Cleared memory caches")
    
    def reset(self):
        """Complete reset (for testing)."""
        self._initialize()
        logger.info("In-memory database reset")


# Global instance
memory_db = InMemoryDatabase()


# ============================================================================
# Fast In-Memory Vector Store (Replaces FAISS)
# ============================================================================

class FastInMemoryVectorStore:
    """
    Ultra-fast in-memory vector store using NumPy.
    No disk I/O, all operations in RAM.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.db = memory_db  # Use global memory database
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents with embeddings to memory store."""
        for i, doc in enumerate(documents):
            doc_id = f"doc_{len(self.db.documents)}_{i}"
            
            self.db.add_document(
                doc_id=doc_id,
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                embedding=doc.get("embedding")
            )
        
        # Rebuild index
        self.db.build_vector_index()
        logger.info(f"Added {len(documents)} documents to in-memory store")
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents in memory."""
        return self.db.search_vectors(query_embedding, k)
    
    def clear(self):
        """Clear the vector store."""
        self.db.documents.clear()
        self.db.document_embeddings.clear()
        self.db.vector_index = None
        self.db.vector_metadata = []


# ============================================================================
# Cached Embedding Provider
# ============================================================================

class CachedEmbeddingProvider:
    """
    Embedding provider with in-memory caching.
    Avoids recomputing embeddings for seen texts.
    """
    
    def __init__(self, base_provider):
        self.base_provider = base_provider
        self.db = memory_db
    
    def embed_text(self, text: str) -> np.ndarray:
        """Get embedding with caching."""
        # Check cache first
        cached = self.db.get_cached_embedding(text)
        if cached is not None:
            logger.debug("Embedding cache hit")
            return cached
        
        # Compute and cache
        embedding = self.base_provider.embed_text(text)
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        self.db.cache_embedding(text, embedding)
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed with caching."""
        results = []
        texts_to_compute = []
        indices_to_compute = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self.db.get_cached_embedding(text)
            if cached is not None:
                results.append((i, cached))
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(i)
        
        # Compute missing embeddings
        if texts_to_compute:
            new_embeddings = self.base_provider.embed_batch(texts_to_compute)
            
            for idx, text, embedding in zip(indices_to_compute, texts_to_compute, new_embeddings):
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                self.db.cache_embedding(text, embedding)
                results.append((idx, embedding))
        
        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]


# ============================================================================
# Model Manager (Keep models in memory)
# ============================================================================

class InMemoryModelManager:
    """
    Manages models in memory to avoid repeated loading.
    """
    
    def __init__(self):
        self.db = memory_db
    
    def get_or_load_model(self, model_name: str, loader_func):
        """Get model from cache or load it."""
        model = self.db.get_cached_model(model_name)
        
        if model is None:
            logger.info(f"Loading model {model_name} into memory")
            model = loader_func()
            self.db.cache_model(model_name, model)
        else:
            logger.info(f"Using cached model {model_name}")
        
        return model
    
    def preload_models(self, model_list: List[tuple]):
        """Preload multiple models into memory."""
        for model_name, loader_func in model_list:
            self.get_or_load_model(model_name, loader_func)
    
    def clear_model(self, model_name: str):
        """Remove model from memory."""
        if model_name in self.db.models:
            del self.db.models[model_name]
            logger.info(f"Cleared model {model_name} from memory")


# Global model manager
model_manager = InMemoryModelManager()
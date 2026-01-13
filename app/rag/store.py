import os
import faiss
import pickle
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from app.config import get_config
from app.rag.schemas import RagChunk
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorStore:
    """
    FAISS-based vector storage.
    Persists index and metadata to disk specified in config.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.config = get_config()
        
        # Paths from config
        self.index_dir = Path(self.config.get("paths", {}).get("index_dir", "faiss_index"))
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "index.pkl"
        
        self.index = None
        self.chunks: Dict[int, RagChunk] = {} # Map ID -> Chunk Data (Fixed from tuple to dict)
        
        self._load_or_create()

    def _load_or_create(self):
        """Load existing index or create new one."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "rb") as f:
                    self.chunks = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Error loading index: {e}. Creating new one.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a generic IndexFlatIP (Inner Product) for cosine similarity."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = {} # Ensure dictionary initialization
        logger.info("Created new empty FAISS index.")

    def add(self, chunks: List[RagChunk], vectors: np.ndarray):
        """Add vectors and metadata to the store."""
        if len(chunks) != len(vectors):
            raise ValueError("Count mismatch between chunks and vectors")
            
        start_id = self.index.ntotal
        # Ensure vectors are float32 for FAISS
        self.index.add(vectors.astype('float32'))
        
        for i, chunk in enumerate(chunks):
            # Store metadata mapped to the FAISS internal ID
            self.chunks[start_id + i] = chunk
            
        self._save()
        logger.info(f"Added {len(chunks)} vectors to store.")

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[RagChunk, float]]:
        """Search the index."""
        if self.index is None or self.index.ntotal == 0:
            return []
            
        # Ensure query_vector is 2D
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        D, I = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for j, idx in enumerate(I[0]):
            if idx != -1 and idx in self.chunks:
                chunk = self.chunks[idx]
                score = float(D[0][j])
                results.append((chunk, score))
                
        return results

    def _save(self):
        """Persist to disk."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.chunks, f)
            logger.debug("Vector store saved to disk.")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def clear(self):
        """Wipe the index."""
        self._create_new_index()
        self._save()

# --- Singleton and Helper Functions ---

# Initialize global instance
vector_store = VectorStore(dimension=384)

def search_vector_store(query: str, k: int = 3) -> List[str]:
    """
    Bridge function for the RagService.
    Converts text to embedding and returns content strings.
    """
    from app.rag.embeddings import get_embeddings
    
    try:
        model = get_embeddings()
        # model.encode returns the vector for the text
        query_vector = model.encode([query])
        
        # Search the global vector_store
        matches = vector_store.search(query_vector, k=k)
        
        # Return only the content strings from the RagChunk objects
        return [match[0].content for match in matches]
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        return []
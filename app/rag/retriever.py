from typing import List, Dict, Any
import numpy as np

from app.core.interfaces import Retriever as CoreRetriever
from app.rag.embeddings import get_embedding_service
from app.rag.store import VectorStore
from app.rag.schemas import SearchResult
from app.config import get_config

class RagRetriever(CoreRetriever):
    """
    Standard RAG Retriever.
    """
    
    def __init__(self):
        self.embedder = get_embedding_service()
        self.store = VectorStore(self.embedder.dimension)
        self.config = get_config().get("rag", {})
        self.min_score = self.config.get("min_relevance_score", 0.3)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        query_vec = self.embedder.encode([query])
        
        raw_results = self.store.search(query_vec, k=k)
        
        results = []
        for chunk, score in raw_results:
            if score < self.min_score:
                continue
                
            res = SearchResult(
                content=chunk.content,
                source=chunk.source,
                score=score,
                metadata=chunk.metadata
            )
            results.append(res.to_dict())
            
        return results

# --- CRITICAL FIX: Instantiate the Singleton ---
rag_retriever = RagRetriever()
# app/rag/retrievers.py
"""
RAG retrievers - re-exports from split modules.
"""

import time
from typing import List, Dict, Any, Optional

from app.core.interfaces import Retriever
from app.rag.embedding_providers import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    MockEmbeddingProvider
)
from app.rag.vector_stores import (
    Document,
    RetrievalResult,
    VectorStore,
    InMemoryVectorStore,
    FAISSVectorStore
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

__all__ = [
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "MockEmbeddingProvider",
    "Document",
    "RetrievalResult",
    "VectorStore",
    "InMemoryVectorStore",
    "FAISSVectorStore",
    "SimpleRetriever",
    "HybridRetriever",
    "DocumentProcessor",
    "create_simple_retriever",
    "create_mock_retriever"
]


class SimpleRetriever(Retriever):
    """Retriever combining embedding and vector store."""
    
    def __init__(self, 
                 embedding_provider: EmbeddingProvider,
                 vector_store: VectorStore,
                 min_score: float = 0.3):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.min_score = min_score
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        start_time = time.time()
        
        try:
            query_embedding = self.embedding_provider.embed_text(query)
            
            if hasattr(self.vector_store, 'search'):
                results = self.vector_store.search(query_embedding, k=k)
            else:
                results = []
            
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
    """Retriever combining semantic and keyword search."""
    
    def __init__(self, 
                 semantic_retriever: SimpleRetriever,
                 keyword_retriever: Optional[Retriever] = None):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        semantic_results = self.semantic_retriever.retrieve(query, k=k)
        
        if self.keyword_retriever:
            keyword_results = self.keyword_retriever.retrieve(query, k=k)
            all_results = semantic_results + keyword_results
            return all_results[:k]
        
        return semantic_results


class DocumentProcessor:
    """Processes documents for ingestion."""
    
    def __init__(self, 
                 embedding_provider: EmbeddingProvider,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        chunks = self._split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                **metadata,
                "chunk_id": i,
                "chunk_size": len(chunk)
            }
            
            embedding = self.embedding_provider.embed_text(chunk)
            
            if isinstance(embedding, list):
                import numpy as np
                embedding = np.array(embedding)
            
            doc = Document(content=chunk, metadata=doc_metadata, embedding=embedding)
            documents.append(doc)
        
        logger.info(f"Processed text into {len(documents)} chunks")
        return documents
    
    def _split_text(self, text: str) -> List[str]:
        chunks = []
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


def create_simple_retriever(
    use_faiss: bool = False,
    index_path: Optional[str] = None,
    model_name: Optional[str] = None
) -> SimpleRetriever:
    """Create a simple retriever."""
    embedding_provider = LocalEmbeddingProvider(model_name)
    
    if use_faiss:
        try:
            vector_store = FAISSVectorStore(index_path)
        except ImportError:
            logger.warning("FAISS not available, using in-memory store")
            vector_store = InMemoryVectorStore()
    else:
        vector_store = InMemoryVectorStore()
    
    logger.info(f"Created retriever with {type(vector_store).__name__}")
    return SimpleRetriever(embedding_provider, vector_store)


def create_mock_retriever() -> SimpleRetriever:
    """Create a mock retriever for testing."""
    embedding_provider = MockEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    
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
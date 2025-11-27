# app/rag/__init__.py
from .embedding_providers import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    MockEmbeddingProvider
)
from .vector_stores import (
    Document,
    RetrievalResult,
    VectorStore,
    InMemoryVectorStore,
    FAISSVectorStore
)
from .retrievers import (
    SimpleRetriever,
    HybridRetriever,
    DocumentProcessor,
    create_simple_retriever,
    create_mock_retriever
)
from .ingestion import (
    save_upload,
    ingest_single_file,
    ingest_directory
)

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
    "create_mock_retriever",
    "save_upload",
    "ingest_single_file",
    "ingest_directory"
]
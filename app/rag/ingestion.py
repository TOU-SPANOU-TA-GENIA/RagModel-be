import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.rag.schemas import RagDocument, RagChunk
from app.rag.embeddings import get_embedding_service
from app.rag.store import VectorStore
from app.config import get_config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class IngestionService:
    """
    Handles processing of documents into the vector store.
    """
    
    def __init__(self):
        # Eager loading: This triggers embedding model loading immediately upon instantiation
        self.embedder = get_embedding_service()
        self.store = VectorStore(self.embedder.dimension)
        
        rag_cfg = get_config().get("rag", {})
        self.chunk_size = rag_cfg.get("chunk_size", 500)
        self.chunk_overlap = rag_cfg.get("chunk_overlap", 50)
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def ingest_document(self, doc: RagDocument):
        logger.info(f"Ingesting document: {doc.source}")
        
        texts = self.splitter.split_text(doc.content)
        if not texts:
            return

        chunks = []
        for text in texts:
            chunk_id = str(uuid.uuid4())
            chunks.append(RagChunk(
                chunk_id=chunk_id,
                content=text,
                source=doc.source,
                metadata=doc.metadata
            ))

        if not chunks:
            return

        vectors = self.embedder.encode([c.content for c in chunks])
        self.store.add(chunks, vectors)
        logger.info(f"Successfully indexed {len(chunks)} chunks from {doc.source}")

    def clear_index(self):
        self.store.clear()

# --- CRITICAL FIX: Instantiate the Singleton ---
ingestion_service = IngestionService()
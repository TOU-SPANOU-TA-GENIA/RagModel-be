# app/rag/ingestion.py
"""
Simplified document ingestion module.
Moved from app/ingestion.py and refactored for new architecture.
"""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile

from ..config import RAG_CONFIG
from ..core.exceptions import IngestionException
from ..utils.logger import setup_logger
from .retrievers import DocumentProcessor, LocalEmbeddingProvider, Document

logger = setup_logger(__name__)


async def save_upload(file: UploadFile) -> str:
    """Save an uploaded file to the knowledge directory."""
    try:
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        
        filename = file.filename.replace("/", "_").replace("\\", "_")
        filepath = KNOWLEDGE_DIR / filename
        
        content = await file.read()
        
        with open(filepath, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved uploaded file: {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        raise IngestionException(f"Could not save file: {e}")


def load_text_file(filepath: Path) -> str:
    """Load text from a file with multiple encoding attempts."""
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            logger.debug(f"Successfully read {filepath} with {encoding} encoding")
            return content
        except UnicodeDecodeError:
            continue
    
    raise IngestionException(f"Could not decode file {filepath} with any encoding")


def load_documents_from_directory(directory: Path) -> List[Document]:
    """Load all documents from a directory."""
    documents = []
    supported_extensions = {'.txt', '.md', '.text'}
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return documents
    
    for filepath in directory.rglob('*'):
        if filepath.is_file() and filepath.suffix.lower() in supported_extensions:
            try:
                content = load_text_file(filepath)
                
                doc = Document(
                    content=content,
                    metadata={
                        "source": filepath.name,
                        "filepath": str(filepath),
                        "file_type": filepath.suffix,
                    }
                )
                documents.append(doc)
                logger.info(f"Loaded document: {filepath.name}")
            
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents from {directory}")
    return documents


def ingest_directory(directory: Path = None, rebuild: bool = False) -> dict:
    """Ingest all documents from a directory into in-memory vector store."""
    try:
        logger.info(f"Starting ingestion from {directory} (rebuild={rebuild})")
        
        # Load documents
        raw_documents = load_documents_from_directory(directory)
        
        if not raw_documents:
            logger.warning("No documents found to ingest")
            return {
                "success": False,
                "message": "No documents found",
                "documents_loaded": 0,
                "chunks_created": 0
            }
        
        # Use in-memory components
        from app.core.memory_store import FastInMemoryVectorStore, CachedEmbeddingProvider
        
        # Process documents
        base_embedding_provider = LocalEmbeddingProvider()
        embedding_provider = CachedEmbeddingProvider(base_embedding_provider)
        
        processor = DocumentProcessor(
            embedding_provider=embedding_provider,
            chunk_size=RAG_CONFIG["chunk_size"],
            chunk_overlap=RAG_CONFIG["chunk_overlap"]
        )
        
        # Process each document into chunks with embeddings
        all_chunks = []
        for doc in raw_documents:
            chunks = processor.process_text(doc.content, doc.metadata)
            all_chunks.extend(chunks)
        
        # Use in-memory vector store
        vector_store = FastInMemoryVectorStore(dimension=384)
        
        if rebuild:
            vector_store.clear()
        
        # Convert Document objects to dicts for in-memory store
        chunk_dicts = [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding
            }
            for chunk in all_chunks
        ]
        
        vector_store.add_documents(chunk_dicts)
        
        logger.info("Ingestion completed successfully (all in memory)")
        return {
            "success": True,
            "message": "Ingestion completed successfully",
            "documents_loaded": len(raw_documents),
            "chunks_created": len(all_chunks)
        }
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return {
            "success": False,
            "message": str(e),
            "documents_loaded": 0,
            "chunks_created": 0
        }


async def ingest_single_file(filepath: str) -> dict:
    """Ingest a single file into the vector store."""
    try:
        path = Path(filepath)
        
        if not path.exists():
            raise IngestionException(f"File not found: {filepath}")
        
        # Load document
        content = load_text_file(path)
        doc = Document(
            content=content,
            metadata={
                "source": path.name,
                "filepath": str(path),
                "file_type": path.suffix,
            }
        )
        
        # Process into chunks
        embedding_provider = LocalEmbeddingProvider()
        processor = DocumentProcessor(
            embedding_provider=embedding_provider,
            chunk_size=RAG_CONFIG["chunk_size"],
            chunk_overlap=RAG_CONFIG["chunk_overlap"]
        )
        
        chunks = processor.process_text(doc.content, doc.metadata)
        
        # Add to vector store
        from .retrievers import FAISSVectorStore
        from ..config import INDEX_DIR
        
        vector_store = FAISSVectorStore(
            index_path=str(INDEX_DIR / "index.faiss")
        )
        vector_store.add_documents(chunks)
        
        return {
            "success": True,
            "message": f"File {path.name} ingested successfully",
            "chunks_created": len(chunks)
        }
    
    except Exception as e:
        logger.error(f"Failed to ingest file {filepath}: {e}")
        return {
            "success": False,
            "message": str(e),
            "chunks_created": 0
        }
        
        
# ADD THIS FUNCTION to app/rag/ingestion.py
# Add it right after the ingest_single_file function (around line 150)

def ingest_file(filepath: Path) -> bool:
    """
    Simple synchronous wrapper for ingesting a single file.
    Used by network filesystem integration.
    
    Args:
        filepath: Path to file to ingest
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert to string if Path object
        filepath_str = str(filepath) if isinstance(filepath, Path) else filepath
        
        # Load document
        path = Path(filepath_str)
        if not path.exists():
            logger.warning(f"File not found: {filepath_str}")
            return False
        
        content = load_text_file(path)
        doc = Document(
            content=content,
            metadata={
                "source": path.name,
                "filepath": str(path),
                "file_type": path.suffix,
            }
        )
        
        # Process into chunks with embeddings
        from app.core.memory_store import CachedEmbeddingProvider
        
        base_embedding_provider = LocalEmbeddingProvider()
        embedding_provider = CachedEmbeddingProvider(base_embedding_provider)
        
        processor = DocumentProcessor(
            embedding_provider=embedding_provider,
            chunk_size=RAG_CONFIG["chunk_size"],
            chunk_overlap=RAG_CONFIG["chunk_overlap"]
        )
        
        chunks = processor.process_text(doc.content, doc.metadata)
        
        # Add to in-memory vector store
        from app.core.memory_store import FastInMemoryVectorStore
        
        # Get or create global vector store (dimension=384 for all-MiniLM-L6-v2)
        vector_store = FastInMemoryVectorStore(dimension=384)
        
        # Convert chunks to dicts
        chunk_dicts = [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding
            }
            for chunk in chunks
        ]
        
        vector_store.add_documents(chunk_dicts)
        
        logger.info(f"✅ Ingested file: {path.name} ({len(chunks)} chunks)")
        return True
    
    except Exception as e:
        logger.error(f"Failed to ingest {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return False
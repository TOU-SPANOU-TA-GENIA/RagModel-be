from typing import List, Tuple, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import torch

from .config import EMBEDDING_MODEL_NAME, INDEX_DIR, RAG_CONFIG
from .exceptions import VectorStoreNotInitializedException
from .logger import setup_logger

logger = setup_logger(__name__)

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[FAISS] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    
    if _embeddings is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device: {device}")
        
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}  # Better similarity scores
        )
        logger.info("Embedding model loaded successfully")
    
    return _embeddings

def init_vectorstore() -> FAISS:
    global _vectorstore
    
    if _vectorstore is not None:
        return _vectorstore
    
    embeddings = get_embeddings()
    
    # Check if index exists
    if not INDEX_DIR.exists():
        logger.error(f"FAISS index not found at {INDEX_DIR}")
        raise VectorStoreNotInitializedException()
    
    try:
        logger.info(f"Loading FAISS index from {INDEX_DIR}")
        _vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully")
        return _vectorstore
    
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise VectorStoreNotInitializedException()

def retrieve(query: str, k: int = None) -> List[Tuple[str, dict, float]]:
    if k is None:
        k = RAG_CONFIG["top_k"]
    
    try:
        vectorstore = init_vectorstore()
        
        # Use similarity_search_with_score for relevance filtering
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        # Filter by minimum relevance score
        min_score = RAG_CONFIG["min_relevance_score"]
        filtered_results = []
        
        for doc, score in results:
            # FAISS returns distance (lower is better), convert to similarity
            similarity = 1 - score if score < 1 else 0
            
            if similarity >= min_score:
                filtered_results.append((
                    doc.page_content,
                    doc.metadata,
                    similarity
                ))
        
        logger.info(f"Retrieved {len(filtered_results)} relevant documents")
        return filtered_results
    
    except VectorStoreNotInitializedException:
        logger.warning("Vector store not initialized")
        return []
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []

def add_documents(documents: List[Document]) -> bool:
    global _vectorstore
    
    try:
        embeddings = get_embeddings()
        
        if _vectorstore is None:
            logger.info("Creating new FAISS index")
            _vectorstore = FAISS.from_documents(documents, embeddings)
        else:
            logger.info(f"Adding {len(documents)} documents to existing index")
            _vectorstore.add_documents(documents)
        
        # Save to disk
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _vectorstore.save_local(str(INDEX_DIR))
        logger.info(f"Vector store saved to {INDEX_DIR}")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        return False


def reset_vectorstore():
    global _vectorstore
    _vectorstore = None
    logger.info("Vector store reset")
from typing import List
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import logging

from app.config import get_config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingService:
    """
    Generates vector embeddings for text.
    Configurable via 'rag.embeddings' in config.json.
    """
    
    _instance = None
    
    def __init__(self):
        self.config = get_config()
        self.rag_config = self.config.get("rag", {})
        self.model_name = self.rag_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = self._get_device()
        self.model = None
        self._load_model()

    def _get_device(self) -> str:
        cfg_device = self.rag_config.get("device", "auto")
        if cfg_device != "auto":
            return cfg_device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Lazy load the model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError("Embedding model initialization failed")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of strings."""
        if not self.model:
            self._load_model()
        
        # Normalize/clean text
        clean_texts = [t.replace("\n", " ") for t in texts]
        return self.model.encode(clean_texts, convert_to_numpy=True, normalize_embeddings=True)

    @property
    def dimension(self) -> int:
        """Return the vector dimension (e.g., 384 for MiniLM)."""
        if not self.model:
            self._load_model()
        return self.model.get_sentence_embedding_dimension()

# --- STANDALONE WRAPPERS FOR EXTERNAL IMPORTS ---

def get_embedding_service():
    """Returns the singleton instance of the EmbeddingService class."""
    if EmbeddingService._instance is None:
        EmbeddingService._instance = EmbeddingService()
    return EmbeddingService._instance

def get_embeddings():
    """
    Bridge function to satisfy existing imports in your app.
    Returns the actual SentenceTransformer model.
    """
    service = get_embedding_service()
    return service.model
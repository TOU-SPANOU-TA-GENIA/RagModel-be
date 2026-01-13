from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from .base import ConfigField, ConfigCategory

@dataclass
class RAGSettings:
    """Retrieval Augmented Generation settings."""
    top_k: int = 3
    min_relevance_score: float = 0.2
    chunk_size: int = 500
    chunk_overlap: int = 50
    rerank_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class NetworkFilesystemSettings:
    """Settings for monitoring and indexing network folders."""
    enabled: bool = False
    auto_start_monitoring: bool = True
    shares: List[Dict[str, Any]] = field(default_factory=list)
    # Example share dict: {"name": "Logistics", "path": "/mnt/share", "auto_index": True}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DocumentSettings:
    """Settings for generated documents (reports)."""
    default_font: str = "Arial"
    default_font_size: int = 11
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
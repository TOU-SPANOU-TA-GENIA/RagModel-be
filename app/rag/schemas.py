from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class RagDocument:
    """A raw document to be indexed."""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None

@dataclass
class RagChunk:
    """A piece of text with its vector."""
    chunk_id: str
    content: str
    source: str
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None

@dataclass
class SearchResult:
    """Standardized return format for retrieval."""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "content": self.content,
            "metadata": {
                "source": self.source,
                "score": self.score,
                **self.metadata
            }
        }
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class AnalysisType(str, Enum):
    INTELLIGENCE = "intelligence"
    LOGISTICS = "logistics"
    GENERAL = "general"

@dataclass
class ExtractedContent:
    """Input: Raw text content from a file."""
    source_name: str
    text_content: str
    content_type: str
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    source_path: str = ""

@dataclass
class DetectedPattern:
    """Output: A specific item found in text."""
    category: str       # e.g., 'date', 'weapon', 'location'
    value: str          # e.g., '2024-01-01', 'M16'
    confidence: float
    source: str
    context: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisFinding:
    """Output: A high-level insight derived from patterns."""
    title: str
    description: str
    severity: str = "info" # critical, high, medium, low, info
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0

@dataclass
class AnalysisResult:
    """Final Output: The complete analysis report."""
    timestamp: str
    doc_count: int
    patterns: List[DetectedPattern]
    findings: List[AnalysisFinding]
    graph: Dict[str, List[str]] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
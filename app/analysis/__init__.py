# app/analysis/__init__.py
"""
Intelligence analysis module.

Provides document extraction, pattern analysis, and briefing generation.
"""

from app.analysis.content_extractors import (
    ExtractedContent,
    ContentExtractor,
    ContentExtractorRegistry,
    get_extractor_registry,
    extract_content,
    extract_batch
)
from app.analysis.intelligence_analyzer import (
    IntelligencePattern,
    CrossReference,
    AnalysisResult,
    IntelligenceAnalyzer,
    PatternDetector,
    analyze_documents
)
from app.analysis.briefing_generator import (
    BriefingConfig,
    ClassificationLevel,
    ReportSection,
    BriefingGenerator,
    generate_briefing
)

__all__ = [
    # Content Extraction
    "ExtractedContent",
    "ContentExtractor",
    "ContentExtractorRegistry",
    "get_extractor_registry",
    "extract_content",
    "extract_batch",
    
    # Intelligence Analysis
    "IntelligencePattern",
    "CrossReference", 
    "AnalysisResult",
    "IntelligenceAnalyzer",
    "PatternDetector",
    "analyze_documents",
    
    # Briefing Generation
    "BriefingConfig",
    "ClassificationLevel",
    "ReportSection",
    "BriefingGenerator",
    "generate_briefing"
]
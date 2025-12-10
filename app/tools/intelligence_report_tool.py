# app/tools/intelligence_report_tool.py
"""
Intelligence Report Generation Tool.

Processes multiple documents to generate comprehensive briefing reports.
Integrates with the existing tool system.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

from app.tools.models import BaseTool, ToolResult
from app.utils.logger import setup_logger
from app.config import BASE_DIR

logger = setup_logger(__name__)


@dataclass
class ReportRequest:
    """Request parameters for report generation."""
    file_paths: List[str]
    title: str = "Αναφορά Πληροφοριών"
    classification: str = "ΑΔΙΑΒΑΘΜΗΤΟ"
    output_format: str = "docx"
    include_timeline: bool = True
    include_entity_graph: bool = True


class IntelligenceReportTool(BaseTool):
    """
    Tool for generating intelligence briefing reports from multiple documents.
    
    Accepts multiple files, extracts content, analyzes patterns,
    and generates a structured briefing document in Greek.
    """
    
    def __init__(self, output_dir: Path = None, llm_provider=None):
        super().__init__()
        self.output_dir = output_dir or (BASE_DIR / "outputs" / "briefings")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_provider = llm_provider
    
    @property
    def name(self) -> str:
        return "generate_intelligence_report"
    
    @property
    def description(self) -> str:
        return """Generate comprehensive intelligence briefing from multiple documents.
        
        Accepts: file_paths (list), title, classification, output_format
        Supports: PDF, DOCX, XLSX, TXT, MD, images
        Returns: Path to generated briefing document with citations"""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to analyze"
                },
                "title": {
                    "type": "string",
                    "description": "Report title",
                    "default": "Αναφορά Πληροφοριών"
                },
                "classification": {
                    "type": "string",
                    "enum": ["ΑΔΙΑΒΑΘΜΗΤΟ", "ΠΕΡΙΟΡΙΣΜΕΝΗΣ ΧΡΗΣΗΣ", "ΕΜΠΙΣΤΕΥΤΙΚΟ", "ΑΠΟΡΡΗΤΟ"],
                    "default": "ΑΔΙΑΒΑΘΜΗΤΟ"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["docx", "pdf", "md"],
                    "default": "docx"
                }
            },
            "required": ["file_paths"]
        }
    
    def _execute_impl(
        self,
        file_paths: List[str],
        title: str = "Αναφορά Πληροφοριών",
        classification: str = "ΑΔΙΑΒΑΘΜΗΤΟ",
        output_format: str = "docx",
        include_timeline: bool = True,
        include_entity_graph: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute the intelligence report generation."""
        
        # Validate inputs
        if not file_paths:
            return ToolResult(
                success=False,
                error="Δεν παρέχθηκαν αρχεία για ανάλυση",
                data=None
            )
        
        # Convert to Path objects and validate
        paths = []
        missing = []
        for fp in file_paths:
            path = Path(fp)
            if path.exists():
                paths.append(path)
            else:
                missing.append(fp)
        
        if missing:
            logger.warning(f"Files not found: {missing}")
        
        if not paths:
            return ToolResult(
                success=False,
                error=f"Κανένα αρχείο δεν βρέθηκε. Έλεγξε τις διαδρομές: {missing}",
                data=None
            )
        
        try:
            # Import analysis components
            from app.analysis import (
                extract_batch,
                analyze_documents,
                generate_briefing,
                BriefingConfig,
                ClassificationLevel
            )
            
            # Step 1: Extract content from all documents
            logger.info(f"Extracting content from {len(paths)} documents...")
            extracted_contents = extract_batch(paths)
            
            successful_extractions = [e for e in extracted_contents if e.content_type != "error"]
            logger.info(f"Successfully extracted {len(successful_extractions)}/{len(paths)} documents")
            
            if not successful_extractions:
                return ToolResult(
                    success=False,
                    error="Αποτυχία εξαγωγής περιεχομένου από όλα τα αρχεία",
                    data={
                        "attempted_files": file_paths,
                        "extraction_errors": [
                            {"file": e.source_name, "error": e.metadata.get("error")}
                            for e in extracted_contents if e.content_type == "error"
                        ]
                    }
                )
            
            # Step 2: Analyze documents for intelligence patterns
            logger.info("Analyzing documents for intelligence patterns...")
            analysis_result = analyze_documents(successful_extractions, self.llm_provider)
            
            # Step 3: Map classification string to enum
            classification_map = {
                "ΑΔΙΑΒΑΘΜΗΤΟ": ClassificationLevel.UNCLASSIFIED,
                "ΠΕΡΙΟΡΙΣΜΕΝΗΣ ΧΡΗΣΗΣ": ClassificationLevel.RESTRICTED,
                "ΕΜΠΙΣΤΕΥΤΙΚΟ": ClassificationLevel.CONFIDENTIAL,
                "ΑΠΟΡΡΗΤΟ": ClassificationLevel.SECRET,
                "ΑΚΡΩΣ ΑΠΟΡΡΗΤΟ": ClassificationLevel.TOP_SECRET
            }
            class_level = classification_map.get(classification, ClassificationLevel.UNCLASSIFIED)
            
            # Step 4: Configure and generate briefing
            config = BriefingConfig(
                classification=class_level,
                include_sources=True,
                include_timeline=include_timeline,
                include_entity_graph=include_entity_graph,
                include_confidence_scores=True,
                output_format=output_format
            )
            
            logger.info(f"Generating {output_format.upper()} briefing...")
            report_path = generate_briefing(
                analysis=analysis_result,
                documents=successful_extractions,
                title=title,
                config=config
            )
            
            # Build result data
            result_data = {
                "report_path": str(report_path),
                "report_name": report_path.name,
                "report_format": output_format,
                "download_ready": True,
                "download_url": f"/download/{report_path.name}",
                "analysis_summary": {
                    "documents_analyzed": len(successful_extractions),
                    "patterns_found": len(analysis_result.patterns),
                    "cross_references": len(analysis_result.cross_references),
                    "confidence_score": f"{analysis_result.confidence_score:.0%}"
                },
                "key_findings": analysis_result.key_findings[:5],
                "high_priority_items": [
                    p.value for p in analysis_result.patterns
                    if 'high_priority' in p.pattern_type
                ][:5]
            }
            
            if missing:
                result_data["files_not_found"] = missing
            
            return ToolResult(
                success=True,
                data=result_data
            )
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return ToolResult(
                success=False,
                error=f"Λείπει απαιτούμενη βιβλιοθήκη: {str(e)}",
                data=None
            )
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                error=f"Αποτυχία δημιουργίας αναφοράς: {str(e)}",
                data=None
            )


class BatchDocumentAnalysisTool(BaseTool):
    """
    Tool for analyzing multiple documents without generating a full report.
    Returns analysis results that can be used for further processing.
    """
    
    def __init__(self, llm_provider=None):
        super().__init__()
        self.llm_provider = llm_provider
    
    @property
    def name(self) -> str:
        return "analyze_documents"
    
    @property
    def description(self) -> str:
        return """Analyze multiple documents for patterns and insights.
        
        Returns structured analysis without generating a document.
        Useful for preliminary analysis before report generation."""
    
    def _execute_impl(self, file_paths: List[str], **kwargs) -> ToolResult:
        """Execute document analysis."""
        
        if not file_paths:
            return ToolResult(
                success=False,
                error="Δεν παρέχθηκαν αρχεία",
                data=None
            )
        
        paths = [Path(fp) for fp in file_paths if Path(fp).exists()]
        
        if not paths:
            return ToolResult(
                success=False,
                error="Κανένα αρχείο δεν βρέθηκε",
                data=None
            )
        
        try:
            from app.analysis import extract_batch, analyze_documents
            
            # Extract and analyze
            contents = extract_batch(paths)
            successful = [c for c in contents if c.content_type != "error"]
            
            if not successful:
                return ToolResult(
                    success=False,
                    error="Αποτυχία εξαγωγής περιεχομένου",
                    data=None
                )
            
            analysis = analyze_documents(successful, self.llm_provider)
            
            return ToolResult(
                success=True,
                data={
                    "documents_analyzed": len(successful),
                    "summary": analysis.summary,
                    "key_findings": analysis.key_findings,
                    "pattern_count": len(analysis.patterns),
                    "confidence_score": analysis.confidence_score,
                    "top_patterns": [
                        {
                            "type": p.pattern_type,
                            "value": p.value,
                            "frequency": p.frequency,
                            "sources": p.sources
                        }
                        for p in analysis.patterns[:10]
                    ]
                }
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                data=None
            )


def register_intelligence_tools(registry, llm_provider=None):
    """Register intelligence analysis tools with the tool registry."""
    registry.register(IntelligenceReportTool(llm_provider=llm_provider))
    registry.register(BatchDocumentAnalysisTool(llm_provider=llm_provider))
    logger.info("Registered intelligence analysis tools")
# app/api/logistics_routes.py
"""
API endpoints for logistics anomaly detection.

Provides document upload, analysis, and audit report generation.
"""

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil
import uuid

from app.api.auth_routes import get_current_user_dep
from app.utils.logger import setup_logger
from app.config import BASE_DIR

logger = setup_logger(__name__)

router = APIRouter(prefix="/logistics", tags=["Logistics"])

# Upload directory for logistics documents
UPLOAD_DIR = BASE_DIR / "data" / "logistics_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Request/Response Models
# =============================================================================

class AnalysisRequest(BaseModel):
    """Request for logistics analysis."""
    file_paths: List[str] = Field(..., description="Paths to uploaded files")
    generate_report: bool = Field(True, description="Generate audit report")
    report_format: str = Field("docx", description="Report format: docx, pdf, md")
    include_evidence: bool = Field(True, description="Include evidence in findings")


class ComparisonRequest(BaseModel):
    """Request for document comparison."""
    baseline_files: List[str] = Field(..., description="Baseline document paths")
    comparison_files: List[str] = Field(..., description="Comparison document paths")


class AnomalyResponse(BaseModel):
    """Individual anomaly finding."""
    anomaly_id: str
    severity: str
    category: str
    title: str
    description: str
    source_documents: List[str]
    evidence: List[str]
    suggested_actions: List[str]
    confidence: float


class AnalysisResponse(BaseModel):
    """Response from logistics analysis."""
    success: bool
    documents_analyzed: int
    entities_found: int
    anomaly_count: int
    severity_breakdown: Dict[str, int]
    category_breakdown: Dict[str, int]
    confidence_score: str
    recommendations: List[str]
    anomalies: Optional[List[AnomalyResponse]] = None
    report_path: Optional[str] = None
    report_name: Optional[str] = None
    download_url: Optional[str] = None
    error: Optional[str] = None


class ComparisonResponse(BaseModel):
    """Response from document comparison."""
    success: bool
    summary: Dict[str, Any]
    changes: List[Dict[str, Any]]
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Response from file upload."""
    success: bool
    session_id: str
    files: List[Dict[str, Any]]
    total_size_bytes: int


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    user: dict = Depends(get_current_user_dep)
):
    """
    Upload multiple documents for logistics analysis.
    
    Supports: Excel (.xlsx, .xls), CSV, PDF, Word (.docx), text files.
    Returns session ID and file paths for subsequent analysis.
    """
    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    total_size = 0
    
    for file in files:
        try:
            file_path = session_dir / file.filename
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            file_size = len(content)
            total_size += file_size
            
            uploaded_files.append({
                "filename": file.filename,
                "path": str(file_path),
                "size_bytes": file_size,
                "content_type": file.content_type
            })
            
            logger.info(f"Uploaded {file.filename} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
    
    return UploadResponse(
        success=True,
        session_id=session_id,
        files=uploaded_files,
        total_size_bytes=total_size
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_documents(
    request: AnalysisRequest,
    user: dict = Depends(get_current_user_dep)
):
    """
    Analyze uploaded documents for logistics anomalies.
    
    Detects:
    - Inventory discrepancies across documents
    - Budget/expense anomalies
    - Resource allocation conflicts
    - Expiration warnings
    - Maintenance pattern irregularities
    """
    try:
        from app.analysis import extract_batch
        from app.analysis.logistics_analyzer import analyze_logistics
        
        paths = [Path(p) for p in request.file_paths if Path(p).exists()]
        
        if not paths:
            return AnalysisResponse(
                success=False,
                documents_analyzed=0,
                entities_found=0,
                anomaly_count=0,
                severity_breakdown={},
                category_breakdown={},
                confidence_score="0%",
                recommendations=[],
                error="Δεν βρέθηκαν έγκυρα αρχεία"
            )
        
        # Extract content
        contents = extract_batch(paths)
        successful = [c for c in contents if c.content_type != "error"]
        
        if not successful:
            return AnalysisResponse(
                success=False,
                documents_analyzed=0,
                entities_found=0,
                anomaly_count=0,
                severity_breakdown={},
                category_breakdown={},
                confidence_score="0%",
                recommendations=[],
                error="Αποτυχία εξαγωγής περιεχομένου"
            )
        
        # Get LLM provider if available
        llm_provider = None
        try:
            from app.agent.integration import get_agent
            llm_provider = get_agent().llm_provider
        except Exception:
            pass
        
        # Analyze
        result = analyze_logistics(successful, llm_provider)
        
        # Build response
        response = AnalysisResponse(
            success=True,
            documents_analyzed=result.documents_analyzed,
            entities_found=result.entities_found,
            anomaly_count=len(result.anomalies),
            severity_breakdown=result.severity_summary,
            category_breakdown=result.category_summary,
            confidence_score=f"{result.confidence_score:.0%}",
            recommendations=result.recommendations,
            anomalies=[
                AnomalyResponse(
                    anomaly_id=a.anomaly_id,
                    severity=a.severity.value,
                    category=a.category.value,
                    title=a.title,
                    description=a.description,
                    source_documents=a.source_documents,
                    evidence=a.evidence if request.include_evidence else [],
                    suggested_actions=a.suggested_actions,
                    confidence=a.confidence
                )
                for a in result.anomalies
            ]
        )
        
        # Generate report if requested
        if request.generate_report and result.anomalies:
            from app.tools.logistics_anomaly_tool import LogisticsAnomalyTool
            tool = LogisticsAnomalyTool(llm_provider=llm_provider)
            report_path = tool._generate_audit_report(result, request.report_format)
            
            response.report_path = str(report_path)
            response.report_name = report_path.name
            response.download_url = f"/logistics/download/{report_path.name}"
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return AnalysisResponse(
            success=False,
            documents_analyzed=0,
            entities_found=0,
            anomaly_count=0,
            severity_breakdown={},
            category_breakdown={},
            confidence_score="0%",
            recommendations=[],
            error=str(e)
        )


@router.post("/compare", response_model=ComparisonResponse)
async def compare_documents(
    request: ComparisonRequest,
    user: dict = Depends(get_current_user_dep)
):
    """
    Compare two sets of documents to identify changes.
    
    Useful for:
    - Period-over-period inventory comparisons
    - Audit reconciliation
    - Change detection between reports
    """
    try:
        from app.tools.logistics_anomaly_tool import LogisticsComparisonTool
        
        tool = LogisticsComparisonTool()
        result = tool._execute_impl(
            baseline_files=request.baseline_files,
            comparison_files=request.comparison_files
        )
        
        if result.success:
            return ComparisonResponse(
                success=True,
                summary=result.data.get("summary", {}),
                changes=result.data.get("changes", [])
            )
        else:
            return ComparisonResponse(
                success=False,
                summary={},
                changes=[],
                error=result.error
            )
            
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        return ComparisonResponse(
            success=False,
            summary={},
            changes=[],
            error=str(e)
        )


@router.get("/download/{filename}")
async def download_report(
    filename: str,
    user: dict = Depends(get_current_user_dep)
):
    """Download a generated audit report."""
    # Check in outputs directory
    output_dir = BASE_DIR / "outputs" / "audit_reports"
    file_path = output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Αρχείο δεν βρέθηκε: {filename}"
        )
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )


@router.delete("/session/{session_id}")
async def cleanup_session(
    session_id: str,
    user: dict = Depends(get_current_user_dep)
):
    """Clean up uploaded files for a session."""
    session_dir = UPLOAD_DIR / session_id
    
    if session_dir.exists():
        shutil.rmtree(session_dir)
        logger.info(f"Cleaned up session {session_id}")
        return {"success": True, "message": f"Session {session_id} cleaned up"}
    
    return {"success": False, "message": "Session not found"}


@router.get("/categories")
async def get_anomaly_categories():
    """Get list of anomaly categories with descriptions."""
    from app.analysis.logistics_analyzer import AnomalyCategory, AnomalySeverity
    
    return {
        "categories": [
            {
                "id": cat.value,
                "name": cat.name,
                "description": {
                    "inventory_discrepancy": "Διαφορές στις καταμετρήσεις αποθέματος",
                    "budget_anomaly": "Ασυνήθιστες δαπάνες ή υπερβάσεις προϋπολογισμού",
                    "supply_chain_gap": "Κενά στην εφοδιαστική αλυσίδα",
                    "resource_conflict": "Πόροι με διπλές αναθέσεις",
                    "maintenance_pattern": "Ανωμαλίες στα μοτίβα συντήρησης",
                    "expiration_warning": "Προϊόντα που πλησιάζουν λήξη",
                    "usage_anomaly": "Ασυνήθιστα μοτίβα χρήσης",
                    "documentation_mismatch": "Ασυμφωνίες μεταξύ εγγράφων"
                }.get(cat.value, "")
            }
            for cat in AnomalyCategory
        ],
        "severities": [
            {
                "id": sev.value,
                "name": sev.name,
                "priority": idx + 1
            }
            for idx, sev in enumerate(AnomalySeverity)
        ]
    }
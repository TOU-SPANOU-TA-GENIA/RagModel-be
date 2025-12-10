# app/api/intelligence_routes.py
"""
API endpoints for intelligence analysis and report generation.

Provides batch file upload, analysis, and briefing generation.
"""

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import shutil
import uuid

from app.api.auth_routes import get_current_user_dep
from app.utils.logger import setup_logger
from app.config import BASE_DIR

logger = setup_logger(__name__)

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

# Upload directory for analysis files
UPLOAD_DIR = BASE_DIR / "data" / "intelligence_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Request/Response Models
# =============================================================================

class AnalysisRequest(BaseModel):
    """Request for document analysis."""
    file_paths: List[str] = Field(..., description="Paths to uploaded files")


class ReportRequest(BaseModel):
    """Request for report generation."""
    file_paths: List[str] = Field(..., description="Paths to uploaded files")
    title: str = Field("Αναφορά Πληροφοριών", description="Report title")
    classification: str = Field("ΑΔΙΑΒΑΘΜΗΤΟ", description="Classification level")
    output_format: str = Field("docx", description="Output format: docx, pdf, md")
    include_timeline: bool = Field(True)
    include_entity_graph: bool = Field(True)


class AnalysisResponse(BaseModel):
    """Response from document analysis."""
    success: bool
    documents_analyzed: int
    summary: Optional[str] = None
    key_findings: List[str] = []
    pattern_count: int = 0
    confidence_score: float = 0.0
    error: Optional[str] = None


class ReportResponse(BaseModel):
    """Response from report generation."""
    success: bool
    report_path: Optional[str] = None
    report_name: Optional[str] = None
    download_url: Optional[str] = None
    analysis_summary: Optional[dict] = None
    key_findings: List[str] = []
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Response from file upload."""
    success: bool
    session_id: str
    files: List[dict]
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
    Upload multiple documents for intelligence analysis.
    
    Returns a session ID and file paths for subsequent analysis.
    """
    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    total_size = 0
    
    for file in files:
        try:
            safe_name = file.filename.replace("/", "_").replace("\\", "_")
            file_path = session_dir / safe_name
            
            content = await file.read()
            file_path.write_bytes(content)
            
            file_size = len(content)
            total_size += file_size
            
            uploaded_files.append({
                "name": safe_name,
                "path": str(file_path),
                "size_bytes": file_size,
                "content_type": file.content_type
            })
            
            logger.info(f"Uploaded: {safe_name} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            uploaded_files.append({
                "name": file.filename,
                "path": None,
                "error": str(e)
            })
    
    return UploadResponse(
        success=len([f for f in uploaded_files if f.get("path")]) > 0,
        session_id=session_id,
        files=uploaded_files,
        total_size_bytes=total_size
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_documents_endpoint(
    request: AnalysisRequest,
    user: dict = Depends(get_current_user_dep)
):
    """
    Analyze uploaded documents for intelligence patterns.
    """
    try:
        from app.analysis import extract_batch, analyze_documents
        
        paths = [Path(p) for p in request.file_paths if Path(p).exists()]
        
        if not paths:
            return AnalysisResponse(
                success=False,
                documents_analyzed=0,
                error="Δεν βρέθηκαν έγκυρα αρχεία"
            )
        
        contents = extract_batch(paths)
        successful = [c for c in contents if c.content_type != "error"]
        
        if not successful:
            return AnalysisResponse(
                success=False,
                documents_analyzed=0,
                error="Αποτυχία εξαγωγής περιεχομένου"
            )
        
        result = analyze_documents(successful)
        
        return AnalysisResponse(
            success=True,
            documents_analyzed=len(successful),
            summary=result.summary,
            key_findings=result.key_findings,
            pattern_count=len(result.patterns),
            confidence_score=result.confidence_score
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return AnalysisResponse(
            success=False,
            documents_analyzed=0,
            error=str(e)
        )


@router.post("/report", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    user: dict = Depends(get_current_user_dep)
):
    """
    Generate intelligence briefing report from uploaded documents.
    """
    try:
        from app.analysis import (
            extract_batch,
            analyze_documents,
            generate_briefing,
            BriefingConfig,
            ClassificationLevel
        )
        
        paths = [Path(p) for p in request.file_paths if Path(p).exists()]
        
        if not paths:
            return ReportResponse(success=False, error="Δεν βρέθηκαν έγκυρα αρχεία")
        
        contents = extract_batch(paths)
        successful = [c for c in contents if c.content_type != "error"]
        
        if not successful:
            return ReportResponse(success=False, error="Αποτυχία εξαγωγής περιεχομένου")
        
        # Get LLM provider if available
        llm_provider = None
        try:
            from app.agent.integration import get_agent
            llm_provider = get_agent().llm_provider
        except Exception:
            pass
        
        analysis = analyze_documents(successful, llm_provider)
        
        classification_map = {
            "ΑΔΙΑΒΑΘΜΗΤΟ": ClassificationLevel.UNCLASSIFIED,
            "ΠΕΡΙΟΡΙΣΜΕΝΗΣ ΧΡΗΣΗΣ": ClassificationLevel.RESTRICTED,
            "ΕΜΠΙΣΤΕΥΤΙΚΟ": ClassificationLevel.CONFIDENTIAL,
            "ΑΠΟΡΡΗΤΟ": ClassificationLevel.SECRET
        }
        class_level = classification_map.get(request.classification, ClassificationLevel.UNCLASSIFIED)
        
        config = BriefingConfig(
            classification=class_level,
            include_timeline=request.include_timeline,
            include_entity_graph=request.include_entity_graph,
            output_format=request.output_format
        )
        
        report_path = generate_briefing(
            analysis=analysis,
            documents=successful,
            title=request.title,
            config=config
        )
        
        return ReportResponse(
            success=True,
            report_path=str(report_path),
            report_name=report_path.name,
            download_url=f"/intelligence/download/{report_path.name}",
            analysis_summary={
                "documents_analyzed": len(successful),
                "patterns_found": len(analysis.patterns),
                "cross_references": len(analysis.cross_references),
                "confidence_score": f"{analysis.confidence_score:.0%}"
            },
            key_findings=analysis.key_findings[:5]
        )
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        return ReportResponse(success=False, error=str(e))


@router.get("/download/{filename}")
async def download_report(
    filename: str,
    user: dict = Depends(get_current_user_dep)
):
    """Download a generated report."""
    output_dir = BASE_DIR / "outputs" / "briefings"
    file_path = output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Το αρχείο δεν βρέθηκε")
    
    media_types = {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pdf": "application/pdf",
        ".md": "text/markdown"
    }
    media_type = media_types.get(file_path.suffix.lower(), "application/octet-stream")
    
    return FileResponse(path=str(file_path), filename=filename, media_type=media_type)


@router.delete("/session/{session_id}")
async def cleanup_session(
    session_id: str,
    user: dict = Depends(get_current_user_dep)
):
    """Clean up uploaded files from a session."""
    session_dir = UPLOAD_DIR / session_id
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session δεν βρέθηκε")
    
    try:
        shutil.rmtree(session_dir)
        return {"success": True, "message": f"Session {session_id} διαγράφηκε"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    try:
        from app.analysis import get_extractor_registry
        return {
            "supported_extensions": get_extractor_registry().supported_extensions,
            "output_formats": ["docx", "pdf", "md"]
        }
    except Exception:
        return {
            "supported_extensions": [".txt", ".md", ".pdf", ".docx", ".xlsx", ".png", ".jpg"],
            "output_formats": ["docx", "pdf", "md"]
        }
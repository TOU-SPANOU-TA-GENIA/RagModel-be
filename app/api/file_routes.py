# app/api/file_routes.py
"""
File upload and download API routes.
Handles file uploads, downloads, and file generation.
"""

import base64
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from app.api.auth_routes import get_current_user_dep
from app.services.file_service import file_service, FileMetadata
from app.services.file_generator import FileGenerator, GeneratedFile
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


# =============================================================================
# Request/Response Models
# =============================================================================

class FileUploadResponse(BaseModel):
    """Response from file upload."""
    file_id: str
    original_name: str
    mime_type: str
    size_bytes: int
    content_type: str
    extracted_preview: str


class FileMetadataResponse(BaseModel):
    """File metadata response."""
    file_id: str
    original_name: str
    mime_type: str
    size_bytes: int
    upload_time: str
    content_type: str
    has_content: bool


class GenerateFileRequest(BaseModel):
    """Request to generate a file."""
    content: str
    file_type: str = "docx"  # docx, xlsx, pdf, txt, md, csv
    title: Optional[str] = None
    filename: Optional[str] = None
    data: Optional[List[List]] = None  # For spreadsheets
    headers: Optional[List[str]] = None  # For spreadsheets


class GeneratedFileResponse(BaseModel):
    """Response with generated file."""
    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    download_url: str


# =============================================================================
# Upload Endpoints
# =============================================================================

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    chat_id: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user_dep)
):
    """
    Upload a file for processing.
    
    The file is stored in Redis and content is extracted for AI processing.
    """
    if not file_service.is_available:
        raise HTTPException(
            status_code=503,
            detail="File storage unavailable. Redis may not be running."
        )
    
    try:
        # Read file data
        file_data = await file.read()
        
        # Upload and extract content
        metadata = file_service.upload_file(
            file_data=file_data,
            filename=file.filename or "uploaded_file",
            chat_id=chat_id,
            user_id=current_user["id"]
        )
        
        if not metadata:
            raise HTTPException(status_code=500, detail="Upload failed")
        
        # Return response with preview of extracted content
        preview = metadata.extracted_content[:500] + "..." if len(metadata.extracted_content) > 500 else metadata.extracted_content
        
        return FileUploadResponse(
            file_id=metadata.file_id,
            original_name=metadata.original_name,
            mime_type=metadata.mime_type,
            size_bytes=metadata.size_bytes,
            content_type=metadata.content_type,
            extracted_preview=preview
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-base64", response_model=FileUploadResponse)
async def upload_file_base64(
    filename: str,
    base64_data: str,
    chat_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user_dep)
):
    """Upload a file from base64 encoded data."""
    if not file_service.is_available:
        raise HTTPException(status_code=503, detail="File storage unavailable")
    
    try:
        file_data = base64.b64decode(base64_data)
        
        metadata = file_service.upload_file(
            file_data=file_data,
            filename=filename,
            chat_id=chat_id,
            user_id=current_user["id"]
        )
        
        if not metadata:
            raise HTTPException(status_code=500, detail="Upload failed")
        
        preview = metadata.extracted_content[:500]
        
        return FileUploadResponse(
            file_id=metadata.file_id,
            original_name=metadata.original_name,
            mime_type=metadata.mime_type,
            size_bytes=metadata.size_bytes,
            content_type=metadata.content_type,
            extracted_preview=preview
        )
        
    except Exception as e:
        logger.error(f"Base64 upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Download Endpoints
# =============================================================================

@router.get("/download/{file_id}")
async def download_file(
    file_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """
    Download a file by ID.
    
    Returns the file with proper Content-Disposition for browser download.
    """
    result = file_service.get_file(file_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_data, metadata = result
    
    # Verify user has access
    if metadata.user_id and metadata.user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return Response(
        content=file_data,
        media_type=metadata.mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="{metadata.original_name}"',
            "Content-Length": str(metadata.size_bytes)
        }
    )


@router.get("/content/{file_id}")
async def get_file_content(
    file_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Get extracted text content from a file."""
    metadata = file_service.get_metadata(file_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")
    
    if metadata.user_id and metadata.user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "file_id": file_id,
        "filename": metadata.original_name,
        "content": metadata.extracted_content,
        "content_type": metadata.content_type
    }


@router.get("/metadata/{file_id}", response_model=FileMetadataResponse)
async def get_file_metadata(
    file_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Get file metadata without downloading the file."""
    metadata = file_service.get_metadata(file_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")
    
    if metadata.user_id and metadata.user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileMetadataResponse(
        file_id=metadata.file_id,
        original_name=metadata.original_name,
        mime_type=metadata.mime_type,
        size_bytes=metadata.size_bytes,
        upload_time=metadata.upload_time,
        content_type=metadata.content_type,
        has_content=bool(metadata.extracted_content)
    )


@router.get("/chat/{chat_id}", response_model=List[FileMetadataResponse])
async def get_chat_files(
    chat_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Get all files associated with a chat."""
    files = file_service.get_chat_files(chat_id)
    
    return [
        FileMetadataResponse(
            file_id=f.file_id,
            original_name=f.original_name,
            mime_type=f.mime_type,
            size_bytes=f.size_bytes,
            upload_time=f.upload_time,
            content_type=f.content_type,
            has_content=bool(f.extracted_content)
        )
        for f in files
    ]


# =============================================================================
# File Generation Endpoints
# =============================================================================

@router.post("/generate")
async def generate_file(
    request: GenerateFileRequest,
    current_user: dict = Depends(get_current_user_dep)
):
    """
    Generate a downloadable file from content.
    
    Supports: docx, xlsx, pdf, txt, md, csv
    """
    try:
        generated = FileGenerator.create_from_type(
            file_type=request.file_type,
            content=request.content,
            title=request.title,
            data=request.data,
            headers=request.headers,
            filename=request.filename
        )
        
        # Store in Redis for download
        if file_service.is_available:
            metadata = file_service.upload_file(
                file_data=generated.data,
                filename=generated.filename,
                user_id=current_user["id"]
            )
            
            return {
                "file_id": metadata.file_id,
                "filename": generated.filename,
                "mime_type": generated.mime_type,
                "size_bytes": generated.size_bytes,
                "download_url": f"/files/download/{metadata.file_id}"
            }
        else:
            # Return base64 encoded if Redis unavailable
            return {
                "filename": generated.filename,
                "mime_type": generated.mime_type,
                "size_bytes": generated.size_bytes,
                "data_base64": generated.to_base64()
            }
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"File generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-direct")
async def generate_file_direct(
    request: GenerateFileRequest,
    current_user: dict = Depends(get_current_user_dep)
):
    """
    Generate and immediately download a file.
    
    Returns the file directly without storing in Redis.
    """
    try:
        generated = FileGenerator.create_from_type(
            file_type=request.file_type,
            content=request.content,
            title=request.title,
            data=request.data,
            headers=request.headers,
            filename=request.filename
        )
        
        return Response(
            content=generated.data,
            media_type=generated.mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{generated.filename}"',
                "Content-Length": str(generated.size_bytes)
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"File generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Delete Endpoint
# =============================================================================

@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Delete a file."""
    metadata = file_service.get_metadata(file_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")
    
    if metadata.user_id and metadata.user_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = file_service.delete_file(file_id)
    
    return {"success": success, "file_id": file_id}
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
import shutil
import os
from pathlib import Path
from app.api.schemas import FileInfo, FileUploadResponse, User
from app.api.deps import get_current_user
from app.config import get_config

router = APIRouter()

def get_upload_dir() -> Path:
    config = get_config()
    # Default to 'data/uploads' if not configured
    path = Path(config.get("files", {}).get("upload_path", "data/uploads"))
    path.mkdir(parents=True, exist_ok=True)
    return path

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    try:
        upload_dir = get_upload_dir()
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return FileUploadResponse(
            filename=file.filename,
            status="success",
            message=f"File saved to {file_path}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[FileInfo])
async def list_files(current_user: User = Depends(get_current_user)):
    upload_dir = get_upload_dir()
    files = []
    for f in upload_dir.glob("*"):
        if f.is_file():
            stat = f.stat()
            files.append(FileInfo(
                filename=f.name,
                path=str(f),
                size=stat.st_size,
                modified_at=stat.st_mtime
            ))
    return files
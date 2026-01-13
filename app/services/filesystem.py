import os
import shutil
from pathlib import Path
from typing import List, Optional, BinaryIO
from datetime import datetime

from app.config import get_config
from app.api.schemas import FileInfo
from app.core.exceptions import AppError
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class FileSystemService:
    """
    Centralized file operations.
    Enforces security boundaries defined in 'paths' config.
    """
    
    def __init__(self):
        self.config = get_config()
        self.base_path = Path(self.config.get("paths", {}).get("data_dir", "data"))
        self.upload_path = self.base_path / "uploads"
        self.output_path = self.base_path / "outputs"
        
        # Ensure directories exist
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, path: Path) -> Path:
        """Security check to prevent directory traversal."""
        try:
            full_path = path.resolve()
            if not str(full_path).startswith(str(self.base_path.resolve())):
                raise AppError("Access denied: Path outside data directory.", code="SECURITY_ERROR")
            return full_path
        except Exception:
             raise AppError("Invalid path.", code="PATH_ERROR")

    def list_files(self, subdir: str = "uploads") -> List[FileInfo]:
        """List files in a specific subdirectory."""
        target_dir = self.base_path / subdir
        if not target_dir.exists():
            return []
            
        files = []
        for f in target_dir.glob("*"):
            if f.is_file():
                stat = f.stat()
                files.append(FileInfo(
                    filename=f.name,
                    path=str(f),
                    size=stat.st_size,
                    modified_at=stat.st_mtime
                ))
        return files

    def save_file(self, filename: str, content: bytes, subdir: str = "uploads") -> str:
        """Save binary content to disk."""
        target_dir = self.base_path / subdir
        target_dir.mkdir(exist_ok=True)
        
        file_path = target_dir / filename
        # Basic sanitation could go here
        
        with open(file_path, "wb") as f:
            f.write(content)
            
        logger.info(f"File saved: {file_path}")
        return str(file_path)

    def read_file(self, filename: str, subdir: str = "uploads") -> str:
        """Read text content from file."""
        file_path = self.base_path / subdir / filename
        self._validate_path(file_path)
        
        if not file_path.exists():
            raise AppError(f"File {filename} not found", code="FILE_NOT_FOUND")
            
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise AppError(f"Error reading file: {e}", code="READ_ERROR")

    def delete_file(self, filename: str, subdir: str = "uploads"):
        file_path = self.base_path / subdir / filename
        self._validate_path(file_path)
        if file_path.exists():
            os.remove(file_path)
            logger.info(f"File deleted: {file_path}")

# Global instance
filesystem_service = FileSystemService()
# app/services/file_server.py
"""
Windows File Server Integration Service.
Works with SMB shares via mapped drives or UNC paths.
"""

import os
import re
import shutil
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Information about a file on the server."""
    name: str
    path: str
    size: int
    modified: datetime
    is_directory: bool
    extension: str = ""
    
    def __post_init__(self):
        if not self.is_directory:
            self.extension = Path(self.name).suffix.lower()


@dataclass
class SearchResult:
    """Result from a file search."""
    files: List[FileInfo]
    total_count: int
    search_path: str
    query: str


@dataclass
class FileServerConfig:
    """Configuration for file server connection."""
    base_path: str
    mount_point: str
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.xlsx', '.xls', '.csv', '.pdf', '.docx', '.doc', '.txt', '.md'
    ])
    max_file_size: int = 50 * 1024 * 1024
    folder_aliases: Dict[str, str] = field(default_factory=dict)


class FileServerService:
    """
    Service for accessing files from a Windows file server.
    Supports mapped drives (Z:) and UNC paths (\\\\server\\share).
    """
    
    def __init__(self, config: FileServerConfig):
        self.config = config
        self._connected = self._validate_connection()
        
    def _validate_connection(self) -> bool:
        """Verify the file server is accessible."""
        try:
            mount_path = Path(self.config.mount_point)
            if mount_path.exists():
                logger.info(f"File server connected: {self.config.mount_point}")
                return True
            else:
                logger.warning(f"Mount point not accessible: {self.config.mount_point}")
                return False
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def _resolve_path(self, path_or_alias: str) -> Path:
        """Resolve a path or folder alias to actual filesystem path."""
        if not path_or_alias:
            return Path(self.config.mount_point)
        
        normalized = path_or_alias.strip().lower()
        
        # Check folder aliases (case-insensitive)
        for alias, actual_path in self.config.folder_aliases.items():
            if normalized == alias.lower() or alias.lower() in normalized:
                resolved = Path(self.config.mount_point) / actual_path
                logger.info(f"Resolved alias '{path_or_alias}' -> '{resolved}'")
                return resolved
        
        # Try direct path
        direct = Path(self.config.mount_point) / path_or_alias
        if direct.exists():
            return direct
        
        # Fuzzy search for folder
        return self._fuzzy_find_folder(path_or_alias)
    
    def _fuzzy_find_folder(self, query: str) -> Path:
        """Find a folder by fuzzy matching."""
        base = Path(self.config.mount_point)
        query_lower = query.lower().strip()
        
        # Remove accents for comparison
        query_normalized = self._normalize_greek(query_lower)
        
        best_match = None
        best_score = 0
        
        try:
            for item in base.iterdir():
                if item.is_dir():
                    dir_name = item.name.lower()
                    dir_normalized = self._normalize_greek(dir_name)
                    
                    # Exact match
                    if dir_name == query_lower or dir_normalized == query_normalized:
                        return item
                    
                    # Partial match scoring
                    score = 0
                    if query_lower in dir_name or query_normalized in dir_normalized:
                        score += 5
                    
                    # Word overlap
                    query_words = set(query_lower.split())
                    dir_words = set(dir_name.replace('_', ' ').split())
                    score += len(query_words & dir_words) * 2
                    
                    if score > best_score:
                        best_score = score
                        best_match = item
        except PermissionError:
            logger.error(f"Permission denied accessing: {base}")
        
        return best_match if best_match else base
    
    def _normalize_greek(self, text: str) -> str:
        """Normalize Greek text by removing accents."""
        replacements = {
            'ά': 'α', 'έ': 'ε', 'ή': 'η', 'ί': 'ι', 'ϊ': 'ι', 'ΐ': 'ι',
            'ό': 'ο', 'ύ': 'υ', 'ϋ': 'υ', 'ΰ': 'υ', 'ώ': 'ω'
        }
        for accented, plain in replacements.items():
            text = text.replace(accented, plain)
        return text
    
    def list_folder(self, path_or_alias: str = "") -> List[FileInfo]:
        """List contents of a folder."""
        folder_path = self._resolve_path(path_or_alias)
        
        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            return []
        
        results = []
        try:
            for item in folder_path.iterdir():
                try:
                    stat = item.stat()
                    results.append(FileInfo(
                        name=item.name,
                        path=str(item),
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime),
                        is_directory=item.is_dir()
                    ))
                except (PermissionError, OSError):
                    continue
        except PermissionError:
            logger.error(f"Permission denied: {folder_path}")
        
        results.sort(key=lambda x: (not x.is_directory, x.name.lower()))
        return results
    
    def get_files(
        self,
        folder: str,
        extensions: Optional[List[str]] = None,
        pattern: Optional[str] = None
    ) -> List[str]:
        """Get file paths from a folder."""
        folder_path = self._resolve_path(folder)
        
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            return []
        
        if extensions is None:
            extensions = self.config.allowed_extensions
        
        regex = re.compile(pattern, re.IGNORECASE) if pattern else None
        
        results = []
        try:
            for item in folder_path.iterdir():
                if item.is_file():
                    ext = item.suffix.lower()
                    if extensions and ext not in extensions:
                        continue
                    if regex and not regex.search(item.name):
                        continue
                    try:
                        if item.stat().st_size <= self.config.max_file_size:
                            results.append(str(item))
                    except OSError:
                        continue
        except PermissionError:
            logger.error(f"Permission denied: {folder_path}")
        
        logger.info(f"Found {len(results)} files in {folder_path}")
        return results
    
    def search_files(
        self,
        query: str,
        folder: str = "",
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        max_results: int = 50
    ) -> SearchResult:
        """Search for files matching a query."""
        search_path = self._resolve_path(folder)
        
        if extensions is None:
            extensions = self.config.allowed_extensions
        
        query_lower = query.lower()
        results = []
        
        try:
            if recursive:
                items = search_path.rglob('*')
            else:
                items = search_path.glob('*')
            
            for item in items:
                if not item.is_file():
                    continue
                
                ext = item.suffix.lower()
                if extensions and ext not in extensions:
                    continue
                
                # Score relevance
                name_lower = item.name.lower()
                score = 0
                if query_lower in name_lower:
                    score += 10
                if name_lower.startswith(query_lower):
                    score += 5
                
                if score > 0:
                    try:
                        stat = item.stat()
                        if stat.st_size <= self.config.max_file_size:
                            results.append((score, FileInfo(
                                name=item.name,
                                path=str(item),
                                size=stat.st_size,
                                modified=datetime.fromtimestamp(stat.st_mtime),
                                is_directory=False
                            )))
                    except OSError:
                        continue
                
                if len(results) >= max_results * 2:
                    break
                    
        except PermissionError:
            logger.error(f"Permission denied: {search_path}")
        
        results.sort(key=lambda x: -x[0])
        files = [r[1] for r in results[:max_results]]
        
        return SearchResult(
            files=files,
            total_count=len(results),
            search_path=str(search_path),
            query=query
        )
    
    def copy_to_temp(self, file_paths: List[str]) -> List[str]:
        """Copy files to temp directory for processing."""
        temp_dir = tempfile.mkdtemp(prefix="prometheus_")
        temp_paths = []
        
        for src_path in file_paths:
            src = Path(src_path)
            if src.exists() and src.is_file():
                dest = Path(temp_dir) / src.name
                try:
                    shutil.copy2(src, dest)
                    temp_paths.append(str(dest))
                    logger.info(f"Copied: {src.name}")
                except Exception as e:
                    logger.error(f"Failed to copy {src}: {e}")
        
        return temp_paths


def create_file_server(
    mount_point: str,
    base_path: str = "",
    aliases: Optional[Dict[str, str]] = None
) -> FileServerService:
    """Create a FileServerService instance."""
    config = FileServerConfig(
        base_path=base_path or mount_point,
        mount_point=mount_point,
        folder_aliases=aliases or {}
    )
    return FileServerService(config)
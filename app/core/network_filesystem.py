# app/core/network_filesystem.py
"""
Network filesystem integration - monitors and indexes files from network shares.
Supports SMB/CIFS, NFS, and other network filesystems.
Auto-discovers files and updates index when filesystems change.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import os
import time
import threading
import hashlib
from datetime import datetime

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class NetworkShare:
    """Configuration for a network share."""
    name: str  # Friendly name for the share
    mount_path: Path  # Where it's mounted locally
    share_type: str = "smb"  # smb, nfs, local
    enabled: bool = True
    
    # Auto-discovery settings
    auto_index: bool = True  # Automatically index files
    watch_for_changes: bool = True  # Monitor for changes
    scan_interval: int = 300  # Seconds between scans (5 min default)
    
    # Filtering
    include_extensions: List[str] = field(default_factory=lambda: [
        '.txt', '.md', '.pdf', '.doc', '.docx', 
        '.xls', '.xlsx', '.csv', '.json', '.yaml'
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '.*', '~$*', 'Thumbs.db', 'desktop.ini'
    ])
    max_file_size_mb: int = 100  # Don't index files larger than this
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mount_path": str(self.mount_path),
            "share_type": self.share_type,
            "enabled": self.enabled,
            "auto_index": self.auto_index,
            "watch_for_changes": self.watch_for_changes,
            "scan_interval": self.scan_interval,
            "include_extensions": self.include_extensions,
            "exclude_patterns": self.exclude_patterns,
            "max_file_size_mb": self.max_file_size_mb
        }


@dataclass
class FileMetadata:
    """Metadata for a discovered file."""
    path: Path
    share_name: str  # Which share it came from
    filename: str
    extension: str
    size_bytes: int
    modified_time: float
    file_hash: str  # For detecting changes
    indexed: bool = False
    last_indexed: Optional[float] = None


class NetworkFilesystemMonitor:
    """
    Monitors network filesystems and maintains an index of available files.
    Automatically detects changes and updates the index.
    """
    
    def __init__(self, shares: List[NetworkShare]):
        self.shares = {s.name: s for s in shares if s.enabled}
        self.file_index: Dict[str, FileMetadata] = {}  # file_hash -> metadata
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_scan: Dict[str, float] = {name: 0 for name in self.shares}
        
        logger.info(f"Initialized monitor for {len(self.shares)} network shares")
    
    def start(self):
        """Start monitoring network shares."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Monitor already running")
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        # Do initial scan immediately
        self.scan_all_shares()
        
        logger.info("Network filesystem monitor started")
    
    def stop(self):
        """Stop monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Network filesystem monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            for share_name, share in self.shares.items():
                if not share.watch_for_changes:
                    continue
                
                # Check if it's time to scan this share
                time_since_last = time.time() - self._last_scan[share_name]
                if time_since_last >= share.scan_interval:
                    try:
                        self.scan_share(share)
                        self._last_scan[share_name] = time.time()
                    except Exception as e:
                        logger.error(f"Error scanning share {share_name}: {e}")
            
            # Sleep for a bit before next check
            self._stop_event.wait(timeout=10)
    
    def scan_all_shares(self) -> Dict[str, int]:
        """Scan all enabled shares. Returns count of files per share."""
        results = {}
        for share_name, share in self.shares.items():
            if share.auto_index:
                count = self.scan_share(share)
                results[share_name] = count
        return results
    
    def scan_share(self, share: NetworkShare) -> int:
        """
        Scan a single network share and update file index.
        Returns number of files found.
        """
        mount_path = Path(share.mount_path)
        if not mount_path.exists():
            # Try UNC path format
            if not str(mount_path).startswith('\\\\'):
                logger.warning(f"Share mount path does not exist: {mount_path}")
                return 0
        
        logger.info(f"Scanning share: {share.name} at {share.mount_path}")
        
        discovered_files = []
        
        for root, dirs, files in os.walk(share.mount_path):
            root_path = Path(root)
            
            # Filter directories (modify in place to affect walk)
            dirs[:] = [d for d in dirs if not self._should_exclude(d, share)]
            
            for filename in files:
                if self._should_exclude(filename, share):
                    continue
                
                file_path = root_path / filename
                
                # Check extension
                if not self._has_valid_extension(filename, share):
                    continue
                
                # Check file size
                try:
                    size_bytes = file_path.stat().st_size
                    if size_bytes > share.max_file_size_mb * 1024 * 1024:
                        continue
                except:
                    continue
                
                # Create metadata
                try:
                    metadata = self._create_metadata(file_path, share)
                    discovered_files.append(metadata)
                except Exception as e:
                    logger.debug(f"Could not process {file_path}: {e}")
        
        # Update index
        with self._lock:
            for metadata in discovered_files:
                existing = self.file_index.get(metadata.file_hash)
                
                if existing:
                    # Update if modified time changed
                    if existing.modified_time != metadata.modified_time:
                        logger.debug(f"File updated: {metadata.filename}")
                        metadata.indexed = False  # Mark for re-indexing
                        self.file_index[metadata.file_hash] = metadata
                else:
                    # New file
                    logger.debug(f"New file discovered: {metadata.filename}")
                    self.file_index[metadata.file_hash] = metadata
        
        logger.info(f"Share {share.name}: found {len(discovered_files)} files")
        return len(discovered_files)
    
    def _create_metadata(self, file_path: Path, share: NetworkShare) -> FileMetadata:
        """Create metadata for a file."""
        stat = file_path.stat()
        
        # Create hash from path + size + mtime (faster than content hash)
        hash_input = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()
        
        return FileMetadata(
            path=file_path,
            share_name=share.name,
            filename=file_path.name,
            extension=file_path.suffix.lower(),
            size_bytes=stat.st_size,
            modified_time=stat.st_mtime,
            file_hash=file_hash
        )
    
    def _should_exclude(self, name: str, share: NetworkShare) -> bool:
        """Check if file/dir should be excluded."""
        import fnmatch
        
        for pattern in share.exclude_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False
    
    def _has_valid_extension(self, filename: str, share: NetworkShare) -> bool:
        """Check if file has a valid extension."""
        ext = Path(filename).suffix.lower()
        return ext in share.include_extensions
    
    def search_files(self, query: str, share_name: Optional[str] = None) -> List[FileMetadata]:
        """
        Search for files by name.
        
        Args:
            query: Search term (supports wildcards)
            share_name: Limit to specific share (optional)
        
        Returns:
            List of matching FileMetadata objects
        """
        import fnmatch
        
        results = []
        query_lower = query.lower()
        
        with self._lock:
            for metadata in self.file_index.values():
                # Filter by share if specified
                if share_name and metadata.share_name != share_name:
                    continue
                
                # Match filename
                if (fnmatch.fnmatch(metadata.filename.lower(), f"*{query_lower}*") or
                    query_lower in metadata.filename.lower()):
                    results.append(metadata)
        
        # Sort by relevance (exact matches first, then by modified time)
        results.sort(key=lambda m: (
            m.filename.lower() != query_lower,  # Exact match first
            -m.modified_time  # Then newest
        ))
        
        return results
    
    def get_file_by_name(self, filename: str, share_name: Optional[str] = None) -> Optional[FileMetadata]:
        """Get file metadata by exact filename match."""
        matches = self.search_files(filename, share_name)
        
        # Return exact match if found
        for match in matches:
            if match.filename.lower() == filename.lower():
                return match
        
        # Return closest match
        return matches[0] if matches else None
    
    def get_all_files(self, share_name: Optional[str] = None) -> List[FileMetadata]:
        """Get all files, optionally filtered by share."""
        with self._lock:
            if share_name:
                return [m for m in self.file_index.values() if m.share_name == share_name]
            return list(self.file_index.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed files."""
        with self._lock:
            total_files = len(self.file_index)
            total_size = sum(m.size_bytes for m in self.file_index.values())
            
            by_share = {}
            by_extension = {}
            
            for metadata in self.file_index.values():
                # By share
                by_share[metadata.share_name] = by_share.get(metadata.share_name, 0) + 1
                
                # By extension
                ext = metadata.extension or "no_extension"
                by_extension[ext] = by_extension.get(ext, 0) + 1
            
            return {
                "total_files": total_files,
                "total_size_mb": total_size / (1024 * 1024),
                "by_share": by_share,
                "by_extension": by_extension,
                "shares_configured": len(self.shares),
                "last_scan": {name: datetime.fromtimestamp(ts).isoformat() 
                            for name, ts in self._last_scan.items()}
            }
    
    def mark_indexed(self, file_hash: str):
        """Mark a file as indexed (for RAG system integration)."""
        with self._lock:
            if file_hash in self.file_index:
                self.file_index[file_hash].indexed = True
                self.file_index[file_hash].last_indexed = time.time()
    
    def get_unindexed_files(self) -> List[FileMetadata]:
        """Get files that haven't been indexed yet."""
        with self._lock:
            return [m for m in self.file_index.values() if not m.indexed]


# Global instance (will be initialized with config)
network_monitor: Optional[NetworkFilesystemMonitor] = None


def initialize_network_monitor(shares: List[NetworkShare]) -> NetworkFilesystemMonitor:
    """Initialize and start the global network monitor."""
    global network_monitor
    network_monitor = NetworkFilesystemMonitor(shares)
    network_monitor.start()
    return network_monitor


def get_network_monitor() -> Optional[NetworkFilesystemMonitor]:
    """Get the global network monitor instance."""
    return network_monitor
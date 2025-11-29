# app/core/network_rag_integration.py
"""
Integration between network filesystem monitoring and RAG system.
Automatically indexes new/changed files from network shares.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading

from app.core.network_filesystem import NetworkFilesystemMonitor, FileMetadata
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Global instance
_network_integrator: Optional['NetworkRAGIntegrator'] = None


class NetworkRAGIntegrator:
    """
    Bridges network filesystem monitoring and RAG system.
    Automatically indexes discovered files into the knowledge base.
    """
    
    def __init__(self, monitor: NetworkFilesystemMonitor, rag_ingestion_func):
        """
        Args:
            monitor: Network filesystem monitor instance
            rag_ingestion_func: Function to call for indexing files
                               Signature: func(file_path: Path) -> bool
        """
        self.monitor = monitor
        self.rag_ingestion_func = rag_ingestion_func
        self._index_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._index_interval = 60  # Check for new files every 60 seconds
        
        logger.info("Network-RAG integrator initialized")
    
    def start(self):
        """Start automatic indexing of new files."""
        if self._index_thread and self._index_thread.is_alive():
            logger.warning("Indexer already running")
            return
        
        self._stop_event.clear()
        self._index_thread = threading.Thread(target=self._index_loop, daemon=True)
        self._index_thread.start()
        
        logger.info("Network-RAG automatic indexing started")
    
    def stop(self):
        """Stop automatic indexing."""
        if not self._index_thread:
            return
        
        self._stop_event.set()
        if self._index_thread.is_alive():
            self._index_thread.join(timeout=5)
        
        logger.info("Network-RAG automatic indexing stopped")
    
    def _index_loop(self):
        """Background loop that periodically checks for files to index."""
        logger.info(f"Indexing loop started (checking every {self._index_interval}s)")
        
        while not self._stop_event.is_set():
            try:
                self._index_pending_files()
            except Exception as e:
                logger.error(f"Error in indexing loop: {e}")
            
            # Sleep in small increments so we can stop quickly
            for _ in range(self._index_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
    
    def _index_pending_files(self):
        """Index all files that haven't been indexed yet."""
        unindexed = self.monitor.get_unindexed_files()
        
        if not unindexed:
            return
        
        logger.info(f"Found {len(unindexed)} files to index")
        
        indexed_count = 0
        for file_meta in unindexed:
            try:
                # Call the RAG ingestion function
                success = self.rag_ingestion_func(file_meta.file_path)
                
                if success:
                    # Mark as indexed
                    self.monitor.mark_as_indexed(file_meta.file_path)
                    indexed_count += 1
                    logger.info(f"Indexed: {file_meta.filename}")
                else:
                    logger.warning(f"Failed to index: {file_meta.filename}")
            
            except Exception as e:
                logger.error(f"Error indexing {file_meta.filename}: {e}")
        
        if indexed_count > 0:
            logger.info(f"Successfully indexed {indexed_count} files")
    
    def index_all_now(self) -> Dict[str, Any]:
        """
        Immediately index all pending files (synchronous).
        Useful for manual triggering.
        
        Returns:
            Dict with results: {
                "success": bool,
                "files_indexed": int,
                "total_files": int,
                "message": str
            }
        """
        try:
            unindexed = self.monitor.get_unindexed_files()
            total_files = len(self.monitor.get_all_files())
            
            if not unindexed:
                return {
                    "success": True,
                    "files_indexed": 0,
                    "total_files": total_files,
                    "message": "No files to index"
                }
            
            logger.info(f"Manually indexing {len(unindexed)} files...")
            
            indexed_count = 0
            for file_meta in unindexed:
                try:
                    success = self.rag_ingestion_func(file_meta.file_path)
                    
                    if success:
                        self.monitor.mark_as_indexed(file_meta.file_path)
                        indexed_count += 1
                        logger.info(f"✓ Indexed: {file_meta.filename}")
                    else:
                        logger.warning(f"✗ Failed: {file_meta.filename}")
                
                except Exception as e:
                    logger.error(f"Error indexing {file_meta.filename}: {e}")
            
            return {
                "success": True,
                "files_indexed": indexed_count,
                "total_files": total_files,
                "message": f"Indexed {indexed_count} out of {len(unindexed)} pending files"
            }
        
        except Exception as e:
            logger.error(f"Manual indexing failed: {e}")
            return {
                "success": False,
                "files_indexed": 0,
                "total_files": 0,
                "message": f"Error: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get indexing status."""
        stats = self.monitor.get_stats()
        unindexed = self.monitor.get_unindexed_files()
        
        return {
            "total_network_files": stats["total_files"],
            "indexed_files": stats["total_files"] - len(unindexed),
            "pending_files": len(unindexed),
            "shares": stats["by_share"],
            "auto_indexing_active": self._index_thread.is_alive() if self._index_thread else False
        }


class NetworkFileAccessor:
    """
    Provides AI agent access to network files.
    Helper class for finding and reading network files.
    """
    
    def __init__(self, monitor: NetworkFilesystemMonitor):
        """
        Args:
            monitor: Network filesystem monitor instance
        """
        self.monitor = monitor
    
    def find_file(self, filename: str, share_name: Optional[str] = None) -> Optional[Path]:
        """
        Find a file by name in network shares.
        
        Args:
            filename: Name of file to find
            share_name: Optional share name to limit search to
        
        Returns:
            Path to file if found, None otherwise
        """
        results = self.monitor.search_files(filename, share_name=share_name)
        
        if results:
            # Return first match
            return results[0].file_path
        
        return None
    
    def search_files(self, query: str, share_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for files matching query.
        
        Args:
            query: Search query
            share_name: Optional share to limit search to
        
        Returns:
            List of matching files with metadata
        """
        results = self.monitor.search_files(query, share_name=share_name)
        
        return [
            {
                "filename": r.filename,
                "path": str(r.file_path),
                "share": r.share_name,
                "size_mb": round(r.size_bytes / (1024 * 1024), 2),
                "modified": r.modified_time.isoformat() if r.modified_time else None,
                "indexed": r.indexed
            }
            for r in results
        ]
    
    def get_file_content(self, filename: str, share_name: Optional[str] = None) -> Optional[str]:
        """
        Get content of a file.
        
        Args:
            filename: File to read
            share_name: Optional share to limit search to
        
        Returns:
            File content as string, or None if not found/readable
        """
        file_path = self.find_file(filename, share_name)
        
        if not file_path:
            return None
        
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return None
    
    def list_shares(self) -> List[Dict[str, Any]]:
        """List all configured network shares."""
        stats = self.monitor.get_stats()
        
        shares = []
        for share_name, file_count in stats["by_share"].items():
            share_config = self.monitor.shares.get(share_name)
            if share_config:
                shares.append({
                    "name": share_name,
                    "type": share_config.share_type,
                    "path": str(share_config.mount_path),
                    "file_count": file_count,
                    "enabled": share_config.enabled
                })
        
        return shares


# =============================================================================
# Global initialization functions
# =============================================================================

def initialize_network_rag(monitor: NetworkFilesystemMonitor, rag_ingestion_func) -> NetworkRAGIntegrator:
    """
    Initialize and return global network-RAG integrator.
    
    Args:
        monitor: Network filesystem monitor instance
        rag_ingestion_func: Function for indexing files
    
    Returns:
        NetworkRAGIntegrator instance
    """
    global _network_integrator
    
    _network_integrator = NetworkRAGIntegrator(monitor, rag_ingestion_func)
    logger.info("Global network-RAG integrator initialized")
    
    return _network_integrator


def get_network_integrator() -> Optional[NetworkRAGIntegrator]:
    """
    Get the global network-RAG integrator instance.
    
    Returns:
        NetworkRAGIntegrator instance or None if not initialized
    """
    global _network_integrator
    return _network_integrator
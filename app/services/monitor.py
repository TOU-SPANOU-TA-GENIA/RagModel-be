import time
import threading
from pathlib import Path
from typing import List, Dict
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from app.config import get_config
from app.rag.ingestion import ingestion_service
from app.rag.schemas import RagDocument
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class RagEventHandler(FileSystemEventHandler):
    """Handles file system events for the monitor."""
    
    def on_created(self, event):
        if event.is_directory:
            return
        self._process(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        self._process(event.src_path)

    def _process(self, file_path: str):
        path = Path(file_path)
        # Filter for relevant file types
        if path.suffix.lower() not in ['.txt', '.md', '.log', '.csv', '.json', '.pdf', '.docx']:
            return

        # Network drives can be slow to release file locks; wait a bit
        time.sleep(2)
        
        try:
            logger.info(f"Monitor detected change in: {path.name}")
            # Use specific reading strategy (ignore errors for robust network reading)
            content = path.read_text(encoding="utf-8", errors="ignore")
            
            doc = RagDocument(
                content=content,
                source=path.name,
                metadata={
                    "path": str(path),
                    "ingested_via": "monitor",
                    "timestamp": time.time(),
                    "drive_type": "network"
                }
            )
            ingestion_service.ingest_document(doc)
            
        except Exception as e:
            logger.error(f"Failed to ingest monitored file {path}: {e}")

class FolderMonitorService:
    """
    Watches directories defined in config.json.
    Supports Polling for Network Drives (Z:/).
    """
    
    def __init__(self):
        self.observer = None
        self.watches = []
        self._running = False

    def start(self):
        """Start monitoring configured folders."""
        if self._running:
            return

        config = get_config()
        fs_config = config.get("network_filesystem", {})
        
        if not fs_config.get("enabled", False):
            logger.info("Folder monitor disabled in config.")
            return

        shares = fs_config.get("shares", [])
        if not shares:
            return

        # CRITICAL CHANGE: Use PollingObserver if configured (Required for Z: drives)
        if fs_config.get("force_polling", False):
            logger.info("Using PollingObserver (Optimized for Network Drives)")
            # timeout is the check interval in seconds
            self.observer = PollingObserver(timeout=fs_config.get("polling_interval", 5))
        else:
            self.observer = Observer()

        handler = RagEventHandler()
        
        for share in shares:
            path_str = share.get("path")
            if path_str:
                path = Path(path_str)
                # Ensure the drive exists before watching
                if path.exists():
                    try:
                        watch = self.observer.schedule(handler, str(path), recursive=False)
                        self.watches.append(watch)
                        logger.info(f"Monitoring folder: {path}")
                    except Exception as e:
                        logger.error(f"Could not watch {path}: {e}")
                else:
                    logger.warning(f"Network path not found: {path} (Check VPN/Drive Mapping)")

        if self.watches:
            self.observer.start()
            self._running = True
            logger.info("Folder monitoring started.")

    def stop(self):
        """Stop monitoring."""
        if self._running and self.observer:
            self.observer.stop()
            self.observer.join()
            self._running = False
            logger.info("Folder monitoring stopped.")

# Global instance
monitor_service = FolderMonitorService()
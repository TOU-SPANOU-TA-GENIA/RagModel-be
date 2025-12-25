# app/workflows/services/file_watcher.py
"""
File Watcher Service.
Monitors folders for changes and triggers workflows.
"""

import asyncio
import threading
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime
from dataclasses import dataclass, field

from app.workflows.storage import workflow_storage
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Try to import watchdog, fallback to polling if not available
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("watchdog not installed, using polling mode")


@dataclass
class WatchConfig:
    """Configuration for a file watch."""
    workflow_id: str
    watch_path: str
    recursive: bool = True
    include_patterns: List[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: List[str] = field(default_factory=list)
    debounce_seconds: float = 5.0


class FileWatcherService:
    """
    Service that monitors file system changes and triggers workflows.
    
    Uses watchdog library if available, otherwise falls back to polling.
    """
    
    def __init__(self):
        self._watches: Dict[str, WatchConfig] = {}  # workflow_id -> config
        self._observer: Optional["Observer"] = None
        self._polling_task: Optional[asyncio.Task] = None
        self._running = False
        self._pending_events: Dict[str, Dict] = {}  # For debouncing
        self._on_trigger: Optional[Callable] = None
        self._file_states: Dict[str, Dict[str, float]] = {}  # For polling mode
    
    def set_trigger_callback(self, callback: Callable):
        """Set callback for when a workflow should be triggered."""
        self._on_trigger = callback
    
    async def start(self):
        """Start the file watcher service."""
        if self._running:
            return
        
        self._running = True
        
        # Load active watches from database
        await self._load_watches()
        
        if WATCHDOG_AVAILABLE:
            self._start_watchdog()
        else:
            self._polling_task = asyncio.create_task(self._polling_loop())
        
        logger.info(f"👁️  File watcher service started ({len(self._watches)} watches)")
    
    async def stop(self):
        """Stop the file watcher service."""
        self._running = False
        
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
        
        logger.info("👁️  File watcher service stopped")
    
    async def _load_watches(self):
        """Load active watches from database."""
        watches = workflow_storage.get_active_watches()
        
        for watch in watches:
            config = WatchConfig(
                workflow_id=watch['workflow_id'],
                watch_path=watch['watch_path'],
                recursive=watch.get('recursive', True),
                include_patterns=watch.get('include_patterns', ['*']),
                exclude_patterns=watch.get('exclude_patterns', [])
            )
            self._watches[watch['workflow_id']] = config
            
            # Initialize file state for polling
            if not WATCHDOG_AVAILABLE:
                self._file_states[watch['workflow_id']] = self._get_folder_state(
                    config.watch_path, 
                    config
                )
    
    def add_watch(self, config: WatchConfig):
        """Add a new file watch."""
        self._watches[config.workflow_id] = config
        
        if WATCHDOG_AVAILABLE and self._observer and self._observer.is_alive():
            # Add to existing observer
            self._add_watchdog_watch(config)
        elif not WATCHDOG_AVAILABLE:
            self._file_states[config.workflow_id] = self._get_folder_state(
                config.watch_path,
                config
            )
        
        logger.info(f"Added watch for workflow {config.workflow_id}: {config.watch_path}")
    
    def remove_watch(self, workflow_id: str):
        """Remove a file watch."""
        if workflow_id in self._watches:
            del self._watches[workflow_id]
        if workflow_id in self._file_states:
            del self._file_states[workflow_id]
        
        logger.info(f"Removed watch for workflow {workflow_id}")
    
    # =========================================================================
    # Watchdog Mode
    # =========================================================================
    
    def _start_watchdog(self):
        """Start watchdog observer."""
        self._observer = Observer()
        
        for workflow_id, config in self._watches.items():
            self._add_watchdog_watch(config)
        
        self._observer.start()
    
    def _add_watchdog_watch(self, config: WatchConfig):
        """Add a watch to the watchdog observer."""
        if not Path(config.watch_path).exists():
            logger.warning(f"Watch path does not exist: {config.watch_path}")
            return
        
        handler = WorkflowFileHandler(
            config=config,
            on_change=self._on_file_change
        )
        
        self._observer.schedule(
            handler,
            config.watch_path,
            recursive=config.recursive
        )
    
    def _on_file_change(
        self,
        workflow_id: str,
        event_type: str,
        file_path: str
    ):
        """Handle a file change event (with debouncing)."""
        key = f"{workflow_id}:{file_path}"
        
        # Record event for debouncing
        if key not in self._pending_events:
            self._pending_events[key] = {
                'workflow_id': workflow_id,
                'files': set(),
                'event_type': event_type,
                'first_event': datetime.utcnow()
            }
        
        self._pending_events[key]['files'].add(file_path)
        self._pending_events[key]['last_event'] = datetime.utcnow()
        
        # Schedule processing
        config = self._watches.get(workflow_id)
        debounce = config.debounce_seconds if config else 5.0
        
        asyncio.get_event_loop().call_later(
            debounce,
            lambda: asyncio.create_task(self._process_pending_event(key))
        )
    
    async def _process_pending_event(self, key: str):
        """Process a debounced event."""
        if key not in self._pending_events:
            return
        
        event = self._pending_events[key]
        config = self._watches.get(event['workflow_id'])
        
        if not config:
            del self._pending_events[key]
            return
        
        # Check if we should process (no new events in debounce window)
        elapsed = (datetime.utcnow() - event['last_event']).total_seconds()
        if elapsed < config.debounce_seconds:
            return  # Still receiving events, wait
        
        # Trigger workflow
        del self._pending_events[key]
        
        if self._on_trigger:
            files = list(event['files'])
            await self._on_trigger(
                workflow_id=event['workflow_id'],
                trigger_data={
                    'event_type': event['event_type'],
                    'changed_files': [{'path': f, 'name': Path(f).name} for f in files]
                }
            )
    
    # =========================================================================
    # Polling Mode
    # =========================================================================
    
    async def _polling_loop(self):
        """Polling loop for environments without watchdog."""
        poll_interval = 30  # seconds
        
        while self._running:
            try:
                for workflow_id, config in self._watches.items():
                    await self._check_folder_changes(workflow_id, config)
            except Exception as e:
                logger.error(f"Polling error: {e}")
            
            await asyncio.sleep(poll_interval)
    
    async def _check_folder_changes(
        self,
        workflow_id: str,
        config: WatchConfig
    ):
        """Check a folder for changes (polling mode)."""
        current_state = self._get_folder_state(config.watch_path, config)
        previous_state = self._file_states.get(workflow_id, {})
        
        # Find changes
        changed_files = []
        
        for path, mtime in current_state.items():
            if path not in previous_state:
                changed_files.append({'path': path, 'event': 'created'})
            elif mtime > previous_state[path]:
                changed_files.append({'path': path, 'event': 'modified'})
        
        for path in previous_state:
            if path not in current_state:
                changed_files.append({'path': path, 'event': 'deleted'})
        
        # Update state
        self._file_states[workflow_id] = current_state
        
        # Trigger if changes found
        if changed_files and self._on_trigger:
            await self._on_trigger(
                workflow_id=workflow_id,
                trigger_data={
                    'event_type': 'modified',
                    'changed_files': [
                        {'path': f['path'], 'name': Path(f['path']).name}
                        for f in changed_files
                    ]
                }
            )
    
    def _get_folder_state(
        self,
        folder_path: str,
        config: WatchConfig
    ) -> Dict[str, float]:
        """Get current state of a folder (file paths -> modification times)."""
        state = {}
        path = Path(folder_path)
        
        if not path.exists():
            return state
        
        try:
            if config.recursive:
                files = path.rglob('*')
            else:
                files = path.glob('*')
            
            for file_path in files:
                if not file_path.is_file():
                    continue
                
                # Check patterns
                if not self._matches_patterns(file_path.name, config):
                    continue
                
                state[str(file_path)] = file_path.stat().st_mtime
        
        except Exception as e:
            logger.error(f"Error getting folder state: {e}")
        
        return state
    
    def _matches_patterns(self, filename: str, config: WatchConfig) -> bool:
        """Check if filename matches include/exclude patterns."""
        # Check include patterns
        included = any(
            fnmatch.fnmatch(filename, pattern)
            for pattern in config.include_patterns
        )
        
        if not included:
            return False
        
        # Check exclude patterns
        excluded = any(
            fnmatch.fnmatch(filename, pattern)
            for pattern in config.exclude_patterns
        )
        
        return not excluded


if WATCHDOG_AVAILABLE:
    class WorkflowFileHandler(FileSystemEventHandler):
        """Watchdog event handler for workflow triggers."""
        
        def __init__(
            self,
            config: WatchConfig,
            on_change: Callable
        ):
            self.config = config
            self.on_change = on_change
        
        def _should_process(self, event: FileSystemEvent) -> bool:
            """Check if event should be processed."""
            if event.is_directory:
                return False
            
            filename = Path(event.src_path).name
            
            # Check patterns
            included = any(
                fnmatch.fnmatch(filename, pattern)
                for pattern in self.config.include_patterns
            )
            
            excluded = any(
                fnmatch.fnmatch(filename, pattern)
                for pattern in self.config.exclude_patterns
            )
            
            return included and not excluded
        
        def on_created(self, event):
            if self._should_process(event):
                self.on_change(
                    self.config.workflow_id,
                    'created',
                    event.src_path
                )
        
        def on_modified(self, event):
            if self._should_process(event):
                self.on_change(
                    self.config.workflow_id,
                    'modified',
                    event.src_path
                )
        
        def on_deleted(self, event):
            if self._should_process(event):
                self.on_change(
                    self.config.workflow_id,
                    'deleted',
                    event.src_path
                )


# Global instance
file_watcher_service = FileWatcherService()
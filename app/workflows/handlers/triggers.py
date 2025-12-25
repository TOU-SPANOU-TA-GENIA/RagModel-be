# app/workflows/handlers/triggers.py
"""
Trigger node handlers.
These nodes initiate workflow execution based on various conditions.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from app.workflows.engine import NodeResult, ExecutionContext
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


async def handle_file_watcher_trigger(
    config: Dict[str, Any],
    context: ExecutionContext
) -> NodeResult:
    """
    File watcher trigger handler.
    
    This is called when the file watcher service detects changes.
    The trigger_data in context contains information about changed files.
    
    Config:
        watch_path: str - Path to watch
        recursive: bool - Watch subdirectories
        include_patterns: List[str] - File patterns to include
        exclude_patterns: List[str] - File patterns to exclude
    """
    watch_path = config.get('watch_path', '')
    trigger_data = context.trigger_data
    
    # Get changed files from trigger data
    changed_files = trigger_data.get('changed_files', [])
    event_type = trigger_data.get('event_type', 'unknown')
    
    if not changed_files:
        # If triggered manually without file data, scan the folder
        changed_files = _scan_folder(watch_path, config)
    
    # Store file info in context for downstream nodes
    context.set('source_files', changed_files)
    context.set('source_path', watch_path)
    context.set('trigger_time', datetime.utcnow().isoformat())
    context.set('event_type', event_type)
    
    logger.info(f"📁 File watcher triggered: {len(changed_files)} files from {watch_path}")
    
    return NodeResult(
        success=True,
        output={
            'file_count': len(changed_files),
            'files': changed_files[:10],  # Limit for output
            'watch_path': watch_path,
            'event_type': event_type
        }
    )


async def handle_schedule_trigger(
    config: Dict[str, Any],
    context: ExecutionContext
) -> NodeResult:
    """
    Schedule trigger handler.
    
    This is called by the scheduler service when the scheduled time arrives.
    
    Config:
        schedule_type: str - 'cron', 'interval', 'daily', etc.
        Additional config depends on schedule_type
    """
    schedule_type = config.get('schedule_type', 'manual')
    trigger_data = context.trigger_data
    
    context.set('trigger_time', datetime.utcnow().isoformat())
    context.set('schedule_type', schedule_type)
    context.set('scheduled_at', trigger_data.get('scheduled_at'))
    
    logger.info(f"⏰ Schedule triggered: {schedule_type}")
    
    return NodeResult(
        success=True,
        output={
            'schedule_type': schedule_type,
            'trigger_time': context.get('trigger_time'),
            'scheduled_at': context.get('scheduled_at')
        }
    )


async def handle_manual_trigger(
    config: Dict[str, Any],
    context: ExecutionContext
) -> NodeResult:
    """
    Manual trigger handler.
    
    This is called when user manually starts the workflow.
    
    Config:
        input_schema: Optional schema for user input
    """
    trigger_data = context.trigger_data
    
    context.set('trigger_time', datetime.utcnow().isoformat())
    context.set('manual_input', trigger_data.get('input', {}))
    
    logger.info(f"👆 Manual trigger activated")
    
    return NodeResult(
        success=True,
        output={
            'trigger_type': 'manual',
            'trigger_time': context.get('trigger_time'),
            'input': context.get('manual_input')
        }
    )


def _scan_folder(
    folder_path: str,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Scan a folder for files matching patterns."""
    include_patterns = config.get('include_patterns', ['*'])
    exclude_patterns = config.get('exclude_patterns', [])
    recursive = config.get('recursive', True)
    
    # System/junk files to always ignore
    JUNK_FILES = {'thumbs.db', 'desktop.ini', '.ds_store', '.gitkeep', '.gitignore'}
    
    files = []
    path = Path(folder_path)
    
    if not path.exists():
        logger.warning(f"Watch path does not exist: {folder_path}")
        return files
    
    try:
        # Get all files
        if recursive:
            all_files = list(path.rglob('*'))
        else:
            all_files = list(path.glob('*'))
        
        for file_path in all_files:
            if not file_path.is_file():
                continue
            
            # Skip junk files
            if file_path.name.lower() in JUNK_FILES:
                continue
            
            # Check include patterns
            included = any(
                file_path.match(pattern) for pattern in include_patterns
            )
            
            # Check exclude patterns
            excluded = any(
                file_path.match(pattern) for pattern in exclude_patterns
            )
            
            if included and not excluded:
                stat = file_path.stat()
                files.append({
                    'path': str(file_path),
                    'name': file_path.name,
                    'extension': file_path.suffix.lower(),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
    
    except Exception as e:
        logger.error(f"Error scanning folder {folder_path}: {e}")
    
    return files
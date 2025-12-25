# app/workflows/handlers/__init__.py
"""
Workflow node handlers.
Each handler implements the logic for a specific node type.
"""

from app.workflows.handlers.triggers import (
    handle_file_watcher_trigger,
    handle_schedule_trigger,
    handle_manual_trigger
)
from app.workflows.handlers.processors import (
    handle_extract_content,
    handle_llm_analysis,
    handle_anomaly_detection,
    handle_cross_reference,
    handle_summarize
)
from app.workflows.handlers.actions import (
    handle_generate_report,
    handle_send_email,
    handle_save_to_folder
)
from app.workflows.handlers.flow import (
    handle_condition,
    handle_delay
)

__all__ = [
    'handle_file_watcher_trigger',
    'handle_schedule_trigger',
    'handle_manual_trigger',
    'handle_extract_content',
    'handle_llm_analysis',
    'handle_anomaly_detection',
    'handle_cross_reference',
    'handle_summarize',
    'handle_generate_report',
    'handle_send_email',
    'handle_save_to_folder',
    'handle_condition',
    'handle_delay',
]
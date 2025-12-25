# app/workflows/__init__.py
"""
Workflow automation system.
Provides workflow definition, execution, and monitoring.
"""

from app.workflows.engine import workflow_engine, NodeResult, ExecutionContext
from app.workflows.storage import workflow_storage
from app.workflows.models import (
    NodeType, TriggerType, ExecutionStatus,
    Workflow, WorkflowCreate, WorkflowUpdate,
    Execution, ExecutionCreate
)

# Register all node handlers
def register_handlers():
    """Register all built-in node handlers with the engine."""
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
        handle_delay,
        handle_loop,
        handle_merge
    )
    
    # Triggers
    workflow_engine.register_handler('trigger_file_watcher', handle_file_watcher_trigger)
    workflow_engine.register_handler('trigger_schedule', handle_schedule_trigger)
    workflow_engine.register_handler('trigger_manual', handle_manual_trigger)
    
    # Processors
    workflow_engine.register_handler('processor_extract_content', handle_extract_content)
    workflow_engine.register_handler('processor_llm_analysis', handle_llm_analysis)
    workflow_engine.register_handler('processor_anomaly_detection', handle_anomaly_detection)
    workflow_engine.register_handler('processor_cross_reference', handle_cross_reference)
    workflow_engine.register_handler('processor_summarize', handle_summarize)
    
    # Actions
    workflow_engine.register_handler('action_generate_report', handle_generate_report)
    workflow_engine.register_handler('action_send_email', handle_send_email)
    workflow_engine.register_handler('action_save_to_folder', handle_save_to_folder)
    
    # Flow control
    workflow_engine.register_handler('flow_condition', handle_condition)
    workflow_engine.register_handler('flow_delay', handle_delay)
    workflow_engine.register_handler('flow_loop', handle_loop)
    workflow_engine.register_handler('flow_merge', handle_merge)


async def initialize_workflow_system():
    """Initialize the workflow system (call on app startup)."""
    from app.workflows.services.file_watcher import file_watcher_service
    from app.workflows.services.scheduler import scheduler_service
    from app.db.workflow_schema import init_workflow_schema
    
    # Initialize database schema
    init_workflow_schema()
    
    # Register handlers
    register_handlers()
    
    # Set up trigger callbacks
    async def on_file_trigger(workflow_id: str, trigger_data: dict):
        await workflow_engine.start_execution(
            workflow_id=workflow_id,
            trigger_type='file_watcher',
            trigger_data=trigger_data
        )
    
    async def on_schedule_trigger(workflow_id: str, trigger_data: dict):
        await workflow_engine.start_execution(
            workflow_id=workflow_id,
            trigger_type='schedule',
            trigger_data=trigger_data
        )
    
    file_watcher_service.set_trigger_callback(on_file_trigger)
    scheduler_service.set_trigger_callback(on_schedule_trigger)
    
    # Start services
    await file_watcher_service.start()
    await scheduler_service.start()


async def shutdown_workflow_system():
    """Shutdown the workflow system (call on app shutdown)."""
    from app.workflows.services.file_watcher import file_watcher_service
    from app.workflows.services.scheduler import scheduler_service
    
    await file_watcher_service.stop()
    await scheduler_service.stop()


__all__ = [
    'workflow_engine',
    'workflow_storage',
    'NodeResult',
    'ExecutionContext',
    'NodeType',
    'TriggerType',
    'ExecutionStatus',
    'Workflow',
    'WorkflowCreate',
    'WorkflowUpdate',
    'Execution',
    'ExecutionCreate',
    'register_handlers',
    'initialize_workflow_system',
    'shutdown_workflow_system',
]
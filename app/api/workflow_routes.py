# app/api/workflow_routes.py
"""
API routes for workflow management.
IMPORTANT: Static routes must be defined BEFORE parameterized routes (/{workflow_id})
"""

import json
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Body
from typing import List, Optional
from datetime import datetime

from app.workflows.models import (
    WorkflowCreate, WorkflowUpdate, Workflow, WorkflowSummary,
    ExecutionCreate, Execution, ExecutionStatus,
    EmailSettingsUpdate, UserSettings, WorkflowResponse, ExecutionResponse
)
from app.workflows.storage import workflow_storage
from app.workflows.engine import workflow_engine
from app.workflows.services.file_watcher import file_watcher_service, WatchConfig
from app.workflows.services.scheduler import scheduler_service, ScheduleConfig
from app.api.auth_routes import get_current_user_dep
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/workflows", tags=["Workflows"])


# =============================================================================
# STATIC ROUTES (Must be BEFORE /{workflow_id} to avoid conflicts)
# =============================================================================

@router.get("/node-types")
async def get_node_types():
    """Get available node types and their configurations."""
    return {
        "triggers": [
            {
                "type": "trigger_file_watcher",
                "label": "Παρακολούθηση Φακέλου",
                "label_en": "File Watcher",
                "icon": "folder-eye",
                "config_schema": {
                    "watch_path": {"type": "string", "label": "Διαδρομή", "required": True},
                    "recursive": {"type": "boolean", "label": "Υποφάκελοι", "default": True},
                    "include_patterns": {"type": "array", "label": "Τύποι αρχείων", "default": ["*"]},
                    "debounce_seconds": {"type": "number", "label": "Καθυστέρηση (sec)", "default": 5}
                }
            },
            {
                "type": "trigger_schedule",
                "label": "Χρονοπρογραμματισμός",
                "label_en": "Schedule",
                "icon": "clock",
                "config_schema": {
                    "schedule_type": {"type": "select", "options": ["daily", "weekly", "monthly", "cron", "interval"]},
                    "time": {"type": "time", "label": "Ώρα"},
                    "cron_expression": {"type": "string", "label": "Cron"},
                    "interval_seconds": {"type": "number", "label": "Διάστημα (sec)"}
                }
            },
            {
                "type": "trigger_manual",
                "label": "Χειροκίνητη Εκκίνηση",
                "label_en": "Manual",
                "icon": "play"
            }
        ],
        "processors": [
            {
                "type": "processor_extract_content",
                "label": "Εξαγωγή Περιεχομένου",
                "label_en": "Extract Content",
                "icon": "file-text",
                "config_schema": {
                    "file_types": {"type": "array", "default": ["pdf", "docx", "xlsx", "txt", "csv"]},
                    "extract_tables": {"type": "boolean", "default": True},
                    "ocr_enabled": {"type": "boolean", "default": True}
                }
            },
            {
                "type": "processor_llm_analysis",
                "label": "Ανάλυση AI",
                "label_en": "LLM Analysis",
                "icon": "brain",
                "config_schema": {
                    "prompt_template": {"type": "textarea", "label": "Prompt", "required": True},
                    "system_prompt": {"type": "textarea", "label": "System Prompt"},
                    "max_tokens": {"type": "number", "default": 2000},
                    "temperature": {"type": "number", "default": 0.3, "min": 0, "max": 1}
                }
            },
            {
                "type": "processor_anomaly_detection",
                "label": "Ανίχνευση Ανωμαλιών",
                "label_en": "Anomaly Detection",
                "icon": "alert-triangle",
                "config_schema": {
                    "detection_type": {"type": "select", "options": ["statistical", "llm"]},
                    "threshold_std": {"type": "number", "default": 2.0},
                    "use_baseline": {"type": "boolean", "default": True},
                    "fields_to_monitor": {"type": "array", "label": "Πεδία"}
                }
            },
            {
                "type": "processor_summarize",
                "label": "Σύνοψη",
                "label_en": "Summarize",
                "icon": "file-minus",
                "config_schema": {
                    "summary_type": {"type": "select", "options": ["brief", "detailed", "bullets"]},
                    "max_length": {"type": "number", "default": 500}
                }
            }
        ],
        "actions": [
            {
                "type": "action_generate_report",
                "label": "Δημιουργία Αναφοράς",
                "label_en": "Generate Report",
                "icon": "file-chart",
                "config_schema": {
                    "output_format": {"type": "select", "options": ["docx", "pdf", "html", "markdown"]},
                    "output_path": {"type": "string", "label": "Διαδρομή αποθήκευσης"},
                    "filename_template": {"type": "string", "default": "report_{timestamp}"},
                    "include_sections": {"type": "array", "default": ["summary", "details", "recommendations"]}
                }
            },
            {
                "type": "action_send_email",
                "label": "Αποστολή Email",
                "label_en": "Send Email",
                "icon": "mail",
                "config_schema": {
                    "recipients": {"type": "array", "label": "Παραλήπτες", "required": True},
                    "subject_template": {"type": "string", "label": "Θέμα", "required": True},
                    "body_template": {"type": "textarea", "label": "Σώμα"},
                    "attach_report": {"type": "boolean", "default": False}
                }
            },
            {
                "type": "action_save_to_folder",
                "label": "Αποθήκευση σε Φάκελο",
                "label_en": "Save to Folder",
                "icon": "folder-plus",
                "config_schema": {
                    "destination_path": {"type": "string", "label": "Διαδρομή", "required": True},
                    "create_subfolders": {"type": "boolean", "default": True}
                }
            }
        ],
        "flow": [
            {
                "type": "flow_condition",
                "label": "Συνθήκη",
                "label_en": "Condition",
                "icon": "git-branch",
                "config_schema": {
                    "condition_type": {"type": "select", "options": ["expression", "threshold"]},
                    "expression": {"type": "string", "label": "Έκφραση"},
                    "true_branch": {"type": "node_select", "label": "Αληθές"},
                    "false_branch": {"type": "node_select", "label": "Ψευδές"}
                }
            },
            {
                "type": "flow_delay",
                "label": "Καθυστέρηση",
                "label_en": "Delay",
                "icon": "timer",
                "config_schema": {
                    "delay_seconds": {"type": "number", "label": "Δευτερόλεπτα"},
                    "delay_until": {"type": "time", "label": "Ώρα"}
                }
            }
        ]
    }


# -----------------------------------------------------------------------------
# Templates (static routes)
# -----------------------------------------------------------------------------

@router.get("/templates/list")
async def list_templates():
    """Get available workflow templates."""
    from app.workflows.manager import workflow_manager
    return workflow_manager.get_workflow_templates()


@router.post("/templates/{template_id}/create", response_model=WorkflowResponse)
async def create_from_template(
    template_id: str,
    name: Optional[str] = None,
    current_user: dict = Depends(get_current_user_dep)
):
    """Create a workflow from a template."""
    from app.workflows.manager import workflow_manager
    
    templates = workflow_manager.get_workflow_templates()
    template = next((t for t in templates if t['id'] == template_id), None)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    workflow_name = name or template['name']
    
    workflow_id = workflow_manager.create_workflow(
        name=workflow_name,
        user_id=current_user['id'],
        description=template.get('description'),
        nodes=template['nodes'],
        edges=template['edges']
    )
    
    return WorkflowResponse(
        success=True,
        message="Workflow created from template",
        data={'workflow_id': workflow_id}
    )


# -----------------------------------------------------------------------------
# Settings (static routes)
# -----------------------------------------------------------------------------

@router.get("/settings/email", response_model=UserSettings)
async def get_email_settings(
    current_user: dict = Depends(get_current_user_dep)
):
    """Get user email/notification settings."""
    settings = workflow_storage.get_user_settings(current_user['id'])
    
    if not settings:
        return UserSettings(user_id=current_user['id'])
    
    return UserSettings(
        user_id=current_user['id'],
        email_notifications_enabled=settings.get('email_notifications_enabled', False),
        notification_email=settings.get('notification_email'),
        smtp_configured=bool(settings.get('smtp_host'))
    )


@router.put("/settings/email", response_model=WorkflowResponse)
async def update_email_settings(
    data: EmailSettingsUpdate,
    current_user: dict = Depends(get_current_user_dep)
):
    """Update user email/notification settings."""
    settings = {}
    
    if data.email_notifications_enabled is not None:
        settings['email_notifications_enabled'] = data.email_notifications_enabled
    if data.notification_email is not None:
        settings['notification_email'] = data.notification_email
    if data.smtp_host is not None:
        settings['smtp_host'] = data.smtp_host
    if data.smtp_port is not None:
        settings['smtp_port'] = data.smtp_port
    if data.smtp_username is not None:
        settings['smtp_username'] = data.smtp_username
    if data.smtp_password is not None:
        settings['smtp_password'] = data.smtp_password
    if data.smtp_use_tls is not None:
        settings['smtp_use_tls'] = data.smtp_use_tls
    
    workflow_storage.upsert_user_settings(current_user['id'], settings)
    
    return WorkflowResponse(
        success=True,
        message="Email settings updated"
    )


@router.post("/settings/email/test", response_model=WorkflowResponse)
async def test_email_settings(
    current_user: dict = Depends(get_current_user_dep)
):
    """Send a test email to verify settings."""
    settings = workflow_storage.get_user_settings(current_user['id'])
    
    if not settings or not settings.get('notification_email'):
        raise HTTPException(status_code=400, detail="Email not configured")
    
    try:
        from app.workflows.handlers.actions import _send_email
        
        await _send_email(
            recipients=[settings['notification_email']],
            subject="Prometheus - Test Email",
            body="This is a test email from Prometheus workflow system.",
            smtp_config={
                'host': settings.get('smtp_host'),
                'port': settings.get('smtp_port'),
                'username': settings.get('smtp_username'),
                'password': settings.get('smtp_password'),
                'use_tls': settings.get('smtp_use_tls', True)
            }
        )
        
        return WorkflowResponse(
            success=True,
            message="Test email sent"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email failed: {str(e)}")


# -----------------------------------------------------------------------------
# Executions (static routes - before /{workflow_id})
# -----------------------------------------------------------------------------

@router.get("/executions/{execution_id}", response_model=Execution)
async def get_execution(
    execution_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Get execution details."""
    execution = workflow_storage.get_execution(execution_id)
    
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    # Parse JSON string fields in execution
    for field in ['completed_nodes', 'failed_nodes', 'context', 'trigger_data']:
        if isinstance(execution.get(field), str):
            try:
                execution[field] = json.loads(execution[field])
            except:
                execution[field] = {} if field in ['context', 'trigger_data'] else []
    
    # Add node logs and parse JSON strings
    node_logs = workflow_storage.get_node_logs(execution_id)
    for log in node_logs:
        if isinstance(log.get('input_data'), str):
            try:
                log['input_data'] = json.loads(log['input_data'])
            except:
                log['input_data'] = {}
        if isinstance(log.get('output_data'), str):
            try:
                log['output_data'] = json.loads(log['output_data'])
            except:
                log['output_data'] = {}
    
    execution['node_logs'] = node_logs
    
    return Execution(**execution)


@router.post("/executions/{execution_id}/pause", response_model=ExecutionResponse)
async def pause_execution(
    execution_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Pause a running execution."""
    success = await workflow_engine.pause_execution(execution_id)
    
    return ExecutionResponse(
        success=success,
        execution_id=execution_id,
        message="Execution paused" if success else "Failed to pause",
        status=ExecutionStatus.PAUSED if success else None
    )


@router.post("/executions/{execution_id}/resume", response_model=ExecutionResponse)
async def resume_execution(
    execution_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Resume a paused execution."""
    success = await workflow_engine.resume_execution(execution_id)
    
    return ExecutionResponse(
        success=success,
        execution_id=execution_id,
        message="Execution resumed" if success else "Failed to resume",
        status=ExecutionStatus.RUNNING if success else None
    )


@router.post("/executions/{execution_id}/cancel", response_model=ExecutionResponse)
async def cancel_execution(
    execution_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Cancel an execution."""
    success = await workflow_engine.cancel_execution(execution_id)
    
    return ExecutionResponse(
        success=success,
        execution_id=execution_id,
        message="Execution cancelled" if success else "Failed to cancel",
        status=ExecutionStatus.CANCELLED if success else None
    )


# =============================================================================
# WORKFLOW CRUD (parameterized routes - MUST come after static routes)
# =============================================================================

@router.get("/", response_model=List[WorkflowSummary])
async def list_workflows(
    include_shared: bool = True,
    enabled_only: bool = False,
    current_user: dict = Depends(get_current_user_dep)
):
    """List all workflows for the current user."""
    workflows = workflow_storage.list_workflows(
        user_id=current_user['id'],
        include_shared=include_shared,
        enabled_only=enabled_only
    )
    
    return [
        WorkflowSummary(
            id=w['id'],
            name=w['name'],
            description=w.get('description'),
            is_enabled=w['is_enabled'],
            is_shared=w['is_shared'],
            node_count=len(w.get('nodes', [])),
            last_run_at=w.get('last_run_at'),
            updated_at=w['updated_at']
        )
        for w in workflows
    ]


@router.post("/", response_model=WorkflowResponse)
async def create_workflow(
    data: WorkflowCreate,
    current_user: dict = Depends(get_current_user_dep)
):
    """Create a new workflow."""
    workflow_id = workflow_storage.create_workflow(
        name=data.name,
        user_id=current_user['id'],
        description=data.description,
        nodes=[n if isinstance(n, dict) else n.dict() for n in data.nodes],
        edges=[e.dict() for e in data.edges],
        is_shared=data.is_shared
    )
    
    return WorkflowResponse(
        success=True,
        message="Workflow created",
        data={"workflow_id": workflow_id}
    )


@router.get("/{workflow_id}", response_model=Workflow)
async def get_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Get a workflow by ID."""
    workflow = workflow_storage.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Check access
    if workflow['user_id'] != current_user['id'] and not workflow['is_shared']:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return Workflow(**workflow)


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    data: WorkflowUpdate,
    current_user: dict = Depends(get_current_user_dep)
):
    """Update a workflow."""
    workflow = workflow_storage.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if workflow['user_id'] != current_user['id']:
        raise HTTPException(status_code=403, detail="Access denied")
    
    updates = {}
    if data.name is not None:
        updates['name'] = data.name
    if data.description is not None:
        updates['description'] = data.description
    if data.nodes is not None:
        updates['nodes'] = [n if isinstance(n, dict) else n.dict() for n in data.nodes]
    if data.edges is not None:
        updates['edges'] = [e.dict() for e in data.edges]
    if data.is_enabled is not None:
        updates['is_enabled'] = data.is_enabled
    if data.is_shared is not None:
        updates['is_shared'] = data.is_shared
    
    workflow_storage.update_workflow(workflow_id, **updates)
    
    # Update services if enabled state changed
    if data.is_enabled is not None:
        await _sync_workflow_services(workflow_id, data.is_enabled)
    
    return WorkflowResponse(
        success=True,
        message="Workflow updated"
    )


@router.delete("/{workflow_id}", response_model=WorkflowResponse)
async def delete_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Delete a workflow."""
    workflow = workflow_storage.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if workflow['user_id'] != current_user['id']:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Remove from services
    file_watcher_service.remove_watch(workflow_id)
    
    workflow_storage.delete_workflow(workflow_id)
    
    return WorkflowResponse(
        success=True,
        message="Workflow deleted"
    )


@router.post("/{workflow_id}/enable", response_model=WorkflowResponse)
async def enable_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Enable a workflow."""
    workflow = workflow_storage.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_storage.set_workflow_enabled(workflow_id, True)
    await _sync_workflow_services(workflow_id, True)
    
    return WorkflowResponse(
        success=True,
        message="Workflow enabled"
    )


@router.post("/{workflow_id}/disable", response_model=WorkflowResponse)
async def disable_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user_dep)
):
    """Disable a workflow."""
    workflow = workflow_storage.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_storage.set_workflow_enabled(workflow_id, False)
    await _sync_workflow_services(workflow_id, False)
    
    return WorkflowResponse(
        success=True,
        message="Workflow disabled"
    )


@router.post("/{workflow_id}/run", response_model=ExecutionResponse)
async def run_workflow(
    workflow_id: str,
    data: Optional[ExecutionCreate] = Body(default=None),
    background_tasks: BackgroundTasks = None,
    current_user: dict = Depends(get_current_user_dep)
):
    """Manually run a workflow."""
    workflow = workflow_storage.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if workflow['user_id'] != current_user['id'] and not workflow['is_shared']:
        raise HTTPException(status_code=403, detail="Access denied")
    
    trigger_data = data.trigger_data if data else {}
    initial_context = data.initial_context if data else {}
    
    execution_id = await workflow_engine.start_execution(
        workflow_id=workflow_id,
        trigger_type='manual',
        trigger_data=trigger_data,
        initial_context=initial_context
    )
    
    return ExecutionResponse(
        success=True,
        execution_id=execution_id,
        message="Workflow execution started",
        status=ExecutionStatus.RUNNING
    )


@router.get("/{workflow_id}/executions", response_model=List[Execution])
async def list_workflow_executions(
    workflow_id: str,
    status: Optional[str] = None,
    limit: int = 20,
    current_user: dict = Depends(get_current_user_dep)
):
    """List executions for a workflow."""
    workflow = workflow_storage.get_workflow(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    executions = workflow_storage.list_executions(
        workflow_id=workflow_id,
        status=status,
        limit=limit
    )
    
    return [Execution(**e) for e in executions]


# =============================================================================
# Helper Functions
# =============================================================================

async def _sync_workflow_services(workflow_id: str, enabled: bool):
    """Sync workflow triggers with services."""
    workflow = workflow_storage.get_workflow(workflow_id)
    if not workflow:
        return
    
    nodes = workflow.get('nodes', [])
    
    for node in nodes:
        node_type = node.get('type', '')
        config = node.get('config', {})
        
        if node_type == 'trigger_file_watcher':
            if enabled:
                watch_config = WatchConfig(
                    workflow_id=workflow_id,
                    watch_path=config.get('watch_path', ''),
                    recursive=config.get('recursive', True),
                    include_patterns=config.get('include_patterns', ['*']),
                    exclude_patterns=config.get('exclude_patterns', []),
                    debounce_seconds=config.get('debounce_seconds', 5.0)
                )
                file_watcher_service.add_watch(watch_config)
            else:
                file_watcher_service.remove_watch(workflow_id)
        
        elif node_type == 'trigger_schedule':
            if enabled:
                # Register schedule
                task_id = workflow_storage.create_scheduled_task(
                    workflow_id=workflow_id,
                    schedule_type=config.get('schedule_type', 'daily'),
                    schedule_config=config
                )
                schedule_config = ScheduleConfig(
                    task_id=task_id,
                    workflow_id=workflow_id,
                    schedule_type=config.get('schedule_type', 'daily'),
                    config=config
                )
                scheduler_service.add_schedule(schedule_config)
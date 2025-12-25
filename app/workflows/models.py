# app/workflows/models.py
"""
Pydantic models for workflow system.
Defines node types, triggers, and workflow structures.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


# =============================================================================
# Enums
# =============================================================================

class NodeType(str, Enum):
    """Types of nodes available in workflows."""
    # Triggers
    TRIGGER_FILE_WATCHER = "trigger_file_watcher"
    TRIGGER_SCHEDULE = "trigger_schedule"
    TRIGGER_DATA_CONDITION = "trigger_data_condition"
    TRIGGER_MANUAL = "trigger_manual"
    
    # Processors
    PROCESSOR_EXTRACT_CONTENT = "processor_extract_content"
    PROCESSOR_LLM_ANALYSIS = "processor_llm_analysis"
    PROCESSOR_ANOMALY_DETECTION = "processor_anomaly_detection"
    PROCESSOR_CROSS_REFERENCE = "processor_cross_reference"
    PROCESSOR_SUMMARIZE = "processor_summarize"
    PROCESSOR_TRANSLATE = "processor_translate"
    
    # Actions
    ACTION_GENERATE_REPORT = "action_generate_report"
    ACTION_SEND_EMAIL = "action_send_email"
    ACTION_SAVE_TO_FOLDER = "action_save_to_folder"
    ACTION_UPDATE_DATABASE = "action_update_database"
    ACTION_WEBHOOK = "action_webhook"
    
    # Flow Control
    FLOW_CONDITION = "flow_condition"
    FLOW_LOOP = "flow_loop"
    FLOW_MERGE = "flow_merge"
    FLOW_DELAY = "flow_delay"


class TriggerType(str, Enum):
    """Types of workflow triggers."""
    FILE_WATCHER = "file_watcher"
    SCHEDULE = "schedule"
    DATA_CONDITION = "data_condition"
    MANUAL = "manual"


class ExecutionStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduleType(str, Enum):
    """Types of schedule triggers."""
    CRON = "cron"
    INTERVAL = "interval"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# =============================================================================
# Node Configuration Models
# =============================================================================

class Position(BaseModel):
    """Node position on canvas."""
    x: float = 0
    y: float = 0


class NodeBase(BaseModel):
    """Base configuration for all nodes."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: NodeType
    label: str
    position: Position = Field(default_factory=Position)
    config: Dict[str, Any] = Field(default_factory=dict)


# Trigger Node Configs
class FileWatcherConfig(BaseModel):
    """Configuration for file watcher trigger."""
    watch_path: str
    recursive: bool = True
    include_patterns: List[str] = Field(default_factory=lambda: ["*"])
    exclude_patterns: List[str] = Field(default_factory=list)
    debounce_seconds: int = 5  # Wait for file writes to complete


class ScheduleConfig(BaseModel):
    """Configuration for schedule trigger."""
    schedule_type: ScheduleType
    cron_expression: Optional[str] = None  # For cron type
    interval_seconds: Optional[int] = None  # For interval type
    time: Optional[str] = None  # HH:MM for daily/weekly/monthly
    day_of_week: Optional[str] = None  # For weekly
    day_of_month: Optional[int] = None  # For monthly
    timezone: str = "Europe/Athens"


class DataConditionConfig(BaseModel):
    """Configuration for data condition trigger."""
    source_type: str  # 'file', 'database', 'api'
    source_path: str
    condition_field: str
    condition_operator: str  # 'gt', 'lt', 'eq', 'ne', 'contains', 'changed'
    condition_value: Any
    check_interval_seconds: int = 60


# Processor Node Configs
class ExtractContentConfig(BaseModel):
    """Configuration for content extraction."""
    file_types: List[str] = Field(default_factory=lambda: ["pdf", "docx", "xlsx", "txt", "csv"])
    extract_tables: bool = True
    extract_images: bool = False
    ocr_enabled: bool = True
    language: str = "ell+eng"


class LLMAnalysisConfig(BaseModel):
    """Configuration for LLM analysis."""
    prompt_template: str
    system_prompt: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.3
    output_format: str = "text"  # 'text', 'json', 'markdown'
    language: str = "greek"


class AnomalyDetectionConfig(BaseModel):
    """Configuration for anomaly detection."""
    detection_type: str = "statistical"  # 'statistical', 'pattern', 'llm'
    fields_to_monitor: List[str] = Field(default_factory=list)
    threshold_std: float = 2.0  # Standard deviations for statistical
    use_baseline: bool = True
    baseline_window_days: int = 30


class CrossReferenceConfig(BaseModel):
    """Configuration for cross-referencing."""
    reference_source: str  # 'folder', 'database', 'index'
    reference_path: str
    match_fields: List[str] = Field(default_factory=list)
    match_threshold: float = 0.8  # For fuzzy matching


# Action Node Configs
class GenerateReportConfig(BaseModel):
    """Configuration for report generation."""
    report_template: str = "default"
    output_format: str = "docx"  # 'docx', 'pdf', 'html', 'markdown'
    output_path: str
    filename_template: str = "report_{timestamp}"
    language: str = "greek"
    include_sections: List[str] = Field(default_factory=lambda: ["summary", "details", "recommendations"])


class SendEmailConfig(BaseModel):
    """Configuration for sending emails."""
    recipients: List[str]
    subject_template: str
    body_template: str
    attach_report: bool = False
    priority: str = "normal"  # 'low', 'normal', 'high'


class SaveToFolderConfig(BaseModel):
    """Configuration for saving files."""
    destination_path: str
    filename_template: str = "{original_name}_{timestamp}"
    overwrite: bool = False
    create_subfolders: bool = True


# Flow Control Configs
class ConditionConfig(BaseModel):
    """Configuration for conditional branching."""
    condition_type: str = "expression"  # 'expression', 'contains', 'threshold'
    expression: str  # e.g., "context.anomaly_count > 0"
    true_branch: Optional[str] = None  # Node ID
    false_branch: Optional[str] = None  # Node ID


class LoopConfig(BaseModel):
    """Configuration for loop node."""
    loop_type: str = "foreach"  # 'foreach', 'while', 'count'
    source_field: str  # Field in context to iterate over
    max_iterations: int = 100


class DelayConfig(BaseModel):
    """Configuration for delay node."""
    delay_seconds: int = 60
    delay_until: Optional[str] = None  # Time string "HH:MM"


# =============================================================================
# Edge Model
# =============================================================================

class Edge(BaseModel):
    """Connection between nodes."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str  # Source node ID
    target: str  # Target node ID
    source_handle: Optional[str] = None  # For nodes with multiple outputs
    target_handle: Optional[str] = None
    label: Optional[str] = None


# =============================================================================
# Workflow Models
# =============================================================================

class WorkflowCreate(BaseModel):
    """Request to create a new workflow."""
    name: str
    description: Optional[str] = None
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    is_shared: bool = False


class WorkflowUpdate(BaseModel):
    """Request to update a workflow."""
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Edge]] = None
    is_enabled: Optional[bool] = None
    is_shared: Optional[bool] = None


class Workflow(BaseModel):
    """Complete workflow definition."""
    id: str
    user_id: Optional[int] = None
    name: str
    description: Optional[str] = None
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    triggers: List[Dict[str, Any]] = Field(default_factory=list)
    is_enabled: bool = False
    is_shared: bool = False
    created_at: datetime
    updated_at: datetime
    last_run_at: Optional[datetime] = None


class WorkflowSummary(BaseModel):
    """Summary for listing workflows."""
    id: str
    name: str
    description: Optional[str] = None
    is_enabled: bool
    is_shared: bool
    node_count: int
    last_run_at: Optional[datetime] = None
    updated_at: datetime


# =============================================================================
# Execution Models
# =============================================================================

class ExecutionCreate(BaseModel):
    """Request to start a workflow execution."""
    workflow_id: Optional[str] = None  # Optional because it can come from URL path
    trigger_type: TriggerType = TriggerType.MANUAL
    trigger_data: Dict[str, Any] = Field(default_factory=dict)
    initial_context: Dict[str, Any] = Field(default_factory=dict)


class NodeLog(BaseModel):
    """Log entry for a node execution."""
    id: int
    execution_id: str
    node_id: str
    status: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None


class Execution(BaseModel):
    """Workflow execution state."""
    id: str
    workflow_id: str
    status: ExecutionStatus
    current_node_id: Optional[str] = None
    completed_nodes: List[str] = Field(default_factory=list)
    failed_nodes: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    error_node_id: Optional[str] = None
    trigger_type: Optional[str] = None
    trigger_data: Optional[Dict[str, Any]] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    node_logs: List[NodeLog] = Field(default_factory=list)


# =============================================================================
# User Settings Models
# =============================================================================

class EmailSettingsUpdate(BaseModel):
    """Request to update email settings."""
    email_notifications_enabled: Optional[bool] = None
    notification_email: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: Optional[bool] = None


class NotificationPreferences(BaseModel):
    """User notification preferences."""
    notify_on_workflow_complete: bool = True
    notify_on_workflow_failure: bool = True
    notify_on_anomaly_detected: bool = True
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None


class UserSettings(BaseModel):
    """Complete user settings."""
    user_id: int
    email_notifications_enabled: bool = False
    notification_email: Optional[str] = None
    smtp_configured: bool = False
    notification_preferences: NotificationPreferences = Field(default_factory=NotificationPreferences)


# =============================================================================
# Baseline Models
# =============================================================================

class BaselineStats(BaseModel):
    """Statistical baseline for anomaly detection."""
    mean: float
    std: float
    min: float
    max: float
    count: int
    last_n_values: List[float] = Field(default_factory=list)


class Baseline(BaseModel):
    """Baseline record."""
    id: int
    workflow_id: Optional[str] = None
    baseline_type: str
    baseline_key: str
    statistics: Dict[str, Any]
    sample_count: int
    first_sample_at: Optional[datetime] = None
    last_sample_at: Optional[datetime] = None
    is_active: bool = True
    anomaly_threshold: float = 2.0


# =============================================================================
# Response Models
# =============================================================================

class WorkflowResponse(BaseModel):
    """Standard API response for workflow operations."""
    success: bool
    message: str
    data: Optional[Any] = None


class ExecutionResponse(BaseModel):
    """Response for execution operations."""
    success: bool
    execution_id: Optional[str] = None
    message: str
    status: Optional[ExecutionStatus] = None
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import uuid

class WorkflowStep(BaseModel):
    """A single step in the process."""
    id: str
    name: str
    tool: str  # The name of the tool to execute (e.g., 'read_file')
    # Parameters can be static values or template strings like "{{ input_file }}"
    params: Dict[str, Any] = Field(default_factory=dict)
    # If true, the workflow stops if this step fails
    required: bool = True
    # Key to store the result in the shared context
    output_key: Optional[str] = None

class WorkflowTrigger(BaseModel):
    """What starts the workflow."""
    type: Literal["manual", "schedule", "event"]
    config: Dict[str, Any] # e.g. {"cron": "0 9 * * *"} or {"event": "file_created"}

class WorkflowDefinition(BaseModel):
    """The blueprint."""
    id: str
    name: str
    description: str = ""
    enabled: bool = True
    trigger: WorkflowTrigger
    steps: List[WorkflowStep]

class WorkflowExecution(BaseModel):
    """Record of a run."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    status: Literal["running", "completed", "failed"] = "running"
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    logs: List[str] = Field(default_factory=list)
    # The final state of data
    context: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
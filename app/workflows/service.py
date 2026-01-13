from typing import List, Optional
from app.config import get_config
from app.workflows.schemas import WorkflowDefinition, WorkflowExecution
from app.workflows.engine import WorkflowEngine
from app.workflows.triggers import TriggerService

class WorkflowService:
    def __init__(self):
        self.engine = WorkflowEngine()
        self.trigger_service = TriggerService(self.engine)
        self._workflows: List[WorkflowDefinition] = []
        self.reload()

    def reload(self):
        """Load workflows from config.json."""
        config = get_config()
        wf_data_list = config.get("workflows", [])
        
        self._workflows = []
        for data in wf_data_list:
            try:
                self._workflows.append(WorkflowDefinition(**data))
            except Exception as e:
                print(f"Error loading workflow: {e}")
        
        # Update triggers
        self.trigger_service.load_workflows(self._workflows)
        self.trigger_service.start()

    def get_workflows(self) -> List[WorkflowDefinition]:
        return self._workflows

    def run_workflow(self, workflow_id: str, context: dict = None) -> WorkflowExecution:
        wf = next((w for w in self._workflows if w.id == workflow_id), None)
        if not wf:
            raise ValueError("Workflow not found")
        
        return self.engine.run(wf, context)

# Global Instance
workflow_service = WorkflowService()
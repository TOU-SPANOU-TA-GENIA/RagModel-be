import traceback
from datetime import datetime
from typing import Dict, Any

from app.workflows.schemas import WorkflowDefinition, WorkflowExecution, WorkflowStep
from app.workflows.library import StepExecutor
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class WorkflowEngine:
    """
    Runs a workflow definition against a specific context.
    """
    
    def __init__(self):
        self.step_executor = StepExecutor()

    def run(self, workflow: WorkflowDefinition, initial_context: Dict[str, Any] = None) -> WorkflowExecution:
        context = initial_context or {}
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            context=context
        )
        
        logger.info(f"Starting workflow '{workflow.name}' (Run ID: {execution.run_id})")
        
        try:
            for step in workflow.steps:
                self._run_step(step, context, execution)
            
            execution.status = "completed"
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            execution.status = "failed"
            execution.error = str(e)
            execution.logs.append(f"CRITICAL: {str(e)}")
            
        execution.end_time = datetime.utcnow()
        return execution

    def _run_step(self, step: WorkflowStep, context: Dict[str, Any], execution: WorkflowExecution):
        execution.logs.append(f"Step: {step.name} (Tool: {step.tool})")
        
        try:
            output = self.step_executor.execute(step.tool, step.params, context)
            
            # Store output if requested
            if step.output_key:
                context[step.output_key] = output
                # Also store generic 'last_output' for convenience
                context["last_output"] = output
                
            execution.logs.append("  -> Success")
            
        except Exception as e:
            execution.logs.append(f"  -> Failed: {e}")
            if step.required:
                raise e
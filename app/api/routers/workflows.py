from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.api.schemas import User
# FIX: Import workflow schemas from the workflows domain package
from app.workflows.schemas import WorkflowDefinition, WorkflowExecution
from app.workflows import workflow_service

router = APIRouter(prefix="/workflows", tags=["Workflows"])

@router.get("/", response_model=List[WorkflowDefinition])
async def list_workflows(current_user: User = Depends(get_current_user)):
    """List all available automation workflows."""
    return workflow_service.get_workflows()

@router.post("/{workflow_id}/run", response_model=WorkflowExecution)
async def run_workflow(
    workflow_id: str, 
    context: Dict[str, Any] = {},
    current_user: User = Depends(get_current_user)
):
    """
    Manually trigger a specific workflow.
    """
    try:
        execution = workflow_service.run_workflow(workflow_id, context)
        return execution
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {e}")

@router.post("/reload")
async def reload_workflows(current_user: User = Depends(get_current_user)):
    """Force reload workflows from config.json."""
    if not current_user.username == "admin": # Simple check, can be expanded
        raise HTTPException(status_code=403, detail="Only admins can reload workflows")
    
    workflow_service.reload()
    return {"message": "Workflows reloaded successfully"}
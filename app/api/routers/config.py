from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.config import get_config, update_config
from app.api.deps import get_current_user
from app.db.models import User

router = APIRouter()

@router.get("/")
async def get_system_config(
    current_user: User = Depends(get_current_user)
):
    """
    Returns non-sensitive configuration settings.
    """
    # Return a safe subset of config
    conf = get_config()
    return {
        "server": conf.get("server"),
        "rag": conf.get("rag"),
        "llm": {
            "model_path": conf.get("llm", {}).get("model_path"),
            "model_type": conf.get("llm", {}).get("model_type")
        }
    }
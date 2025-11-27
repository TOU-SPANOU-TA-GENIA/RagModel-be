# app/api/config_routes.py
"""
API endpoints for configuration management.
Allows frontend to read and update configuration values.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.config import config_manager
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/config", tags=["Configuration"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ConfigUpdateRequest(BaseModel):
    """Request to update a single configuration value."""
    category: str
    field: str
    value: Any


class ConfigBatchUpdateRequest(BaseModel):
    """Request to update multiple configuration values."""
    updates: Dict[str, Dict[str, Any]]


class ConfigExportRequest(BaseModel):
    """Request to export configuration."""
    include_defaults: bool = True


class ConfigImportRequest(BaseModel):
    """Request to import configuration."""
    config_json: str


class ConfigFieldMetadata(BaseModel):
    """Metadata about a configuration field."""
    name: str
    category: str
    description: str
    type: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    options: Optional[List[str]] = None
    editable: bool = True
    requires_restart: bool = False
    current_value: Any


class ConfigResponse(BaseModel):
    """Standard response for configuration operations."""
    success: bool
    message: str
    data: Optional[Any] = None


# =============================================================================
# Read Endpoints
# =============================================================================

@router.get("/", response_model=Dict[str, Dict[str, Any]])
async def get_all_config():
    """
    Get all current configuration values.
    
    Returns all configuration categories and their current values.
    """
    return config_manager.get_all_values()


@router.get("/metadata", response_model=Dict[str, List[dict]])
async def get_config_metadata():
    """
    Get metadata for all configuration fields.
    
    Returns field descriptions, types, ranges, and current values.
    Useful for rendering configuration UI.
    """
    return config_manager.get_all_metadata()


@router.get("/{category}", response_model=Dict[str, Any])
async def get_category_config(category: str):
    """
    Get configuration for a specific category.
    
    Args:
        category: One of: llm, embedding, rag, agent, tools, memory, server, paths, documents, response
    """
    values = config_manager.get_all_values()
    
    if category not in values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown configuration category: {category}"
        )
    
    return values[category]


@router.get("/{category}/{field}")
async def get_config_field(category: str, field: str):
    """
    Get a specific configuration value.
    
    Args:
        category: Configuration category
        field: Field name within category
    """
    value = config_manager.get(category, field)
    
    if value is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown configuration: {category}.{field}"
        )
    
    return {"category": category, "field": field, "value": value}


# =============================================================================
# Write Endpoints
# =============================================================================

@router.put("/{category}/{field}", response_model=ConfigResponse)
async def update_config_field(category: str, field: str, request: ConfigUpdateRequest):
    """
    Update a specific configuration value.
    
    Args:
        category: Configuration category
        field: Field name within category
        request: New value
    """
    success = config_manager.set(category, field, request.value)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update {category}.{field}. Check value validity."
        )
    
    logger.info(f"Configuration updated: {category}.{field} = {request.value}")
    
    return ConfigResponse(
        success=True,
        message=f"Updated {category}.{field}",
        data={"category": category, "field": field, "value": request.value}
    )


@router.post("/batch", response_model=ConfigResponse)
async def batch_update_config(request: ConfigBatchUpdateRequest):
    """
    Update multiple configuration values at once.
    
    Args:
        request: Dictionary of {category: {field: value, ...}, ...}
    """
    results = config_manager.update_batch(request.updates)
    
    failures = [key for key, success in results.items() if not success]
    
    if failures:
        return ConfigResponse(
            success=False,
            message=f"Some updates failed: {failures}",
            data=results
        )
    
    logger.info(f"Batch configuration update: {len(results)} fields")
    
    return ConfigResponse(
        success=True,
        message=f"Updated {len(results)} configuration values",
        data=results
    )


# =============================================================================
# Import/Export Endpoints
# =============================================================================

@router.post("/save", response_model=ConfigResponse)
async def save_config(file_path: Optional[str] = None):
    """
    Save current configuration to file.
    
    Args:
        file_path: Optional custom path. Uses default if not specified.
    """
    try:
        saved_path = config_manager.save(file_path)
        logger.info(f"Configuration saved to {saved_path}")
        
        return ConfigResponse(
            success=True,
            message=f"Configuration saved",
            data={"path": saved_path}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save configuration: {str(e)}"
        )


@router.get("/export", response_model=ConfigResponse)
async def export_config():
    """
    Export configuration as JSON string.
    
    Returns the complete configuration as a JSON string that can be
    saved externally or imported later.
    """
    config_json = config_manager.export_config()
    
    return ConfigResponse(
        success=True,
        message="Configuration exported",
        data={"config_json": config_json}
    )


@router.post("/import", response_model=ConfigResponse)
async def import_config(request: ConfigImportRequest):
    """
    Import configuration from JSON string.
    
    Args:
        request: JSON string containing configuration values
    """
    success = config_manager.import_config(request.config_json)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to import configuration. Check JSON format."
        )
    
    logger.info("Configuration imported successfully")
    
    return ConfigResponse(
        success=True,
        message="Configuration imported successfully"
    )


@router.post("/reset", response_model=ConfigResponse)
async def reset_config(category: Optional[str] = None):
    """
    Reset configuration to defaults.
    
    Args:
        category: Specific category to reset, or None for all
    """
    # Re-initialize to defaults
    config_manager._init_defaults()
    config_manager._resolve_paths()
    
    logger.info(f"Configuration reset to defaults")
    
    return ConfigResponse(
        success=True,
        message="Configuration reset to defaults"
    )


# =============================================================================
# Validation Endpoints
# =============================================================================

@router.post("/validate", response_model=ConfigResponse)
async def validate_config_value(request: ConfigUpdateRequest):
    """
    Validate a configuration value without applying it.
    
    Useful for real-time validation in UI.
    """
    is_valid = config_manager._validate_field(
        request.category, 
        request.field, 
        request.value
    )
    
    return ConfigResponse(
        success=is_valid,
        message="Valid" if is_valid else "Invalid value",
        data={
            "category": request.category,
            "field": request.field,
            "value": request.value,
            "is_valid": is_valid
        }
    )


@router.get("/categories")
async def get_categories():
    """
    Get list of all configuration categories.
    """
    return {
        "categories": [
            {"name": "llm", "label": "Language Model", "description": "LLM generation settings"},
            {"name": "embedding", "label": "Embeddings", "description": "Embedding model settings"},
            {"name": "rag", "label": "RAG", "description": "Retrieval settings"},
            {"name": "agent", "label": "Agent", "description": "Agent behavior settings"},
            {"name": "tools", "label": "Tools", "description": "Tool system settings"},
            {"name": "memory", "label": "Memory", "description": "Conversation memory settings"},
            {"name": "server", "label": "Server", "description": "Server settings"},
            {"name": "paths", "label": "Paths", "description": "File system paths"},
            {"name": "documents", "label": "Documents", "description": "Document generation settings"},
            {"name": "response", "label": "Response", "description": "Response processing settings"},
        ]
    }
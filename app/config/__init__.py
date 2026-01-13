from pathlib import Path
from typing import Dict, Any
from .manager import config_manager

# Initialize with defaults (will be re-initialized by main.py)
config = config_manager

def get_config() -> Dict[str, Any]:
    """
    Returns the current configuration as a dictionary.
    Use this when you need standard dict access (.get()) instead of objects.
    """
    return config_manager.as_dict()

def update_config(new_config: Dict[str, Any]):
    """Update settings in memory."""
    # Logic to update _settings from dict would go here
    # For now, we can implement basic merging or restart required logic
    pass

def save_config():
    """Save current memory state to file."""
    config_manager.save()

# Direct accessors (The Fluid API)
LLM = config.llm
AGENT = config.agent
TOOLS = config.tools
PROMPTS = config.prompts
RAG = config.rag
PATHS = config.paths
SERVER = config.server
MODELS = config.models

# Legacy Compatibility
BASE_DIR = Path(PATHS.base_dir)
DATA_DIR = Path(PATHS.data_dir)

__all__ = [
    "config_manager",
    "config",
    "get_config",
    "update_config",
    "save_config",
    "LLM", "AGENT", "TOOLS", "PROMPTS", "RAG", "PATHS", "SERVER", "MODELS"
]
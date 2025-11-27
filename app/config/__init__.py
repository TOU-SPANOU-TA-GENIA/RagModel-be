# app/config/__init__.py
"""
Centralized configuration system.

Usage:
    from app.config import config, LLM, RAG, AGENT, TOOLS, PATHS
    
    # Access settings
    max_tokens = LLM.max_new_tokens
    chunk_size = RAG.chunk_size
    
    # Update settings
    config.set('llm', 'temperature', 0.8)
    
    # Get all for UI
    metadata = config.get_all_metadata()
"""

from pathlib import Path

from app.config.schema import (
    LLMSettings,
    EmbeddingSettings,
    RAGSettings,
    AgentSettings,
    ToolSettings,
    MemorySettings,
    ServerSettings,
    PathSettings,
    DocumentSettings,
    ResponseSettings,
    ConfigCategory,
    ConfigField
)
from app.config.manager import config_manager

# Initialize configuration
_base_dir = Path(__file__).parent.parent.parent
config_manager.initialize(base_dir=str(_base_dir))

# Convenient accessors
config = config_manager

# Direct access to settings groups
LLM = config_manager.llm
EMBEDDING = config_manager.embedding
RAG = config_manager.rag
AGENT = config_manager.agent
TOOLS = config_manager.tools
MEMORY = config_manager.memory
SERVER = config_manager.server
PATHS = config_manager.paths
DOCUMENTS = config_manager.documents
RESPONSE = config_manager.response

# =============================================================================
# Legacy compatibility - old config.py style exports
# =============================================================================

# Paths (for backward compatibility)
BASE_DIR = Path(PATHS.base_dir)
DATA_DIR = Path(PATHS.data_dir)
KNOWLEDGE_DIR = Path(PATHS.knowledge_dir)
INSTRUCTIONS_DIR = Path(PATHS.instructions_dir)
INDEX_DIR = Path(PATHS.index_dir)
OFFLOAD_DIR = Path(PATHS.offload_dir)
OFFLINE_MODELS_DIR = Path(PATHS.offline_models_dir)

# Model names
LLM_MODEL_NAME = LLM.model_name
EMBEDDING_MODEL_NAME = EMBEDDING.model_name

# LLM config dict (legacy format)
LLM_CONFIG = {
    "max_new_tokens": LLM.max_new_tokens,
    "temperature": LLM.temperature,
    "top_p": LLM.top_p,
    "do_sample": LLM.do_sample,
    "repetition_penalty": LLM.repetition_penalty,
}

# RAG config dict (legacy format)
RAG_CONFIG = {
    "top_k": RAG.top_k,
    "chunk_size": RAG.chunk_size,
    "chunk_overlap": RAG.chunk_overlap,
    "min_relevance_score": RAG.min_relevance_score,
}

# Fast LLM config
FAST_LLM_CONFIG = {
    "max_new_tokens": LLM.max_new_tokens,
    "temperature": LLM.temperature,
    "top_p": LLM.top_p,
    "do_sample": LLM.do_sample,
    "repetition_penalty": LLM.repetition_penalty,
    "pad_token_id": 128001,
}

# Agent settings (legacy)
AGENT_MODE = AGENT.mode
AGENT_USE_CACHE = AGENT.use_cache
AGENT_DEBUG_MODE = AGENT.debug_mode

# Storage settings
USE_IN_MEMORY_STORAGE = True
CACHE_MODELS_IN_MEMORY = True
CACHE_EMBEDDINGS = EMBEDDING.cache_enabled

# Tool settings (legacy)
AGENT_ALLOWED_DIRECTORIES = [
    DATA_DIR,
    KNOWLEDGE_DIR,
    INSTRUCTIONS_DIR,
    BASE_DIR / "logs",
    BASE_DIR / "config",
]
AGENT_MAX_FILE_SIZE_MB = TOOLS.max_file_size_mb
AGENT_ALLOWED_EXTENSIONS = set(TOOLS.allowed_extensions)
AGENT_ALLOWED_COMMANDS = TOOLS.allowed_commands
AGENT_SHOW_FILE_CONTENT = TOOLS.show_file_content
AGENT_FILE_CONTENT_FORMAT = TOOLS.file_content_format
AGENT_MAX_CONTENT_DISPLAY_LINES = TOOLS.max_content_display_lines

# Server settings
CORS_ORIGINS = SERVER.cors_origins
LOG_LEVEL = SERVER.log_level
LOG_FORMAT = SERVER.log_format


# =============================================================================
# System instruction loading (legacy)
# =============================================================================

def load_system_instructions() -> str:
    """Load system instructions from files."""
    persona_file = INSTRUCTIONS_DIR / "persona.txt"
    rules_file = INSTRUCTIONS_DIR / "rules.txt"
    
    persona = ""
    rules = ""
    
    if persona_file.exists():
        persona = persona_file.read_text(encoding="utf-8")
    
    if rules_file.exists():
        rules = rules_file.read_text(encoding="utf-8")
    
    knowledge_instruction = """

## IMPORTANT: YOUR KNOWLEDGE BASE

You have access to a knowledge base containing information about:
- Panos (the user)
- System documentation
- Configuration files
- Procedures and guides

When asked about information, you should:
1. Use the context provided in <context> tags
2. Answer based on the knowledge base first
3. If information is not in the knowledge base, say so clearly
"""
    
    combined = f"{persona}\n\n{rules}\n\n{knowledge_instruction}".strip()
    
    return combined if combined else """
Your Name is Panos and you are a random 28 years old dude

## IMPORTANT: YOUR KNOWLEDGE BASE

You have access to a knowledge base with specific information.
When context is provided in <context> tags, use it to answer questions.
"""


SYSTEM_INSTRUCTION = load_system_instructions()


# =============================================================================
# LLMConfig dataclass (legacy compatibility)
# =============================================================================

from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    """Configuration for LLM models (legacy compatibility)."""
    model_name: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"
    quantization: Optional[str] = "4bit"
    
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "device": self.device,
            "quantization": self.quantization
        }


__all__ = [
    # New config system
    "config",
    "config_manager",
    "LLM",
    "EMBEDDING", 
    "RAG",
    "AGENT",
    "TOOLS",
    "MEMORY",
    "SERVER",
    "PATHS",
    "DOCUMENTS",
    "RESPONSE",
    
    # Schema classes
    "LLMSettings",
    "EmbeddingSettings",
    "RAGSettings",
    "AgentSettings",
    "ToolSettings",
    "MemorySettings",
    "ServerSettings",
    "PathSettings",
    "DocumentSettings",
    "ResponseSettings",
    "ConfigCategory",
    "ConfigField",
    
    # Legacy exports
    "BASE_DIR",
    "DATA_DIR",
    "KNOWLEDGE_DIR",
    "INSTRUCTIONS_DIR",
    "INDEX_DIR",
    "LLM_MODEL_NAME",
    "EMBEDDING_MODEL_NAME",
    "LLM_CONFIG",
    "RAG_CONFIG",
    "FAST_LLM_CONFIG",
    "AGENT_MODE",
    "AGENT_USE_CACHE",
    "AGENT_DEBUG_MODE",
    "CORS_ORIGINS",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "SYSTEM_INSTRUCTION",
    "LLMConfig",
    "load_system_instructions",
]
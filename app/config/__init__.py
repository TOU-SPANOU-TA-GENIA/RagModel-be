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
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

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
config_manager.initialize(
    config_file="config.json",  # ← ADD THIS!
    base_dir=str(_base_dir)
)

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

# Tool settings (legacy) - UPDATED: Removed KNOWLEDGE_DIR and INSTRUCTIONS_DIR
AGENT_ALLOWED_DIRECTORIES = [
    DATA_DIR,
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
# System instruction loading (UPDATED - no longer uses instruction files)
# =============================================================================

def load_system_instructions() -> str:
    """Load system instructions from configuration."""
    # Try to load from config first (if instructions module exists)
    try:
        from app.config.instructions import InstructionsSettings
        instructions_data = config.get_section('instructions')
        if instructions_data:
            instructions = InstructionsSettings.from_dict(instructions_data)
            return instructions.to_system_prompt()
    except (ImportError, Exception) as e:
        logger.warning(f"Could not load instructions from config: {e}")
    
    # Fallback to default instructions
    return """
You are a helpful AI assistant.

## Core Principles

**Directness:** Answer first, explain after. Don't deflect with questions unless truly ambiguous.

**Context Awareness:** Use remembered information about the user when relevant. Don't force personal context on factual questions.

**Instruction Following:** When users set rules ("when I say X, respond Y", "always be brief"), follow them precisely and consistently.

**Natural Conversation:** Avoid repetition. Match the user's formality and style. Focus on being helpful over being social.

## Response Pattern

1. Direct answer
2. Brief explanation if needed
3. Follow-up only if genuinely relevant

Don't narrate your reasoning process. Don't include meta-commentary. Just respond naturally.

## Network Knowledge Base

You have access to documents from network shares. When provided context in <context> tags, use it to answer questions accurately.

When asked about information, you should:
1. Use the context provided in <context> tags
2. Answer based on the knowledge base first
3. If information is not in the knowledge base, say so clearly

## Tool Handling - File Operations

**What Happens:**
- Tool auto-selects best file when multiple matches exist (prefers network share)
- You receive complete file content with metadata

**Your Response Pattern:**
```
I read [filename] from [location]:

[content or answer based on content]

[Optional: Note about other versions if relevant]
```

**Critical Don'ts:**
- Don't ask "which file?" when content is provided
- Don't repeat file selection questions
- Don't ignore successfully retrieved content
- Don't truncate content unless specifically asked

**Example:**
"I read test.txt from network share. It contains: [content]. Note: Also found versions in other folders."

## Edge Cases

**Ambiguous:** Ask for clarification only when genuinely needed.
**Missing info:** Say you don't know rather than guess (check knowledge base first).
**Conflicts:** Use most recent information or acknowledge the conflict.
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
    
    # Legacy exports (UPDATED - removed KNOWLEDGE_DIR and INSTRUCTIONS_DIR)
    "BASE_DIR",
    "DATA_DIR",
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
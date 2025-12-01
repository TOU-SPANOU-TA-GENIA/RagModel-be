# app/config/schema.py
"""
Configuration schema definitions.
All configurable values are defined here with types, defaults, and validation.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum
import json


class ConfigCategory(Enum):
    """Categories for grouping configurations in UI."""
    LLM = "llm"
    RAG = "rag"
    AGENT = "agent"
    TOOLS = "tools"
    SERVER = "server"
    STORAGE = "storage"
    LOGGING = "logging"


@dataclass
class ConfigField:
    """Metadata for a configuration field."""
    name: str
    category: ConfigCategory
    description: str
    field_type: str  # "int", "float", "str", "bool", "list", "path"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[List[str]] = None  # For dropdown selections
    editable: bool = True  # Can be changed via UI
    requires_restart: bool = False  # Needs server restart to apply


# =============================================================================
# LLM Configuration
# =============================================================================

@dataclass
class LLMSettings:
    """Language model settings."""
    model_name: str = "./offline_models/qwen3-4b"  # ← New default
    max_new_tokens: int = 512  # ← New default
    trust_remote_code: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    quantization: str = "4bit"
    device: str = "auto"
    prewarm_enabled: bool = True
    prewarm_prompts: List[str] = field(default_factory=lambda: ["Hello", "Hi", "Hey"])
    prewarm_delay: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("model_name", ConfigCategory.LLM,
                       "HuggingFace model path", "str", "./offline_models/qwen3-4b",
                       requires_restart=True),
            ConfigField("max_new_tokens", ConfigCategory.LLM,
                       "Maximum tokens to generate", "int", 2048, 1, 4096),
            ConfigField("temperature", ConfigCategory.LLM,
                       "Sampling temperature (higher = more random)", "float", 0.7, 0.0, 2.0),
            ConfigField("top_p", ConfigCategory.LLM,
                       "Nucleus sampling threshold", "float", 0.9, 0.0, 1.0),
            ConfigField("repetition_penalty", ConfigCategory.LLM,
                       "Penalty for repeating tokens", "float", 1.1, 1.0, 2.0),
        ]


# =============================================================================
# Embedding Configuration
# =============================================================================

@dataclass
class EmbeddingSettings:
    """Embedding model settings."""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384
    normalize: bool = True
    batch_size: int = 32
    cache_enabled: bool = True
    max_cache_size: int = 10000
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("model_name", ConfigCategory.LLM,
                       "Embedding model name", "str",
                       "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                       requires_restart=True),
            ConfigField("batch_size", ConfigCategory.LLM,
                       "Batch size for embeddings", "int", 32, 1, 128),
        ]


# =============================================================================
# RAG Configuration
# =============================================================================

@dataclass
class RAGSettings:
    """RAG system settings."""
    top_k: int = 3
    min_relevance_score: float = 0.2
    chunk_size: int = 500
    chunk_overlap: int = 50
    use_faiss: bool = False
    rerank_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("top_k", ConfigCategory.RAG,
                       "Number of documents to retrieve", "int", 3, 1, 20),
            ConfigField("min_relevance_score", ConfigCategory.RAG,
                       "Minimum relevance score threshold", "float", 0.2, 0.0, 1.0),
            ConfigField("chunk_size", ConfigCategory.RAG,
                       "Document chunk size (characters)", "int", 500, 100, 2000),
            ConfigField("chunk_overlap", ConfigCategory.RAG,
                       "Overlap between chunks", "int", 50, 0, 500),
        ]


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass
class AgentSettings:
    """Agent behavior settings."""
    mode: str = "production"
    debug_mode: bool = False
    use_cache: bool = False
    max_tool_retries: int = 2
    tool_timeout: float = 30.0
    action_keyword_weight: float = 1.5
    question_keyword_weight: float = 1.0
    conversation_keyword_weight: float = 0.5
    min_confidence_threshold: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("mode", ConfigCategory.AGENT,
                       "Agent mode", "str", "production",
                       options=["basic", "enhanced", "production"]),
            ConfigField("debug_mode", ConfigCategory.AGENT,
                       "Enable debug output", "bool", False),
            ConfigField("max_tool_retries", ConfigCategory.AGENT,
                       "Maximum tool execution retries", "int", 2, 0, 5),
        ]


# =============================================================================
# Tool Configuration
# =============================================================================

@dataclass
class ToolSettings:
    """Tool system settings."""
    max_file_size_mb: int = 10
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.txt', '.md', '.text', '.json', '.yaml', '.yml',
        '.conf', '.config', '.cfg', '.log', '.py', '.sh',
        '.bash', '.xml', '.html', '.csv'
    ])
    allowed_commands: List[str] = field(default_factory=lambda: [
        'ls', 'pwd', 'echo', 'date', 'whoami', 'df', 'free'
    ])
    show_file_content: bool = True
    file_content_format: str = "pretty"
    max_content_display_lines: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("max_file_size_mb", ConfigCategory.TOOLS,
                       "Maximum file size for reading (MB)", "int", 10, 1, 100),
        ]


# =============================================================================
# Memory Configuration
# =============================================================================

@dataclass
class MemorySettings:
    """Conversation memory settings."""
    max_sessions: int = 100
    session_timeout: int = 3600
    max_history_messages: int = 10
    store_instructions: bool = True
    instruction_timeout: int = 3600
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("max_sessions", ConfigCategory.STORAGE,
                       "Maximum concurrent sessions", "int", 100, 10, 1000),
        ]


# =============================================================================
# Server Configuration
# =============================================================================

@dataclass
class ServerSettings:
    """Server and API settings."""
    host: str = "localhost"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:4200",
        "http://127.0.0.1:4200"
    ])
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("host", ConfigCategory.SERVER,
                       "Server host address", "str", "localhost", requires_restart=True),
            ConfigField("port", ConfigCategory.SERVER,
                       "Server port", "int", 8000, 1024, 65535, requires_restart=True),
        ]


# =============================================================================
# Storage Paths Configuration
# =============================================================================

@dataclass
class PathSettings:
    """File system path settings."""
    base_dir: str = ""
    data_dir: str = "data"
    index_dir: str = "faiss_index"
    outputs_dir: str = "outputs"
    offload_dir: str = "offload"
    offline_models_dir: str = "offline_models"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Document Generation Configuration
# =============================================================================

@dataclass
class DocumentSettings:
    """Document generation settings."""
    default_font: str = "Arial"
    default_font_size: int = 11
    title_font_size: int = 24
    heading1_font_size: int = 18
    heading2_font_size: int = 14
    max_filename_length: int = 200
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Response Processing Configuration
# =============================================================================

@dataclass
class ResponseSettings:
    """Response processing settings."""
    clean_xml_tags: bool = True
    clean_meta_commentary: bool = True
    clean_reasoning_markers: bool = True
    normalize_whitespace: bool = True
    max_response_length: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Network Filesystem Configuration
# =============================================================================

@dataclass
class ShareConfig:
    """Configuration for a single network share."""
    name: str
    mount_path: str
    share_type: str = "smb"
    enabled: bool = True
    auto_index: bool = True
    watch_for_changes: bool = True
    scan_interval: int = 300
    include_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".pdf", ".doc", ".docx",
        ".xls", ".xlsx", ".csv", ".json", ".yaml",
        ".ppt", ".pptx", ".log", ".xml", ".html"
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        ".*", "~$*", "Thumbs.db", "desktop.ini",
        "*.tmp", "*.temp", "$RECYCLE.BIN"
    ])
    max_file_size_mb: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkFilesystemSettings:
    """Network filesystem monitoring settings."""
    enabled: bool = False
    auto_start_monitoring: bool = True
    shares: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkFilesystemSettings':
        """Create from dictionary (used when loading from JSON)."""
        return cls(
            enabled=data.get('enabled', False),
            auto_start_monitoring=data.get('auto_start_monitoring', True),
            shares=data.get('shares', [])
        )
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("enabled", ConfigCategory.STORAGE,
                       "Enable network filesystem monitoring", "bool", False, 
                       requires_restart=True),
            ConfigField("auto_start_monitoring", ConfigCategory.STORAGE,
                       "Auto-start monitoring on startup", "bool", True,
                       requires_restart=True),
        ]
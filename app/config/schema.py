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
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    quantization: str = "4bit"  # "none", "4bit", "8bit"
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Pre-warming settings
    prewarm_enabled: bool = True
    prewarm_prompts: List[str] = field(default_factory=lambda: ["Hello", "Hi", "Hey"])
    prewarm_delay: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("model_name", ConfigCategory.LLM, 
                       "HuggingFace model name or local path", "str",
                       "meta-llama/Llama-3.2-3B-Instruct", requires_restart=True),
            ConfigField("max_new_tokens", ConfigCategory.LLM,
                       "Maximum tokens to generate", "int", 2048, 64, 8192),
            ConfigField("temperature", ConfigCategory.LLM,
                       "Randomness in generation (0=deterministic, 1=creative)", "float", 0.7, 0.0, 2.0),
            ConfigField("top_p", ConfigCategory.LLM,
                       "Nucleus sampling threshold", "float", 0.9, 0.0, 1.0),
            ConfigField("repetition_penalty", ConfigCategory.LLM,
                       "Penalty for repeating tokens", "float", 1.1, 1.0, 2.0),
            ConfigField("quantization", ConfigCategory.LLM,
                       "Model quantization level", "str", "4bit",
                       options=["none", "4bit", "8bit"], requires_restart=True),
            ConfigField("device", ConfigCategory.LLM,
                       "Compute device", "str", "auto",
                       options=["auto", "cuda", "cpu"], requires_restart=True),
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
            ConfigField("model_name", ConfigCategory.RAG,
                       "Embedding model name", "str",
                       "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                       requires_restart=True),
            ConfigField("dimension", ConfigCategory.RAG,
                       "Embedding vector dimension", "int", 384, 64, 4096,
                       editable=False),
            ConfigField("batch_size", ConfigCategory.RAG,
                       "Batch size for embedding generation", "int", 32, 1, 256),
            ConfigField("cache_enabled", ConfigCategory.RAG,
                       "Enable embedding cache", "bool", True),
            ConfigField("max_cache_size", ConfigCategory.RAG,
                       "Maximum cached embeddings", "int", 10000, 100, 100000),
        ]


# =============================================================================
# RAG Configuration
# =============================================================================

@dataclass
class RAGSettings:
    """Retrieval-Augmented Generation settings."""
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
                       "Minimum similarity score for retrieval", "float", 0.2, 0.0, 1.0),
            ConfigField("chunk_size", ConfigCategory.RAG,
                       "Document chunk size in characters", "int", 500, 100, 2000),
            ConfigField("chunk_overlap", ConfigCategory.RAG,
                       "Overlap between chunks", "int", 50, 0, 500),
            ConfigField("use_faiss", ConfigCategory.RAG,
                       "Use FAISS for vector storage", "bool", False, requires_restart=True),
        ]


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass
class AgentSettings:
    """Agent behavior settings."""
    mode: str = "production"  # "production", "development", "military"
    debug_mode: bool = False
    use_cache: bool = False
    max_tool_retries: int = 2
    tool_timeout: float = 30.0
    
    # Intent classification
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
                       "Agent operation mode", "str", "production",
                       options=["production", "development", "military"]),
            ConfigField("debug_mode", ConfigCategory.AGENT,
                       "Enable debug output", "bool", False),
            ConfigField("max_tool_retries", ConfigCategory.AGENT,
                       "Max retries for failed tools", "int", 2, 0, 5),
            ConfigField("tool_timeout", ConfigCategory.AGENT,
                       "Tool execution timeout (seconds)", "float", 30.0, 5.0, 120.0),
            ConfigField("min_confidence_threshold", ConfigCategory.AGENT,
                       "Minimum confidence for intent classification", "float", 0.5, 0.0, 1.0),
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
    file_content_format: str = "pretty"  # "pretty", "raw", "minimal"
    max_content_display_lines: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("max_file_size_mb", ConfigCategory.TOOLS,
                       "Maximum file size for reading (MB)", "int", 10, 1, 100),
            ConfigField("allowed_extensions", ConfigCategory.TOOLS,
                       "Allowed file extensions", "list", 
                       ['.txt', '.md', '.json', '.yaml']),
            ConfigField("allowed_commands", ConfigCategory.TOOLS,
                       "Allowed system commands", "list",
                       ['ls', 'pwd', 'echo', 'date']),
            ConfigField("file_content_format", ConfigCategory.TOOLS,
                       "How to display file content", "str", "pretty",
                       options=["pretty", "raw", "minimal"]),
        ]


# =============================================================================
# Conversation Memory Configuration
# =============================================================================

@dataclass
class MemorySettings:
    """Conversation memory settings."""
    max_sessions: int = 100
    session_timeout: int = 3600  # seconds
    max_history_messages: int = 10
    store_instructions: bool = True
    instruction_timeout: int = 3600  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("max_sessions", ConfigCategory.STORAGE,
                       "Maximum concurrent sessions", "int", 100, 10, 1000),
            ConfigField("session_timeout", ConfigCategory.STORAGE,
                       "Session timeout (seconds)", "int", 3600, 300, 86400),
            ConfigField("max_history_messages", ConfigCategory.STORAGE,
                       "Messages to include in context", "int", 10, 1, 50),
            ConfigField("instruction_timeout", ConfigCategory.STORAGE,
                       "User instruction timeout (seconds)", "int", 3600, 300, 86400),
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
            ConfigField("cors_origins", ConfigCategory.SERVER,
                       "Allowed CORS origins", "list", ["http://localhost:4200"]),
            ConfigField("log_level", ConfigCategory.SERVER,
                       "Logging level", "str", "INFO",
                       options=["DEBUG", "INFO", "WARNING", "ERROR"]),
        ]


# =============================================================================
# Storage Paths Configuration
# =============================================================================

@dataclass
class PathSettings:
    """File system path settings."""
    base_dir: str = ""  # Set at runtime
    data_dir: str = "data"
    knowledge_dir: str = "data/knowledge"
    instructions_dir: str = "data/instructions"
    index_dir: str = "faiss_index"
    outputs_dir: str = "outputs"
    offload_dir: str = "offload"
    offline_models_dir: str = "offline_models"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("knowledge_dir", ConfigCategory.STORAGE,
                       "Knowledge base directory", "path", "data/knowledge"),
            ConfigField("instructions_dir", ConfigCategory.STORAGE,
                       "System instructions directory", "path", "data/instructions"),
            ConfigField("outputs_dir", ConfigCategory.STORAGE,
                       "Generated files output directory", "path", "outputs"),
        ]


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
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("default_font", ConfigCategory.TOOLS,
                       "Default document font", "str", "Arial"),
            ConfigField("default_font_size", ConfigCategory.TOOLS,
                       "Default font size", "int", 11, 8, 24),
            ConfigField("title_font_size", ConfigCategory.TOOLS,
                       "Title font size", "int", 24, 14, 48),
        ]


# =============================================================================
# Response Cleaning Configuration
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
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("clean_xml_tags", ConfigCategory.LLM,
                       "Remove XML tags from responses", "bool", True),
            ConfigField("clean_meta_commentary", ConfigCategory.LLM,
                       "Remove meta-commentary from responses", "bool", True),
            ConfigField("max_response_length", ConfigCategory.LLM,
                       "Maximum response length (None=unlimited)", "int", None, 100, 10000),
        ]
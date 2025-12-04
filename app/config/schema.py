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
    model_name: str = "./offline_models/qwen3-4b"
    max_new_tokens: int = 512
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
                       "Maximum tokens to generate", "int", 512, 1, 2048),
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
    clean_thinking_tags: bool = True
    clean_xml_tags: bool = True
    clean_meta_commentary: bool = True
    normalize_whitespace: bool = True
    max_response_length: Optional[int] = None
    language: str = "greek"
    
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


# =============================================================================
# Streaming Configuration
# =============================================================================

@dataclass
class StreamingSettings:
    """Streaming generation settings."""
    enabled: bool = True
    token_timeout: int = 60
    skip_prompt: bool = True
    skip_special_tokens: bool = True
    chunk_delay_ms: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("enabled", ConfigCategory.LLM,
                       "Enable streaming responses", "bool", True),
            ConfigField("token_timeout", ConfigCategory.LLM,
                       "Timeout per token (seconds)", "int", 60, 10, 300),
            ConfigField("chunk_delay_ms", ConfigCategory.LLM,
                       "Delay between chunks (ms)", "int", 20, 0, 100),
        ]


# =============================================================================
# Model Registry Configuration
# =============================================================================

@dataclass
class ModelsSettings:
    """Model registry settings."""
    active: str = "qwen3-4b"
    registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default model if empty."""
        if not self.registry:
            self.registry = {
                "qwen3-4b": {
                    "family": "qwen",
                    "path": "./offline_models/qwen3-4b",
                    "thinking_tags": {
                        "start": ["<think>", "<thinking>", "<σκέψη>"],
                        "end": ["</think>", "</thinking>", "</σκέψη>"]
                    },
                    "response_tags": {
                        "start": ["<response>", "<απάντηση>"],
                        "end": ["</response>", "</απάντηση>"]
                    },
                    "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
                    "prompt_template": "qwen_chat",
                    "supports_thinking": True,
                    "trust_remote_code": True
                }
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "registry": self.registry
        }
    
    def get_active_model_config(self) -> Optional[Dict[str, Any]]:
        """Get configuration for active model."""
        return self.registry.get(self.active)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("active", ConfigCategory.LLM,
                       "Active model ID", "str", "qwen3-4b",
                       requires_restart=True),
        ]


# =============================================================================
# Prompt Templates Configuration
# =============================================================================

@dataclass
class PromptTemplatesSettings:
    """Prompt template registry settings."""
    templates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default templates."""
        if not self.templates:
            self.templates = {
                "qwen_chat": {
                    "system_format": "<|im_start|>system\n{system}<|im_end|>\n",
                    "user_format": "<|im_start|>user\n{user}<|im_end|>\n",
                    "assistant_format": "<|im_start|>assistant\n",
                    "supports_multi_turn": True
                },
                "llama3_instruct": {
                    "system_format": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
                    "user_format": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
                    "assistant_format": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                    "supports_multi_turn": True
                },
                "default": {
                    "system_format": "System: {system}\n\n",
                    "user_format": "User: {user}\n\n",
                    "assistant_format": "Assistant: ",
                    "supports_multi_turn": True
                }
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {"templates": self.templates}
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get template by name."""
        return self.templates.get(name, self.templates.get("default"))


# =============================================================================
# Response Cleaning Configuration
# =============================================================================

@dataclass
class ResponseCleaningSettings:
    """Response cleaning settings."""
    enabled: bool = True
    clean_thinking: bool = True
    clean_xml_tags: bool = True
    clean_meta_commentary: bool = True
    normalize_whitespace: bool = True
    custom_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("enabled", ConfigCategory.LLM,
                       "Enable response cleaning", "bool", True),
            ConfigField("clean_thinking", ConfigCategory.LLM,
                       "Remove thinking blocks", "bool", True),
            ConfigField("clean_xml_tags", ConfigCategory.LLM,
                       "Remove XML/HTML tags", "bool", True),
            ConfigField("clean_meta_commentary", ConfigCategory.LLM,
                       "Remove meta-commentary", "bool", True),
        ]


# =============================================================================
# Localization Configuration
# =============================================================================

@dataclass
class LocalizationSettings:
    """Localization settings."""
    default_language: str = "greek"
    force_greek_responses: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
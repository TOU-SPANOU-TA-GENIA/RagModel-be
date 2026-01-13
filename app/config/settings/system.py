from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from .base import ConfigField, ConfigCategory

@dataclass
class ServerSettings:
    """FastAPI server settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PathSettings:
    """Application file paths."""
    base_dir: str = ""
    data_dir: str = "data"
    index_dir: str = "faiss_index"
    outputs_dir: str = "outputs"
    offline_models_dir: str = "offline_models"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class LocalizationSettings:
    """Language settings."""
    default_language: str = "greek"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
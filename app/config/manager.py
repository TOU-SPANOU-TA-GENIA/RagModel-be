# app/config/manager.py
"""
Configuration manager - handles loading, saving, validation, and access.
Supports runtime updates and persistence.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar
from dataclasses import asdict
from threading import Lock
import copy

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
    NetworkFilesystemSettings,
    StreamingSettings,
    ModelsSettings,
    PromptTemplatesSettings,
    ResponseCleaningSettings,
    LocalizationSettings,
    ConfigCategory,
    ConfigField
)

T = TypeVar('T')


class ConfigurationManager:
    """
    Centralized configuration manager.
    - Loads from file or uses defaults
    - Validates changes
    - Persists updates
    - Thread-safe access
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._config_file: Optional[Path] = None
        self._settings: Dict[str, Any] = {}
        self._change_callbacks: Dict[str, list] = {}
        self._lock = Lock()
        
        # Initialize all settings with defaults
        self._init_defaults()
        self._initialized = True
    
    def _init_defaults(self):
        """Initialize all settings with default values."""
        self._settings = {
            'llm': LLMSettings(),
            'embedding': EmbeddingSettings(),
            'rag': RAGSettings(),
            'agent': AgentSettings(),
            'tools': ToolSettings(),
            'memory': MemorySettings(),
            'server': ServerSettings(),
            'paths': PathSettings(),
            'documents': DocumentSettings(),
            'response': ResponseSettings(),
            'network_filesystem': NetworkFilesystemSettings(),
            'streaming': StreamingSettings(),
            'models': ModelsSettings(),
            'prompt_templates': PromptTemplatesSettings(),
            'response_cleaning': ResponseCleaningSettings(),
            'localization': LocalizationSettings(),
        }
    
    def initialize(self, config_file: Optional[str] = None, base_dir: Optional[str] = None):
        """
        Initialize configuration from file or defaults.
        
        Args:
            config_file: Path to JSON config file
            base_dir: Base directory for relative paths
        """
        with self._lock:
            # Set base directory
            if base_dir:
                self._settings['paths'].base_dir = base_dir
            else:
                self._settings['paths'].base_dir = str(Path(__file__).parent.parent.parent)
            
            # Resolve all paths
            self._resolve_paths()
            
            # Load from file if provided
            if config_file:
                self._config_file = Path(config_file)
                if self._config_file.exists():
                    self._load_from_file()
    
    def _resolve_paths(self):
        """Resolve relative paths to absolute paths."""
        base = Path(self._settings['paths'].base_dir)
        paths = self._settings['paths']
        
        # Create absolute paths
        paths.data_dir = str(base / paths.data_dir)
        paths.index_dir = str(base / paths.index_dir)
        paths.outputs_dir = str(base / paths.outputs_dir)
        paths.offload_dir = str(base / paths.offload_dir)
        paths.offline_models_dir = str(base / paths.offline_models_dir)
        
        # Create directories if they don't exist
        for dir_path in [paths.outputs_dir, paths.offload_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _load_from_file(self):
        """Load configuration from JSON file."""
        try:
            with open(self._config_file, 'r') as f:
                data = json.load(f)
            
            # Update each settings group
            for key, settings_obj in self._settings.items():
                if key in data:
                    for field, value in data[key].items():
                        if hasattr(settings_obj, field):
                            setattr(settings_obj, field, value)
            
            print(f"Configuration loaded from {self._config_file}")
        except Exception as e:
            print(f"Failed to load config file: {e}, using defaults")
    
    def save(self, config_file: Optional[str] = None):
        """Save current configuration to file."""
        file_path = Path(config_file) if config_file else self._config_file
        
        if not file_path:
            file_path = Path(self._settings['paths'].base_dir) / "config.json"
        
        with self._lock:
            data = {}
            for key, settings_obj in self._settings.items():
                data[key] = settings_obj.to_dict()
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self._config_file = file_path
        
        return str(file_path)
    
    # =========================================================================
    # Getters for each settings group
    # =========================================================================
    
    @property
    def llm(self) -> LLMSettings:
        return self._settings['llm']
    
    @property
    def embedding(self) -> EmbeddingSettings:
        return self._settings['embedding']
    
    @property
    def rag(self) -> RAGSettings:
        return self._settings['rag']
    
    @property
    def agent(self) -> AgentSettings:
        return self._settings['agent']
    
    @property
    def tools(self) -> ToolSettings:
        return self._settings['tools']
    
    @property
    def memory(self) -> MemorySettings:
        return self._settings['memory']
    
    @property
    def server(self) -> ServerSettings:
        return self._settings['server']
    
    @property
    def paths(self) -> PathSettings:
        return self._settings['paths']
    
    @property
    def documents(self) -> DocumentSettings:
        return self._settings['documents']
    
    @property
    def response(self) -> ResponseSettings:
        return self._settings['response']
    
    @property
    def network_filesystem(self) -> NetworkFilesystemSettings:
        return self._settings['network_filesystem']
    
    @property
    def streaming(self) -> StreamingSettings:
        return self._settings['streaming']
    
    @property
    def models(self) -> ModelsSettings:
        return self._settings['models']
    
    @property
    def prompt_templates(self) -> PromptTemplatesSettings:
        return self._settings['prompt_templates']
    
    @property
    def response_cleaning(self) -> ResponseCleaningSettings:
        return self._settings['response_cleaning']
    
    @property
    def localization(self) -> LocalizationSettings:
        return self._settings['localization']
    
    # =========================================================================
    # Dynamic access and updates
    # =========================================================================
    
    def get(self, category: str, field: str) -> Any:
        """Get a specific configuration value."""
        with self._lock:
            settings = self._settings.get(category)
            if settings and hasattr(settings, field):
                return getattr(settings, field)
            return None
    
    def get_section(self, category: str) -> Optional[Dict[str, Any]]:
        """Get an entire configuration section as dictionary."""
        with self._lock:
            settings = self._settings.get(category)
            if settings:
                return settings.to_dict()
            return None
    
    def set(self, category: str, field: str, value: Any) -> bool:
        """
        Set a configuration value.
        
        Args:
            category: Settings category (llm, rag, etc.)
            field: Field name within category
            value: New value
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            settings = self._settings.get(category)
            if not settings:
                return False
            
            if not hasattr(settings, field):
                return False
            
            # Validate
            if not self._validate_field(category, field, value):
                return False
            
            # Store old value for callbacks
            old_value = getattr(settings, field)
            
            # Update
            setattr(settings, field, value)
            
            # Trigger callbacks
            self._trigger_callbacks(category, field, old_value, value)
            
            return True
    
    def _validate_field(self, category: str, field: str, value: Any) -> bool:
        """Validate a field value."""
        settings = self._settings.get(category)
        if not settings:
            return False
        
        # Get metadata for validation
        metadata_method = getattr(settings.__class__, 'get_field_metadata', None)
        if not metadata_method:
            return True
        
        for field_meta in metadata_method():
            if field_meta.name == field:
                # Check type
                if field_meta.field_type == "int" and not isinstance(value, int):
                    return False
                if field_meta.field_type == "float" and not isinstance(value, (int, float)):
                    return False
                if field_meta.field_type == "bool" and not isinstance(value, bool):
                    return False
                
                # Check range
                if field_meta.min_value is not None and value < field_meta.min_value:
                    return False
                if field_meta.max_value is not None and value > field_meta.max_value:
                    return False
                
                # Check options
                if field_meta.options and value not in field_meta.options:
                    return False
                
                return True
        
        return True
    
    def update_batch(self, updates: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """
        Update multiple settings at once.
        
        Args:
            updates: Dict of {category: {field: value, ...}, ...}
            
        Returns:
            Dict of {category.field: success_bool}
        """
        results = {}
        
        for category, fields in updates.items():
            for field, value in fields.items():
                key = f"{category}.{field}"
                results[key] = self.set(category, field, value)
        
        return results
    
    # =========================================================================
    # Change callbacks
    # =========================================================================
    
    def on_change(self, category: str, field: str, callback):
        """Register a callback for when a field changes."""
        key = f"{category}.{field}"
        if key not in self._change_callbacks:
            self._change_callbacks[key] = []
        self._change_callbacks[key].append(callback)
    
    def _trigger_callbacks(self, category: str, field: str, old_value: Any, new_value: Any):
        """Trigger registered callbacks for a changed field."""
        key = f"{category}.{field}"
        callbacks = self._change_callbacks.get(key, [])
        
        for callback in callbacks:
            try:
                callback(old_value, new_value)
            except Exception as e:
                print(f"Callback error for {key}: {e}")
    
    # =========================================================================
    # API for frontend
    # =========================================================================
    
    def get_all_metadata(self) -> Dict[str, list]:
        """Get all configuration metadata for UI rendering."""
        metadata = {}
        
        for category, settings in self._settings.items():
            metadata_method = getattr(settings.__class__, 'get_field_metadata', None)
            if metadata_method:
                fields = metadata_method()
                metadata[category] = [
                    {
                        'name': f.name,
                        'category': f.category.value,
                        'description': f.description,
                        'type': f.field_type,
                        'default': f.default,
                        'min': f.min_value,
                        'max': f.max_value,
                        'options': f.options,
                        'editable': f.editable,
                        'requires_restart': f.requires_restart,
                        'current_value': getattr(settings, f.name)
                    }
                    for f in fields
                ]
        
        return metadata
    
    def get_all_values(self) -> Dict[str, Dict[str, Any]]:
        """Get all current configuration values."""
        values = {}
        for category, settings in self._settings.items():
            values[category] = settings.to_dict()
        return values
    
    def export_config(self) -> str:
        """Export configuration as JSON string."""
        return json.dumps(self.get_all_values(), indent=2, default=str)
    
    def import_config(self, json_str: str) -> bool:
        """Import configuration from JSON string."""
        try:
            data = json.loads(json_str)
            results = self.update_batch(data)
            return all(results.values())
        except Exception as e:
            print(f"Import failed: {e}")
            return False


# Global configuration manager instance
config_manager = ConfigurationManager()
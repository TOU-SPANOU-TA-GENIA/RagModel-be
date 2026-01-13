import json
from pathlib import Path
from threading import Lock
from typing import Dict, Any, Optional

from app.config.settings import (
    LLMSettings, EmbeddingSettings, StreamingSettings, ModelsSettings,
    AgentSettings, ToolSettings, PromptTemplatesSettings,
    RAGSettings, NetworkFilesystemSettings, DocumentSettings,
    ServerSettings, PathSettings, LocalizationSettings
)

class ConfigurationManager:
    """
    Centralized configuration manager.
    Singleton pattern. Thread-safe.
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
        self._init_defaults()
        self._initialized = True
    
    def _init_defaults(self):
        """Initialize all settings containers with defaults."""
        self._settings = {
            'llm': LLMSettings(),
            'embedding': EmbeddingSettings(),
            'streaming': StreamingSettings(),
            'models': ModelsSettings(),
            'agent': AgentSettings(),
            'tools': ToolSettings(),
            'prompt_templates': PromptTemplatesSettings(),
            'rag': RAGSettings(),
            'network_filesystem': NetworkFilesystemSettings(),
            'documents': DocumentSettings(),
            'server': ServerSettings(),
            'paths': PathSettings(),
            'localization': LocalizationSettings(),
            'security': {}, # Dict for dynamic security settings
            'workflows': [] # List for workflows
        }

    def initialize(self, config_file: str = "config.json", base_dir: str = None):
        """Load configuration from disk."""
        with self._lock:
            # Setup Paths
            if base_dir:
                self._settings['paths'].base_dir = base_dir
            else:
                self._settings['paths'].base_dir = str(Path.cwd())

            self._config_file = Path(self._settings['paths'].base_dir) / config_file
            
            # Load
            if self._config_file.exists():
                self._load_from_file()
            else:
                print(f"Config file {self._config_file} not found. Using defaults.")

    def _load_from_file(self):
        try:
            with open(self._config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for category, settings_obj in self._settings.items():
                if category in data:
                    # Handle raw dicts (security, workflows) directly
                    if isinstance(settings_obj, (dict, list)):
                        self._settings[category] = data[category]
                    else:
                        # Update dataclass fields from dict
                        valid_keys = settings_obj.__dataclass_fields__.keys()
                        subset = {k: v for k, v in data[category].items() if k in valid_keys}
                        for k, v in subset.items():
                            setattr(settings_obj, k, v)
                        
            print("Configuration loaded successfully.")
        except Exception as e:
            print(f"Error loading config: {e}")

    def save(self):
        """Persist current configuration to disk."""
        if not self._config_file:
            return
        
        with self._lock:
            data = self.as_dict()
            with open(self._config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def as_dict(self) -> Dict[str, Any]:
        """Return the complete configuration as a dictionary."""
        data = {}
        for k, v in self._settings.items():
            if hasattr(v, 'to_dict'):
                data[k] = v.to_dict()
            else:
                data[k] = v
        return data

    def get(self, category: str, default=None):
        return self._settings.get(category, default)
        
    # Accessors for easy usage
    @property
    def llm(self) -> LLMSettings: return self._settings['llm']
    @property
    def agent(self) -> AgentSettings: return self._settings['agent']
    @property
    def tools(self) -> ToolSettings: return self._settings['tools']
    @property
    def prompts(self) -> PromptTemplatesSettings: return self._settings['prompt_templates']
    @property
    def rag(self) -> RAGSettings: return self._settings['rag']
    @property
    def paths(self) -> PathSettings: return self._settings['paths']
    @property
    def server(self) -> ServerSettings: return self._settings['server']
    @property
    def models(self) -> ModelsSettings: return self._settings['models']

# Global Instance
config_manager = ConfigurationManager()
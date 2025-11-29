# app/config/schema_additions.py
"""
Additional configuration schemas for network filesystem and instructions.
This extends the existing schema.py with new configuration sections.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path

# Import existing from schema.py
try:
    from app.config.schema import ConfigCategory, ConfigField
except ImportError:
    # Fallback if not available yet
    class ConfigCategory(str, Enum):
        LLM = "llm"
        EMBEDDING = "embedding"
        RAG = "rag"
        AGENT = "agent"
        TOOLS = "tools"
        STORAGE = "storage"
        SERVER = "server"
        DOCUMENTS = "documents"
        RESPONSE = "response"
        NETWORK = "network"  # New category
        INSTRUCTIONS = "instructions"  # New category
    
    @dataclass
    class ConfigField:
        name: str
        category: ConfigCategory
        description: str
        field_type: str
        default: Any
        min_value: Optional[float] = None
        max_value: Optional[float] = None
        options: Optional[List[Any]] = None
        requires_restart: bool = False


# =============================================================================
# Network Filesystem Configuration
# =============================================================================

@dataclass
class NetworkShareConfig:
    """Configuration for a single network share."""
    name: str = "default"
    mount_path: str = ""
    share_type: str = "smb"  # smb, nfs, local
    enabled: bool = False
    auto_index: bool = True
    watch_for_changes: bool = True
    scan_interval: int = 300  # seconds
    include_extensions: List[str] = field(default_factory=lambda: [
        '.txt', '.md', '.pdf', '.doc', '.docx', 
        '.xls', '.xlsx', '.csv', '.json', '.yaml'
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '.*', '~$*', 'Thumbs.db', 'desktop.ini'
    ])
    max_file_size_mb: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkShareConfig':
        return cls(**data)


@dataclass
class NetworkFilesystemSettings:
    """Network filesystem configuration."""
    enabled: bool = False
    shares: List[Dict[str, Any]] = field(default_factory=list)  # List of NetworkShareConfig dicts
    auto_start_monitoring: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "shares": self.shares,
            "auto_start_monitoring": self.auto_start_monitoring
        }
    
    def get_shares(self) -> List[NetworkShareConfig]:
        """Convert share dicts to NetworkShareConfig objects."""
        return [NetworkShareConfig.from_dict(s) for s in self.shares]
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("enabled", ConfigCategory.NETWORK,
                       "Enable network filesystem monitoring", "bool", False),
            ConfigField("auto_start_monitoring", ConfigCategory.NETWORK,
                       "Auto-start monitoring on startup", "bool", True),
        ]


# =============================================================================
# Instructions Configuration (replaces txt files)
# =============================================================================

@dataclass
class InstructionsSettings:
    """AI instructions configuration (replaces persona.txt/rules.txt)."""
    # Persona
    persona_name: str = "Panos"
    persona_description: str = "A helpful AI assistant"
    persona_age: Optional[int] = 28
    persona_role: str = "AI Assistant"
    personality_traits: List[str] = field(default_factory=lambda: [
        "helpful", "professional", "precise", "friendly"
    ])
    
    # Response rules
    directness: bool = True
    use_context_awareness: bool = True
    follow_user_instructions: bool = True
    natural_conversation: bool = True
    avoid_meta_commentary: bool = True
    prefer_prose: bool = True
    
    # Tool handling
    auto_select_best_file: bool = True
    prefer_knowledge_directory: bool = True
    
    # Knowledge base
    check_knowledge_first: bool = True
    cite_sources: bool = True
    
    # Custom additions (for migration from txt files)
    custom_persona_text: str = ""
    custom_rules_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_system_prompt(self) -> str:
        """Generate system prompt from configuration."""
        sections = []
        
        # Persona section
        persona_parts = []
        if self.persona_name:
            persona_parts.append(f"Your name is {self.persona_name}")
        if self.persona_role:
            persona_parts.append(f"You are {self.persona_role}")
        if self.persona_description:
            persona_parts.append(self.persona_description)
        if self.personality_traits:
            traits = ", ".join(self.personality_traits)
            persona_parts.append(f"Your personality: {traits}")
        
        if self.custom_persona_text:
            persona_parts.append(self.custom_persona_text)
        
        if persona_parts:
            sections.append("# PERSONA\n\n" + "\n\n".join(persona_parts))
        
        # Response rules section
        rules_parts = []
        
        if self.directness:
            rules_parts.append("**Directness:** Answer first, explain after. Don't deflect with questions unless truly ambiguous.")
        
        if self.use_context_awareness:
            rules_parts.append("**Context Awareness:** Use remembered information about the user when relevant.")
        
        if self.follow_user_instructions:
            rules_parts.append("**Instruction Following:** Follow user-set rules precisely and consistently.")
        
        if self.natural_conversation:
            rules_parts.append("**Natural Conversation:** Avoid repetition. Match the user's style.")
        
        if self.avoid_meta_commentary:
            rules_parts.append("Don't narrate your reasoning process. Just respond naturally.")
        
        if self.prefer_prose:
            rules_parts.append("**Formatting:** Use prose and paragraphs by default. Avoid bullet points unless requested.")
        
        if self.custom_rules_text:
            rules_parts.append(self.custom_rules_text)
        
        if rules_parts:
            sections.append("# CORE PRINCIPLES\n\n" + "\n\n".join(rules_parts))
        
        # Tool handling section
        if self.auto_select_best_file:
            tool_rules = []
            tool_rules.append("# TOOL HANDLING")
            tool_rules.append("")
            tool_rules.append("**File Operations:**")
            tool_rules.append("- Auto-select best matching file (prefer knowledge directory)")
            tool_rules.append("- Provide complete content unless asked to truncate")
            tool_rules.append("- Don't ask 'which file?' when content is provided")
            
            sections.append("\n".join(tool_rules))
        
        # Knowledge base section
        if self.check_knowledge_first:
            kb_rules = []
            kb_rules.append("# KNOWLEDGE BASE")
            kb_rules.append("")
            kb_rules.append("You have access to a knowledge base. When provided context:")
            kb_rules.append("1. Use the context to answer questions")
            kb_rules.append("2. Knowledge base is authoritative")
            kb_rules.append("3. Check knowledge base BEFORE other sources")
            
            sections.append("\n".join(kb_rules))
        
        return "\n\n".join(sections)
    
    @classmethod
    def get_field_metadata(cls) -> List[ConfigField]:
        return [
            ConfigField("persona_name", ConfigCategory.INSTRUCTIONS,
                       "AI persona name", "str", "Panos"),
            ConfigField("persona_role", ConfigCategory.INSTRUCTIONS,
                       "AI role description", "str", "AI Assistant"),
            ConfigField("directness", ConfigCategory.INSTRUCTIONS,
                       "Answer directly without deflection", "bool", True),
            ConfigField("prefer_prose", ConfigCategory.INSTRUCTIONS,
                       "Use prose instead of bullet points", "bool", True),
            ConfigField("auto_select_best_file", ConfigCategory.INSTRUCTIONS,
                       "Auto-select files with fuzzy matching", "bool", True),
        ]


# Migration helper
def migrate_from_txt_files(instructions_dir: Path) -> InstructionsSettings:
    """
    Migrate from old persona.txt/rules.txt to new config.
    """
    from pathlib import Path
    
    config = InstructionsSettings()
    
    persona_file = Path(instructions_dir) / "persona.txt"
    rules_file = Path(instructions_dir) / "rules.txt"
    
    if persona_file.exists():
        config.custom_persona_text = persona_file.read_text(encoding="utf-8")
    
    if rules_file.exists():
        config.custom_rules_text = rules_file.read_text(encoding="utf-8")
    
    return config
# app/config/instructions.py
"""
Instructions configuration - centralizes all AI behavior settings.
Replaces persona.txt and rules.txt with config-based system.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class PersonaConfig:
    """AI persona configuration."""
    name: str = ""
    description: str = "A helpful AI assistant"
    age: Optional[int] = 28
    role: str = "AI Assistant"
    personality_traits: List[str] = field(default_factory=lambda: [
        "helpful", "professional", "precise", "friendly"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResponseRules:
    """Rules for how the AI responds."""
    # Response structure
    directness: bool = True  # Answer first, explain after
    use_context_awareness: bool = True  # Use remembered info when relevant
    follow_user_instructions: bool = True  # Follow "when I say X, respond Y" rules
    natural_conversation: bool = True  # Match user's style, avoid repetition
    
    # Response patterns
    avoid_meta_commentary: bool = True  # Don't narrate reasoning
    avoid_narration: bool = True  # Don't include "Let me think..." etc
    cite_knowledge_base: bool = True  # Reference knowledge base for answers
    
    # Formatting preferences
    use_bullet_points_by_default: bool = False  # Use prose unless asked
    max_bullet_points_without_request: int = 0  # Don't use lists unless asked
    prefer_prose: bool = True  # Write in paragraphs
    
    # Behavior
    ask_clarification_when_ambiguous: bool = True
    acknowledge_missing_info: bool = True  # Say "I don't know" vs guessing
    use_most_recent_on_conflict: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolHandlingRules:
    """Rules for how AI uses tools."""
    # File operations
    auto_select_best_file: bool = True  # Use fuzzy matching
    prefer_knowledge_directory: bool = False  # Changed to False (no knowledge dir)
    provide_complete_content: bool = True  # Don't truncate unless asked
    
    # Response patterns for file operations
    acknowledge_file_read: bool = True  # "I read X from Y"
    note_alternate_versions: bool = True  # Mention other file versions
    
    # Error handling
    ask_which_file_if_ambiguous: bool = False  # Auto-select instead
    repeat_selection_questions: bool = False  # Don't ask twice
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeBaseRules:
    """Rules for knowledge base usage."""
    check_knowledge_first: bool = True  # Always check KB before other sources
    cite_sources: bool = True  # Reference where info came from
    prefer_kb_over_web: bool = True  # Knowledge base is authoritative
    
    # Context handling
    use_provided_context: bool = True  # Use <context> tags
    trust_context_tags: bool = True  # Context is authoritative
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InstructionsConfig:
    """Complete instructions configuration."""
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    response_rules: ResponseRules = field(default_factory=ResponseRules)
    tool_handling: ToolHandlingRules = field(default_factory=ToolHandlingRules)
    knowledge_base: KnowledgeBaseRules = field(default_factory=KnowledgeBaseRules)
    
    # Custom instructions (user-added)
    custom_persona_text: str = ""
    custom_rules_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "persona": self.persona.to_dict(),
            "response_rules": self.response_rules.to_dict(),
            "tool_handling": self.tool_handling.to_dict(),
            "knowledge_base": self.knowledge_base.to_dict(),
            "custom_persona_text": self.custom_persona_text,
            "custom_rules_text": self.custom_rules_text
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstructionsConfig':
        """Load from dictionary."""
        return cls(
            persona=PersonaConfig(**data.get("persona", {})),
            response_rules=ResponseRules(**data.get("response_rules", {})),
            tool_handling=ToolHandlingRules(**data.get("tool_handling", {})),
            knowledge_base=KnowledgeBaseRules(**data.get("knowledge_base", {})),
            custom_persona_text=data.get("custom_persona_text", ""),
            custom_rules_text=data.get("custom_rules_text", "")
        )
    
    def to_system_prompt(self) -> str:
        """Generate system prompt from configuration."""
        sections = []
        
        # Persona section
        persona_parts = []
        if self.persona.name:
            persona_parts.append(f"Your name is {self.persona.name}")
        if self.persona.role:
            persona_parts.append(f"You are {self.persona.role}")
        if self.persona.description:
            persona_parts.append(self.persona.description)
        if self.persona.personality_traits:
            traits = ", ".join(self.persona.personality_traits)
            persona_parts.append(f"Your personality: {traits}")
        
        if self.custom_persona_text:
            persona_parts.append(self.custom_persona_text)
        
        if persona_parts:
            sections.append("# PERSONA\n\n" + "\n\n".join(persona_parts))
        
        # Response rules section
        rules_parts = []
        
        if self.response_rules.directness:
            rules_parts.append("**Directness:** Answer first, explain after. Don't deflect with questions unless truly ambiguous.")
        
        if self.response_rules.use_context_awareness:
            rules_parts.append("**Context Awareness:** Use remembered information about the user when relevant. Don't force personal context on factual questions.")
        
        if self.response_rules.follow_user_instructions:
            rules_parts.append("**Instruction Following:** When users set rules (\"when I say X, respond Y\", \"always be brief\"), follow them precisely and consistently.")
        
        if self.response_rules.natural_conversation:
            rules_parts.append("**Natural Conversation:** Avoid repetition. Match the user's formality and style. Focus on being helpful over being social.")
        
        if self.response_rules.avoid_meta_commentary:
            rules_parts.append("Don't narrate your reasoning process. Don't include meta-commentary. Just respond naturally.")
        
        if self.response_rules.prefer_prose:
            rules_parts.append("**Formatting:** Use prose and paragraphs by default. Avoid bullet points and lists unless explicitly requested.")
        
        if self.custom_rules_text:
            rules_parts.append(self.custom_rules_text)
        
        if rules_parts:
            sections.append("# CORE PRINCIPLES\n\n" + "\n\n".join(rules_parts))
        
        # Tool handling section
        if self.tool_handling.auto_select_best_file:
            tool_rules = []
            tool_rules.append("**File Operations:**")
            tool_rules.append("- Auto-select best matching file when multiple exist (prefer network shares)")
            tool_rules.append("- Provide complete file content unless specifically asked to truncate")
            tool_rules.append("- Respond: \"I read [filename] from [location]: [content]\"")
            tool_rules.append("- Don't ask \"which file?\" when content is already provided")
            
            sections.append("\n".join(tool_rules))
        
        # Knowledge base section
        if self.knowledge_base.check_knowledge_first:
            kb_rules = []
            kb_rules.append("# NETWORK KNOWLEDGE BASE")
            kb_rules.append("")
            kb_rules.append("You have access to documents from network shares. When provided context in <context> tags:")
            kb_rules.append("1. Use the context to answer questions")
            kb_rules.append("2. The knowledge base is authoritative - prefer it over other sources")
            kb_rules.append("3. Check network knowledge base BEFORE using other tools or sources")
            
            sections.append("\n".join(kb_rules))
        
        return "\n\n".join(sections)
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'InstructionsConfig':
        """Load configuration from JSON file."""
        if not path.exists():
            return cls()  # Return defaults
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


# ============================================================================
# InstructionsSettings - Simplified version for config compatibility
# ============================================================================

@dataclass
class InstructionsSettings:
    """
    Simplified instructions settings for config system compatibility.
    This is what gets loaded from config.json.
    """
    persona_name: str = ""
    persona_description: str = "A helpful AI assistant"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstructionsSettings':
        """Create from dictionary."""
        # Handle both simple and complex formats
        if "persona" in data:
            # Complex format (InstructionsConfig)
            config = InstructionsConfig.from_dict(data)
            return cls(
                persona_name=config.persona.name,
                persona_description=config.persona.description
            )
        else:
            # Simple format
            return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def to_system_prompt(self) -> str:
        """Convert to system prompt."""
        return f"""
Your name is {self.persona_name} and you are {self.persona_description}.

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

You have access to documents from network shares. When provided context in <context> tags:
1. Use the context to answer questions
2. The knowledge base is authoritative - prefer it over other sources
3. Check network knowledge base BEFORE using other tools or sources

## Tool Handling - File Operations

**What Happens:**
- Tool auto-selects best file when multiple matches exist (prefers network shares)
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


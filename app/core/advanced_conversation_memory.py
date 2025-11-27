# app/core/advanced_conversation_memory.py
"""
Advanced conversation memory system for robust context tracking.
Handles multiple conversation patterns, user instructions, and context awareness.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import uuid
import time
import re
from threading import Lock
from enum import Enum

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class InstructionType(Enum):
    """Types of user instructions the system can recognize."""
    RESPONSE_RULE = "response_rule"  # "when I say X, respond Y"
    BEHAVIOR_RULE = "behavior_rule"  # "always be concise", "speak formally"
    CONTEXT_RULE = "context_rule"    # "remember I'm a developer", "I work in healthcare"
    PREFERENCE = "preference"        # "call me John", "use metric units"
    CONSTRAINT = "constraint"        # "don't mention politics", "avoid technical jargon"

@dataclass
class ConversationInstruction:
    """Structured representation of user instructions."""
    instruction_type: InstructionType
    trigger: Optional[str] = None  # For response rules
    response: Optional[str] = None  # For response rules
    content: str = ""  # For other instruction types
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

@dataclass
class ConversationContext:
    """Complete conversation context with robust memory."""
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    instructions: Dict[str, ConversationInstruction] = field(default_factory=dict)
    user_facts: Dict[str, Any] = field(default_factory=dict)  # "user is a developer", etc.
    conversation_topics: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    def add_message(self, role: str, content: str):
        """Add message and extract any instructions or facts."""
        message_data = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        self.messages.append(message_data)
        self.last_accessed = time.time()
        
        if role == "user":
            self._process_user_message(content)
    
    def _process_user_message(self, content: str):
        """Process user message for instructions, facts, and topics."""
        content_lower = content.lower()
        
        # Extract instructions
        instructions = self._extract_instructions(content)
        for instruction in instructions:
            self._store_instruction(instruction)
        
        # Extract user facts
        facts = self._extract_user_facts(content)
        for key, value in facts.items():
            self.user_facts[key] = value
        
        # Extract conversation topics
        topics = self._extract_topics(content)
        self.conversation_topics.update(topics)
    
    def _extract_instructions(self, content: str) -> List[ConversationInstruction]:
        """Extract various types of instructions from user message."""
        instructions = []
        content_lower = content.lower()
        
        # Pattern 1: Response rules - "when I say X, respond Y"
        response_patterns = [
            r'when i say ["\']?([^"\']+)["\']?\s*,\s*(?:you|respond|answer)\s+["\']?([^"\']+)["\']?',
            r'if i say ["\']?([^"\']+)["\']?\s*,\s*(?:you|respond|answer)\s+["\']?([^"\']+)["\']?',
            r'every time i say ["\']?([^"\']+)["\']?\s*,\s*(?:you|respond|answer)\s+["\']?([^"\']+)["\']?',
        ]
        
        for pattern in response_patterns:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                trigger, response = match.groups()
                if trigger and response:
                    instructions.append(ConversationInstruction(
                        instruction_type=InstructionType.RESPONSE_RULE,
                        trigger=trigger.strip(),
                        response=response.strip(),
                        content=f"When user says '{trigger}', respond with '{response}'"
                    ))
        
        # Pattern 2: Behavior rules
        behavior_patterns = [
            (r'(always|never)\s+(be|act|speak|talk|respond)\s+([^.!?]+)', self._parse_behavior_rule),
            (r'(be|act)\s+([^.!?]+)(?:\s+from now on|\s+always)', self._parse_behavior_directive),
        ]
        
        for pattern, parser in behavior_patterns:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                instruction = parser(match, content)
                if instruction:
                    instructions.append(instruction)
        
        # Pattern 3: Preferences and constraints
        preference_patterns = [
            (r'(call me|refer to me as|my name is)\s+([^.!?]+)', InstructionType.PREFERENCE),
            (r'(remember that|keep in mind|note that)\s+([^.!?]+)', InstructionType.CONTEXT_RULE),
            (r'(don\'?t|never)\s+(mention|talk about|discuss)\s+([^.!?]+)', InstructionType.CONSTRAINT),
        ]
        
        for pattern, instr_type in preference_patterns:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                instructions.append(ConversationInstruction(
                    instruction_type=instr_type,
                    content=match.group(0).strip()
                ))
        
        return instructions
    
    def _parse_behavior_rule(self, match: re.Match, original_content: str) -> Optional[ConversationInstruction]:
        """Parse behavior rules like 'always be concise'."""
        try:
            frequency, action, behavior = match.groups()
            return ConversationInstruction(
                instruction_type=InstructionType.BEHAVIOR_RULE,
                content=f"{frequency.capitalize()} {action} {behavior.strip()}",
                confidence=0.9
            )
        except:
            return None
    
    def _parse_behavior_directive(self, match: re.Match, original_content: str) -> Optional[ConversationInstruction]:
        """Parse behavior directives like 'be concise from now on'."""
        try:
            action, behavior = match.groups()
            return ConversationInstruction(
                instruction_type=InstructionType.BEHAVIOR_RULE,
                content=f"Always {action} {behavior.strip()}",
                confidence=0.8
            )
        except:
            return None
    
    def _extract_user_facts(self, content: str) -> Dict[str, Any]:
        """Extract factual information about the user."""
        facts = {}
        content_lower = content.lower()
        
        # Profession/role patterns
        profession_patterns = [
            r'i am a ([^.!?]+)',
            r'i work as a ([^.!?]+)',
            r'i\'m a ([^.!?]+)',
            r'my job is ([^.!?]+)',
        ]
        
        for pattern in profession_patterns:
            match = re.search(pattern, content_lower)
            if match:
                facts['profession'] = match.group(1).strip()
                break
        
        # Location patterns
        location_patterns = [
            r'i live in ([^.!?]+)',
            r'i\'m from ([^.!?]+)',
            r'my city is ([^.!?]+)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, content_lower)
            if match:
                facts['location'] = match.group(1).strip()
                break
        
        return facts
    
    def _extract_topics(self, content: str) -> Set[str]:
        """Extract conversation topics using simple keyword matching."""
        topics = set()
        
        # Common topic keywords
        topic_keywords = {
            'work', 'job', 'career', 'programming', 'coding', 'development',
            'health', 'fitness', 'exercise', 'diet', 'food', 'cooking',
            'travel', 'vacation', 'holiday', 'family', 'friends', 'relationships',
            'technology', 'ai', 'machine learning', 'software', 'hardware',
            'sports', 'games', 'gaming', 'music', 'movies', 'entertainment',
            'finance', 'money', 'investment', 'business', 'education', 'learning'
        }
        
        content_lower = content.lower()
        for topic in topic_keywords:
            if topic in content_lower:
                topics.add(topic)
        
        return topics
    
    def _store_instruction(self, instruction: ConversationInstruction):
        """Store instruction with deduplication."""
        # Create a unique key based on instruction content
        instruction_key = f"{instruction.instruction_type.value}_{hash(instruction.content) % 10000}"
        self.instructions[instruction_key] = instruction
        logger.info(f"📝 Stored {instruction.instruction_type.value}: {instruction.content}")
    
    def get_recent_messages(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent messages for context."""
        return self.messages[-max_messages:]
    
    def get_conversation_context(self) -> str:
        """Generate comprehensive conversation context."""
        context_parts = []
        
        # 1. User facts
        if self.user_facts:
            facts_text = "## User Information:\n" + "\n".join(
                f"- {key}: {value}" for key, value in self.user_facts.items()
            )
            context_parts.append(facts_text)
        
        # 2. Active instructions
        active_instructions = [
            instr for instr in self.instructions.values()
            if time.time() - instr.last_used < 3600  # Instructions active for 1 hour
        ]
        
        if active_instructions:
            instructions_text = "## Active Instructions:\n" + "\n".join(
                f"- {instr.content}" for instr in active_instructions
            )
            context_parts.append(instructions_text)
        
        # 3. Conversation topics
        if self.conversation_topics:
            topics_text = "## Conversation Topics:\n" + ", ".join(sorted(self.conversation_topics))
            context_parts.append(topics_text)
        
        # 4. Recent messages summary
        if len(self.messages) > 1:
            recent_count = min(5, len(self.messages) - 1)
            context_parts.append(f"## Recent Messages ({recent_count} most recent):")
            for msg in self.messages[-recent_count-1:-1]:  # Exclude current message
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                # Truncate long messages
                if len(content) > 150:
                    content = content[:147] + "..."
                context_parts.append(f"{role}: {content}")
        
        return "\n\n".join(context_parts)
    
    def check_instruction_match(self, query: str) -> Optional[str]:
        """Check if query matches any response rules."""
        query_lower = query.lower().strip()
        
        for instruction in self.instructions.values():
            if instruction.instruction_type == InstructionType.RESPONSE_RULE:
                if instruction.trigger and self._matches_trigger(query_lower, instruction.trigger):
                    instruction.last_used = time.time()
                    return instruction.response
        
        return None
    
    def _matches_trigger(self, query: str, trigger: str) -> bool:
        """Check if query matches trigger with flexible matching."""
        trigger_lower = trigger.lower()
        
        # Exact match
        if query == trigger_lower:
            return True
        
        # Contains match
        if trigger_lower in query:
            return True
        
        # Fuzzy match for short triggers
        if len(trigger_lower) <= 10:
            # Remove punctuation and check
            import string
            query_clean = query.translate(str.maketrans('', '', string.punctuation))
            trigger_clean = trigger_lower.translate(str.maketrans('', '', string.punctuation))
            
            if trigger_clean in query_clean:
                return True
        
        return False

class AdvancedConversationMemory:
    """Advanced memory manager with robust session handling."""
    
    def __init__(self, max_sessions: int = 100, session_timeout: int = 3600):
        self.sessions: Dict[str, ConversationContext] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self._lock = Lock()
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationContext:
        """Get existing session or create new one."""
        with self._lock:
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                session.last_accessed = time.time()
                return session
            
            # Create new session
            new_session_id = session_id or str(uuid.uuid4())
            session = ConversationContext(session_id=new_session_id)
            self.sessions[new_session_id] = session
            
            self._cleanup_sessions()
            return session
    
    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """Get session by ID."""
        with self._lock:
            return self.sessions.get(session_id)
    
    def _cleanup_sessions(self):
        """Clean up old sessions."""
        if len(self.sessions) <= self.max_sessions:
            return
        
        current_time = time.time()
        old_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session.last_accessed > self.session_timeout
        ]
        
        for session_id in old_sessions:
            del self.sessions[session_id]
        
        if old_sessions:
            logger.info(f"🧹 Cleaned up {len(old_sessions)} old sessions")

# Global advanced conversation memory
advanced_memory = AdvancedConversationMemory()
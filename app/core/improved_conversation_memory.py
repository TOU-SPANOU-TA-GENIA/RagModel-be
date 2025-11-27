# app/core/improved_conversation_memory.py
"""
Improved conversation memory with better instruction storage.
Uses meaningful names instead of "rule_0", "rule_1", etc.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import uuid
import time
from threading import Lock
import re
import hashlib

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class ConversationContext:
    """Complete conversation context for a session."""
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    user_instructions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self.last_accessed = time.time()
        
        # Extract and store user instructions
        if role == "user" and self._is_instruction(content):
            self._store_instruction(content)
    
    def get_recent_messages(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent messages for context."""
        return self.messages[-max_messages:]
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of key conversation points."""
        if not self.user_instructions:
            return ""
        
        summary = ["Active Instructions:"]
        for key, instruction in self.user_instructions.items():
            if isinstance(instruction, dict):
                summary.append(f"- {instruction.get('description', instruction)}")
            else:
                summary.append(f"- {instruction}")
        
        return "\n".join(summary)
    
    def _is_instruction(self, content: str) -> bool:
        """Check if message contains user instructions."""
        instruction_keywords = [
            "when i say", "you will answer", "respond with", "always respond",
            "always be", "never say", "follow this rule", "remember to",
            "end with", "i prefer", "keep it"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in instruction_keywords)
    
    def _store_instruction(self, content: str):
        """Extract and store user instructions with meaningful naming."""
        content_lower = content.lower()
        
        # Enhanced patterns for instruction detection
        patterns = [
            # Response rules: "when I say X, respond Y"
            (r'when\s+i\s+say\s+["\']?([^"\']+?)["\']?\s*,?\s*(?:you\s+)?(?:respond|answer|say)(?:\s+with)?\s+["\']?([^"\']+?)["\']?(?:\.|$)', 
             'response_trigger', 1, 2),
            
            # Behavioral: "always be X"
            (r'always\s+be\s+([^\.,]+)', 'behavior', 0, 1),
            
            # Behavioral: "always X"
            (r'always\s+([^\.,]{3,50})', 'behavior', 0, 1),
            
            # Format: "end with X"
            (r'end(?:\s+your)?(?:\s+responses?)?\s+with\s+["\']?([^"\']+?)["\']?(?:\.|$)', 
             'format', 0, 1),
            
            # Preference: "I prefer X"
            (r'i\s+prefer\s+([^\.,]{3,50})', 'preference', 0, 1),
            
            # Preference: "keep it X"
            (r'keep\s+it\s+([^\.,]{3,50})', 'preference', 0, 1),
            
            # Never rules
            (r'never\s+([^\.,]{3,50})', 'constraint', 0, 1),
        ]
        
        for pattern_info in patterns:
            pattern = pattern_info[0]
            instruction_type = pattern_info[1]
            trigger_group = pattern_info[2]
            response_group = pattern_info[3]
            
            match = re.search(pattern, content_lower, re.IGNORECASE)
            if match:
                try:
                    # Generate meaningful key
                    if instruction_type == 'response_trigger':
                        trigger = match.group(trigger_group).strip().strip('"\'')
                        response = match.group(response_group).strip().strip('"\'')
                        
                        # Create hash-based key for uniqueness
                        key = f"trigger_{self._hash_text(trigger)[:8]}"
                        
                        self.user_instructions[key] = {
                            'type': 'response_trigger',
                            'trigger': trigger,
                            'response': response,
                            'description': f"When you say '{trigger}', I respond '{response}'"
                        }
                        logger.info(f"📝 Stored trigger: '{trigger}' -> '{response}'")
                        
                    else:
                        value = match.group(response_group).strip().strip('"\'')
                        
                        # Create descriptive key
                        key = f"{instruction_type}_{self._hash_text(value)[:8]}"
                        
                        self.user_instructions[key] = {
                            'type': instruction_type,
                            'value': value,
                            'description': f"{instruction_type.capitalize()}: {value}"
                        }
                        logger.info(f"📝 Stored {instruction_type}: '{value}'")
                    
                    return
                    
                except (IndexError, AttributeError) as e:
                    logger.debug(f"Pattern match error: {e}")
                    continue
        
        # Fallback: store as general instruction
        if any(word in content_lower for word in ["when i say", "always", "never", "remember"]):
            key = f"instruction_{self._hash_text(content)[:8]}"
            self.user_instructions[key] = {
                'type': 'general',
                'value': content,
                'description': content[:100]
            }
            logger.info(f"📝 Stored general instruction")
    
    def _hash_text(self, text: str) -> str:
        """Generate short hash for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def check_trigger_match(self, query: str) -> Optional[str]:
        """Check if query matches any response triggers."""
        query_lower = query.lower().strip()
        
        for instruction in self.user_instructions.values():
            if isinstance(instruction, dict) and instruction.get('type') == 'response_trigger':
                trigger = instruction.get('trigger', '').lower()
                response = instruction.get('response', '')
                
                if self._matches_trigger(query_lower, trigger):
                    return response
        
        return None
    
    def _matches_trigger(self, query: str, trigger: str) -> bool:
        """Check if query matches trigger with flexible matching."""
        # Exact match
        if query == trigger:
            return True
        
        # Contains match
        if trigger in query:
            return True
        
        # Word boundary match for short triggers
        if len(trigger) <= 15:
            pattern = r'\b' + re.escape(trigger) + r'\b'
            if re.search(pattern, query):
                return True
        
        return False
    
    def get_active_behaviors(self) -> List[str]:
        """Get list of active behavioral instructions."""
        behaviors = []
        
        for instruction in self.user_instructions.values():
            if isinstance(instruction, dict):
                inst_type = instruction.get('type')
                if inst_type in ['behavior', 'preference', 'format']:
                    behaviors.append(instruction.get('description', ''))
        
        return behaviors


class ConversationMemory:
    """Manages conversation memory across sessions."""
    
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
            
            # Clean up old sessions if needed
            self._cleanup_sessions()
            
            return session
    
    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """Get session by ID."""
        with self._lock:
            return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str):
        """Delete a session."""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
    
    def _cleanup_sessions(self):
        """Clean up old sessions."""
        if len(self.sessions) <= self.max_sessions:
            return
        
        current_time = time.time()
        sessions_to_delete = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_accessed > self.session_timeout:
                sessions_to_delete.append(session_id)
        
        # If still over limit, remove oldest
        if len(self.sessions) - len(sessions_to_delete) > self.max_sessions:
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].last_accessed
            )
            additional_to_delete = len(self.sessions) - self.max_sessions
            sessions_to_delete.extend([s[0] for s in sorted_sessions[:additional_to_delete]])
        
        for session_id in sessions_to_delete:
            del self.sessions[session_id]
        
        if sessions_to_delete:
            logger.info(f"Cleaned up {len(sessions_to_delete)} old sessions")


# Global conversation memory
conversation_memory = ConversationMemory()
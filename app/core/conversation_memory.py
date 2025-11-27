# app/core/conversation_memory.py
"""
Enhanced conversation memory system that maintains context across messages.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import uuid
import time
from threading import Lock

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
        
        summary = ["## Conversation Instructions:"]
        for key, instruction in self.user_instructions.items():
            summary.append(f"- {instruction}")
        
        return "\n".join(summary)
    
    def _is_instruction(self, content: str) -> bool:
        """Check if message contains user instructions."""
        instruction_keywords = [
            "when i say", "you will answer", "remember that", 
            "always respond", "never say", "follow this rule"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in instruction_keywords)
    
    def _store_instruction(self, content: str):
        """Extract and store user instructions with better parsing."""
        content_lower = content.lower()
        
        # Enhanced patterns for instruction detection
        patterns = [
            # Response rules: "when I say X, respond Y"
            (r'when\s+i\s+say\s+["\']?([^"\']+?)["\']?\s*,?\s*(?:you\s+)?(?:respond|answer|say)(?:\s+with)?\s+["\']?([^"\']+?)["\']?(?:\.|$)', 1, 2),
            # Simpler: "X means Y"
            (r'["\']([^"\']{2,20})["\']?\s+means\s+["\']?([^"\']+?)["\']?(?:\.|$)', 1, 2),
            # Behavioral: "always do X"
            (r'always\s+(.*?)(?:\.|$)', 0, 1),
            # Constraint: "never do X"  
            (r'never\s+(.*?)(?:\.|$)', 0, 1),
            # Format: "end with X"
            (r'end(?:\s+your)?(?:\s+responses?)?\s+with\s+["\']?([^"\']+?)["\']?(?:\.|$)', 0, 1),
        ]
        
        import re
        import hashlib
        for pattern, trigger_group, response_group in patterns:
            match = re.search(pattern, content_lower, re.IGNORECASE)
            if match:
                try:
                    trigger = match.group(trigger_group).strip().strip('"\'') if trigger_group > 0 else None
                    response = match.group(response_group).strip().strip('"\'')
                    
                    if trigger and response:
                        key = f"trigger_{hashlib.md5(trigger.encode()).hexdigest()[:8]}"
                        self.user_instructions[key] = {
                            'trigger': trigger,
                            'response': response,
                            'description': f"When you say '{trigger}', I respond '{response}'"
        }
                    elif response:
                        # Store behavioral instruction
                        self.user_instructions[f"rule_{len(self.user_instructions)}"] = response
                        logger.info(f"📝 Stored rule: '{response}'")
                    else:
                        key = f"behavior_{hashlib.md5(content.encode()).hexdigest()[:8]}"
                        self.user_instructions[key] = {
                            'type': 'behavior',
                            'value': content,
                            'description': content
                        }
                    return
                except IndexError:
                    continue
        
        # If patterns didn't match but looks like instruction, store anyway
        if any(word in content_lower for word in ["when i say", "always", "never", "remember"]):
            self.user_instructions[f"instruction_{len(self.user_instructions)}"] = content
            logger.info(f"📝 Stored general instruction: '{content[:50]}'")


class ConversationMemory:
    """
    Manages conversation memory across sessions.
    """
    
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
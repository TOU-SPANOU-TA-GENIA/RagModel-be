# app/db/storage.py
"""
Hybrid storage layer: SQLite (persistent) + Redis (cache).
Handles users, chats, and messages with write-through caching.
"""

import sqlite3
import redis
import json
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import contextmanager
from pathlib import Path

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridStorage:
    """
    Hybrid storage combining SQLite (durability) and Redis (speed).
    
    Pattern:
    - Writes: Go to both SQLite and Redis
    - Reads: Try Redis first, fallback to SQLite
    """
    
    def __init__(
        self, 
        db_path: str = "data/app.db",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 3600  # 1 hour
    ):
        self.db_path = Path(db_path)
        self.cache_ttl = cache_ttl
        
        # Initialize Redis connection
        try:
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_connect_timeout=2
            )
            self.redis.ping()
            self.redis_available = True
            logger.info(f"✅ Redis connected at {redis_host}:{redis_port}")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.redis_available = False
            self.redis = None
            logger.warning(f"⚠️  Redis unavailable: {e}. Running without cache.")
    
    @contextmanager
    def get_db(self):
        """Context manager for SQLite connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return dict-like rows
        try:
            yield conn
        finally:
            conn.close()
    
    # =========================================================================
    # User Management
    # =========================================================================
    
    def create_user(self, username: str, email: str, hashed_password: str) -> int:
        """Create a new user account."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (username, email, hashed_password)
                VALUES (?, ?, ?)
                """,
                (username, email, hashed_password)
            )
            conn.commit()
            user_id = cursor.lastrowid
            logger.info(f"✅ User created: {username} (ID: {user_id})")
            return user_id
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username."""
        # Try cache first
        if self.redis_available:
            cached = self.redis.get(f"user:username:{username}")
            if cached:
                logger.debug(f"Cache hit: user {username}")
                return json.loads(cached)
        
        # Cache miss - query database
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            
            if row:
                user = dict(row)
                
                # Cache for future requests
                if self.redis_available:
                    self.redis.setex(
                        f"user:username:{username}",
                        self.cache_ttl,
                        json.dumps(user, default=str)
                    )
                
                return user
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_last_login(self, user_id: int):
        """Update user's last login timestamp."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,)
            )
            conn.commit()
    
    # =========================================================================
    # Chat Management
    # =========================================================================
    
    def create_chat(self, user_id: int, title: str = "Νέα Συνομιλία") -> str:
        """Create a new chat session."""
        chat_id = str(uuid.uuid4())
        
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chats (id, user_id, title)
                VALUES (?, ?, ?)
                """,
                (chat_id, user_id, title)
            )
            conn.commit()
        
        logger.info(f"✅ Chat created: {chat_id} for user {user_id}")
        
        # Invalidate user's chat list cache
        if self.redis_available:
            self.redis.delete(f"user:{user_id}:chats")
        
        return chat_id
    
    def get_user_chats(self, user_id: int) -> List[Dict]:
        """Get all chats for a user."""
        # Try cache first
        if self.redis_available:
            cached = self.redis.get(f"user:{user_id}:chats")
            if cached:
                logger.debug(f"Cache hit: chats for user {user_id}")
                return json.loads(cached)
        
        # Cache miss - query database
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    c.id,
                    c.title,
                    c.updated_at,
                    COUNT(m.id) as message_count
                FROM chats c
                LEFT JOIN messages m ON c.id = m.chat_id
                WHERE c.user_id = ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """,
                (user_id,)
            )
            rows = cursor.fetchall()
            chats = [dict(row) for row in rows]
        
        # Cache for future requests
        if self.redis_available:
            self.redis.setex(
                f"user:{user_id}:chats",
                300,  # 5 minutes (chat lists change frequently)
                json.dumps(chats, default=str)
            )
        
        return chats
    
    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get chat metadata."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_chat_title(self, chat_id: str, title: str):
        """Update chat title."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE chats 
                SET title = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
                """,
                (title, chat_id)
            )
            conn.commit()
        
        # Invalidate cache
        if self.redis_available:
            chat = self.get_chat(chat_id)
            if chat:
                self.redis.delete(f"user:{chat['user_id']}:chats")
    
    def delete_chat(self, chat_id: str):
        """Delete a chat and all its messages."""
        chat = self.get_chat(chat_id)
        if not chat:
            return
        
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.commit()
        
        # Invalidate caches
        if self.redis_available:
            self.redis.delete(f"chat:{chat_id}:messages")
            self.redis.delete(f"user:{chat['user_id']}:chats")
        
        logger.info(f"🗑️  Chat deleted: {chat_id}")
    
    # =========================================================================
    # Message Management
    # =========================================================================
    
    def add_message(self, chat_id: str, role: str, content: str) -> int:
        """Add a message to a chat (write-through to both DB and cache)."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            # Insert message
            cursor.execute(
                """
                INSERT INTO messages (chat_id, role, content)
                VALUES (?, ?, ?)
                """,
                (chat_id, role, content)
            )
            message_id = cursor.lastrowid
            
            # Update chat's updated_at timestamp
            cursor.execute(
                "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (chat_id,)
            )
            
            conn.commit()
        
        # Update Redis cache (append to recent messages)
        if self.redis_available:
            message = {
                "id": message_id,
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add to recent messages list (keep last 50)
            self.redis.lpush(
                f"chat:{chat_id}:recent",
                json.dumps(message)
            )
            self.redis.ltrim(f"chat:{chat_id}:recent", 0, 49)
            
            # Invalidate full message cache
            self.redis.delete(f"chat:{chat_id}:messages")
        
        logger.debug(f"💬 Message added to chat {chat_id}")
        return message_id
    
    def get_messages(self, chat_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get messages for a chat (read-through from cache or DB)."""
        # For recent messages, try cache first
        if limit and limit <= 50 and self.redis_available:
            cached = self.redis.lrange(f"chat:{chat_id}:recent", 0, limit - 1)
            if cached:
                logger.debug(f"Cache hit: {len(cached)} messages for chat {chat_id}")
                return [json.loads(msg) for msg in cached]
        
        # Cache miss or requesting all messages - query database
        with self.get_db() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM messages WHERE chat_id = ? ORDER BY timestamp ASC"
            
            if limit:
                query = f"""
                    SELECT * FROM (
                        SELECT * FROM messages 
                        WHERE chat_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ) ORDER BY timestamp ASC
                """
                cursor.execute(query, (chat_id, limit))
            else:
                cursor.execute(query, (chat_id,))
            
            rows = cursor.fetchall()
            messages = [dict(row) for row in rows]
        
        return messages


# Global storage instance
storage = HybridStorage()
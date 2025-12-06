# app/db/storage.py
"""
Hybrid storage layer: SQLite (persistent) + Redis (cache).
Handles users, chats, and messages with write-through caching.

FIXED: Redis caching now properly implemented with:
- Write-through caching (writes go to both SQLite and Redis)
- Read-through caching (reads check Redis first, fallback to SQLite)
- Automatic cache invalidation on writes
- TTL-based expiration for cache entries
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


class CacheKeyBuilder:
    """Centralized cache key construction for consistency."""
    
    @staticmethod
    def user_chats(user_id: int) -> str:
        return f"user:{user_id}:chats"
    
    @staticmethod
    def chat_messages(chat_id: str) -> str:
        return f"chat:{chat_id}:messages"
    
    @staticmethod
    def chat_recent_messages(chat_id: str) -> str:
        return f"chat:{chat_id}:recent"
    
    @staticmethod
    def chat_details(chat_id: str) -> str:
        return f"chat:{chat_id}:details"
    
    @staticmethod
    def user_by_username(username: str) -> str:
        return f"user:name:{username}"
    
    @staticmethod
    def user_by_email(email: str) -> str:
        return f"user:email:{email}"
    
    @staticmethod
    def user_by_id(user_id: int) -> str:
        return f"user:id:{user_id}"


class HybridStorage:
    """
    Hybrid storage combining SQLite (durability) and Redis (speed).
    
    Pattern:
    - Writes: Go to both SQLite and Redis (write-through)
    - Reads: Try Redis first, fallback to SQLite (read-through)
    - Invalidation: Delete cache on data changes
    
    Cache TTL defaults:
    - User data: 1 hour
    - Chat list: 5 minutes (changes frequently)
    - Messages: 10 minutes
    - Chat details: 30 minutes
    """
    
    # Cache TTL constants (in seconds)
    TTL_USER = 3600        # 1 hour
    TTL_CHAT_LIST = 300    # 5 minutes
    TTL_MESSAGES = 600     # 10 minutes
    TTL_CHAT_DETAILS = 1800  # 30 minutes
    
    def __init__(
        self, 
        db_path: str = "data/app.db",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 3600  # Default 1 hour
    ):
        self.db_path = Path(db_path)
        self.cache_ttl = cache_ttl
        self.keys = CacheKeyBuilder()
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize Redis connection
        self._init_redis(redis_host, redis_port)
    
    def _init_redis(self, host: str, port: int):
        """Initialize Redis connection with proper error handling."""
        try:
            self.redis = redis.Redis(
                host=host,
                port=port,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=True
            )
            self.redis.ping()
            self.redis_available = True
            logger.info(f"✅ Redis connected at {host}:{port}")
            
            # Log Redis info
            info = self.redis.info('memory')
            logger.info(f"   Redis memory: {info.get('used_memory_human', 'unknown')}")
            
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.redis_available = False
            self.redis = None
            logger.warning(f"⚠️  Redis unavailable: {e}. Running without cache.")
    
    @contextmanager
    def get_db(self):
        """Context manager for SQLite connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Return dict-like rows
        try:
            yield conn
        finally:
            conn.close()
    
    # =========================================================================
    # Redis Cache Helpers
    # =========================================================================
    
    def _cache_set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set a value in Redis cache with TTL."""
        if not self.redis_available:
            return False
        
        try:
            ttl = ttl or self.cache_ttl
            serialized = json.dumps(value, default=str)
            self.redis.setex(key, ttl, serialized)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache SET failed for {key}: {e}")
            return False
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """Get a value from Redis cache."""
        if not self.redis_available:
            return None
        
        try:
            cached = self.redis.get(key)
            if cached:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(cached)
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.warning(f"Cache GET failed for {key}: {e}")
            return None
    
    def _cache_delete(self, *keys: str) -> int:
        """Delete keys from Redis cache."""
        if not self.redis_available or not keys:
            return 0
        
        try:
            deleted = self.redis.delete(*keys)
            logger.debug(f"Cache DELETE: {keys} (deleted {deleted})")
            return deleted
        except Exception as e:
            logger.warning(f"Cache DELETE failed for {keys}: {e}")
            return 0
    
    def _cache_lpush(self, key: str, value: Any, max_items: int = 50) -> bool:
        """Push to a list in Redis and trim to max size."""
        if not self.redis_available:
            return False
        
        try:
            serialized = json.dumps(value, default=str)
            pipe = self.redis.pipeline()
            pipe.lpush(key, serialized)
            pipe.ltrim(key, 0, max_items - 1)
            pipe.expire(key, self.TTL_MESSAGES)
            pipe.execute()
            logger.debug(f"Cache LPUSH: {key}")
            return True
        except Exception as e:
            logger.warning(f"Cache LPUSH failed for {key}: {e}")
            return False
    
    def _cache_lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range from a list in Redis."""
        if not self.redis_available:
            return []
        
        try:
            items = self.redis.lrange(key, start, end)
            if items:
                logger.debug(f"Cache LRANGE HIT: {key} ({len(items)} items)")
                return [json.loads(item) for item in items]
            return []
        except Exception as e:
            logger.warning(f"Cache LRANGE failed for {key}: {e}")
            return []
    
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
            
            # Invalidate any cached lookups
            self._cache_delete(
                self.keys.user_by_username(username),
                self.keys.user_by_email(email)
            )
            
            return user_id
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username with caching."""
        cache_key = self.keys.user_by_username(username)
        
        # Try cache first
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        # Query database
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email, hashed_password, created_at, last_login, is_active FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            
            if row:
                user = dict(row)
                # Cache the result
                self._cache_set(cache_key, user, self.TTL_USER)
                return user
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email with caching."""
        cache_key = self.keys.user_by_email(email)
        
        # Try cache first
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        # Query database
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email, hashed_password, created_at, last_login, is_active FROM users WHERE email = ?",
                (email,)
            )
            row = cursor.fetchone()
            
            if row:
                user = dict(row)
                self._cache_set(cache_key, user, self.TTL_USER)
                return user
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID with caching."""
        cache_key = self.keys.user_by_id(user_id)
        
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email, created_at, last_login FROM users WHERE id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            
            if row:
                user = dict(row)
                self._cache_set(cache_key, user, self.TTL_USER)
                return user
            return None
    
    def update_last_login(self, user_id: int):
        """Update user's last login timestamp and invalidate cache."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,)
            )
            conn.commit()
        
        # Invalidate user caches since last_login changed
        self._cache_delete(self.keys.user_by_id(user_id))
        
        # Also need to invalidate username/email caches
        # Get user info first to know the keys
        user = self.get_user_by_id(user_id)
        if user:
            self._cache_delete(
                self.keys.user_by_username(user.get('username', '')),
                self.keys.user_by_email(user.get('email', ''))
            )
    
    # =========================================================================
    # Chat Management
    # =========================================================================
    
    def create_chat(self, user_id: int, title: str = "Νέα Συνομιλία") -> str:
        """Create a new chat for a user."""
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
        self._cache_delete(self.keys.user_chats(user_id))
        
        return chat_id
    
    def get_user_chats(self, user_id: int) -> List[Dict]:
        """Get all chats for a user with caching."""
        cache_key = self.keys.user_chats(user_id)
        
        # Try cache first
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        
        # Query database
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    c.id,
                    c.title,
                    c.created_at,
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
        self._cache_set(cache_key, chats, self.TTL_CHAT_LIST)
        
        return chats
    
    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get chat metadata with caching."""
        cache_key = self.keys.chat_details(chat_id)
        
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
            row = cursor.fetchone()
            
            if row:
                chat = dict(row)
                self._cache_set(cache_key, chat, self.TTL_CHAT_DETAILS)
                return chat
            return None
    
    def update_chat_title(self, chat_id: str, title: str):
        """Update chat title and invalidate caches."""
        # Get chat first to know user_id for cache invalidation
        chat = self.get_chat(chat_id)
        if not chat:
            return
        
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
        
        # Invalidate caches
        self._cache_delete(
            self.keys.chat_details(chat_id),
            self.keys.user_chats(chat['user_id'])
        )
    
    def delete_chat(self, chat_id: str):
        """Delete a chat and all its messages."""
        chat = self.get_chat(chat_id)
        if not chat:
            return
        
        with self.get_db() as conn:
            cursor = conn.cursor()
            # Messages are deleted via CASCADE
            cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.commit()
        
        # Invalidate all related caches
        self._cache_delete(
            self.keys.chat_details(chat_id),
            self.keys.chat_messages(chat_id),
            self.keys.chat_recent_messages(chat_id),
            self.keys.user_chats(chat['user_id'])
        )
        
        logger.info(f"🗑️  Chat deleted: {chat_id}")
    
    # =========================================================================
    # Message Management
    # =========================================================================
    
    def add_message(self, chat_id: str, role: str, content: str) -> int:
        """
        Add a message to a chat (write-through to both DB and cache).
        
        This is the key hybrid operation:
        1. Write to SQLite (durability)
        2. Update Redis recent messages list (speed)
        3. Invalidate full message cache
        """
        timestamp = datetime.utcnow().isoformat()
        
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
        
        # Update Redis cache (write-through pattern)
        if self.redis_available:
            message = {
                "id": message_id,
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "chat_id": chat_id
            }
            
            # Add to recent messages list (keep last 50)
            self._cache_lpush(
                self.keys.chat_recent_messages(chat_id),
                message,
                max_items=50
            )
            
            # Invalidate full message cache and chat details
            # (next read will rebuild from DB)
            self._cache_delete(
                self.keys.chat_messages(chat_id),
                self.keys.chat_details(chat_id)
            )
            
            # Also invalidate user's chat list since updated_at changed
            chat = self.get_chat(chat_id)
            if chat:
                self._cache_delete(self.keys.user_chats(chat['user_id']))
        
        logger.debug(f"💬 Message added to chat {chat_id}: {role}")
        return message_id
    
    def get_messages(self, chat_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get messages for a chat (read-through from cache or DB).
        
        For recent messages (limit <= 50), tries Redis list first.
        For all messages, queries SQLite directly.
        """
        # For small limits, try the recent messages cache first
        if limit and limit <= 50 and self.redis_available:
            cached = self._cache_lrange(
                self.keys.chat_recent_messages(chat_id),
                0, limit - 1
            )
            if cached:
                # Recent cache stores newest first, reverse for chronological
                return list(reversed(cached))
        
        # Cache miss or requesting all messages - query database
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            if limit:
                # Get most recent N messages in chronological order
                query = """
                    SELECT * FROM (
                        SELECT * FROM messages 
                        WHERE chat_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    ) ORDER BY timestamp ASC
                """
                cursor.execute(query, (chat_id, limit))
            else:
                query = "SELECT * FROM messages WHERE chat_id = ? ORDER BY timestamp ASC"
                cursor.execute(query, (chat_id,))
            
            rows = cursor.fetchall()
            messages = [dict(row) for row in rows]
        
        # Cache if we fetched all or a reasonable amount
        if not limit or limit > 50:
            self._cache_set(
                self.keys.chat_messages(chat_id),
                messages,
                self.TTL_MESSAGES
            )
        
        return messages
    
    # =========================================================================
    # Cache Statistics (for monitoring)
    # =========================================================================
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        if not self.redis_available:
            return {"available": False, "message": "Redis not connected"}
        
        try:
            info = self.redis.info()
            return {
                "available": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
                "total_keys": self.redis.dbsize(),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict) -> str:
        """Calculate cache hit rate percentage."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        if total == 0:
            return "N/A"
        return f"{(hits / total) * 100:.1f}%"
    
    def clear_all_cache(self):
        """Clear all cached data (useful for testing/debugging)."""
        if not self.redis_available:
            return
        
        try:
            self.redis.flushdb()
            logger.info("🗑️  All cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


# Global storage instance
storage = HybridStorage()
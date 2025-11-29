# app/db/init_db.py
"""
Database initialization and schema setup.
Creates tables for users and chats in SQLite.
"""

import sqlite3
from pathlib import Path
from typing import Optional
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatabaseInitializer:
    """Handles database schema creation and migrations."""
    
    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def initialize(self):
        """Create all required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Chats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chats_user_id 
                ON chats(user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_chat_id 
                ON messages(chat_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages(timestamp)
            """)
            
            conn.commit()
            logger.info(f"✅ Database initialized at {self.db_path}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Database initialization failed: {e}")
            raise
        finally:
            conn.close()
    
    def reset_database(self):
        """Drop all tables and recreate (USE WITH CAUTION)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DROP TABLE IF EXISTS messages")
            cursor.execute("DROP TABLE IF EXISTS chats")
            cursor.execute("DROP TABLE IF EXISTS users")
            conn.commit()
            logger.warning("🗑️  All tables dropped")
            
            # Recreate
            self.initialize()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Database reset failed: {e}")
            raise
        finally:
            conn.close()


def init_database(db_path: str = "data/app.db"):
    """Convenience function to initialize database."""
    initializer = DatabaseInitializer(db_path)
    initializer.initialize()


if __name__ == "__main__":
    # Run this script to create the database
    init_database()
    print("Database initialized successfully!")
# scripts/setup_auth.py
"""
Setup script for authentication system.
Initializes database and runs basic tests.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.init_db import init_database
from app.db.storage import storage
from app.auth.auth import register_user, login_user, AuthenticationError, RegistrationError
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def setup_database():
    """Initialize the database schema."""
    print("\n" + "="*60)
    print("STEP 1: Initializing Database")
    print("="*60)
    
    try:
        init_database()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    
    return True


def test_redis_connection():
    """Test Redis connection."""
    print("\n" + "="*60)
    print("STEP 2: Testing Redis Connection")
    print("="*60)
    
    if storage.redis_available:
        print("✅ Redis is connected and working!")
        print(f"   Host: localhost")
        print(f"   Cache TTL: {storage.cache_ttl} seconds")
    else:
        print("⚠️  Redis is not available")
        print("   System will work but without caching")
        print("   To enable Redis, run: sudo systemctl start redis")
    
    return True


def create_test_user():
    """Create a test user."""
    print("\n" + "="*60)
    print("STEP 3: Creating Test User")
    print("="*60)
    
    try:
        user = register_user(
            username="testuser",
            email="test@example.com",
            password="testpass123"
        )
        print(f"✅ Test user created!")
        print(f"   Username: testuser")
        print(f"   Email: test@example.com")
        print(f"   Password: testpass123")
        print(f"   User ID: {user['id']}")
        
    except RegistrationError as e:
        if "already" in str(e).lower():
            print("ℹ️  Test user already exists (this is fine)")
        else:
            print(f"❌ Failed to create test user: {e}")
            return False
    
    return True


def test_authentication():
    """Test login functionality."""
    print("\n" + "="*60)
    print("STEP 4: Testing Authentication")
    print("="*60)
    
    try:
        result = login_user("testuser", "testpass123")
        print("✅ Login successful!")
        print(f"   Token: {result['access_token'][:50]}...")
        print(f"   User: {result['user']['username']}")
        
    except AuthenticationError as e:
        print(f"❌ Login failed: {e}")
        return False
    
    return True


def test_chat_creation():
    """Test chat creation."""
    print("\n" + "="*60)
    print("STEP 5: Testing Chat Creation")
    print("="*60)
    
    try:
        # Get test user
        user = storage.get_user_by_username("testuser")
        
        # Create a test chat
        chat_id = storage.create_chat(user["id"], "Test Chat")
        print(f"✅ Chat created!")
        print(f"   Chat ID: {chat_id}")
        
        # Add test messages
        storage.add_message(chat_id, "user", "Hello!")
        storage.add_message(chat_id, "assistant", "Hi! How can I help you?")
        print("✅ Test messages added!")
        
        # Retrieve messages
        messages = storage.get_messages(chat_id)
        print(f"✅ Retrieved {len(messages)} messages")
        
        # Test chat listing
        chats = storage.get_user_chats(user["id"])
        print(f"✅ User has {len(chats)} chat(s)")
        
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False
    
    return True


def print_summary():
    """Print setup summary."""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\n📝 Next Steps:")
    print("\n1. Update your main.py to include the new routes:")
    print("   from app.api.auth_routes import router as auth_router")
    print("   from app.api.chat_routes_authenticated import router as chat_router")
    print("   app.include_router(auth_router)")
    print("   app.include_router(chat_router)")
    print("\n2. Start your server:")
    print("   python scripts/run.py --run")
    print("\n3. Test the API:")
    print("   - Visit http://localhost:8000/docs")
    print("   - Try /auth/register endpoint")
    print("   - Try /auth/login endpoint")
    print("   - Use the token in /chats endpoints")
    print("\n4. Test credentials:")
    print("   Username: testuser")
    print("   Password: testpass123")
    print("\n")


def main():
    """Run setup."""
    print("\n🚀 Authentication System Setup")
    print("="*60)
    
    steps = [
        setup_database,
        test_redis_connection,
        create_test_user,
        test_authentication,
        test_chat_creation
    ]
    
    for step in steps:
        if not step():
            print("\n❌ Setup failed! Fix the errors above and try again.")
            sys.exit(1)
    
    print_summary()


if __name__ == "__main__":
    main()
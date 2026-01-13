import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import init_db_engine, Base
from app.db.models import User
from app.config import PATHS

def reset_database():
    print("🛑 WARNING: This will DELETE all data in the database.")
    
    # FIX: Capture the engine returned by the function
    # Do not rely on the global variable import
    active_engine = init_db_engine()
    
    if not active_engine:
        print("❌ Error: Could not initialize database engine.")
        return

    # 1. Drop all tables
    print("   Dropping old tables...")
    Base.metadata.drop_all(bind=active_engine)
    
    # 2. Create new tables
    print("   Creating new tables...")
    Base.metadata.create_all(bind=active_engine)
    
    print("✅ Database successfully reset!")
    print(f"   Location: {os.path.join(PATHS.data_dir, 'app.db')}")

if __name__ == "__main__":
    reset_database()
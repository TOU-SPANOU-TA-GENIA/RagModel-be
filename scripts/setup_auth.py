import sys
import os
import getpass

# Add project root to Python path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to avoid 'NoneType' errors
import app.db.session as db_session
from app.db.models import User
from app.auth.hashing import get_password_hash

def main():
    print("--- RagModel User Setup (Database Mode) ---")
    
    # 1. Initialize Database Connection
    db_session.init_db_engine()
    
    if db_session.SessionLocal is None:
        print("❌ Error: Could not initialize database connection.")
        return

    db = db_session.SessionLocal()

    try:
        # 2. Get User Input
        username = input("Enter username: ").strip()
        if not username:
            print("❌ Username cannot be empty.")
            return

        password = getpass.getpass("Enter password: ").strip()
        confirm = getpass.getpass("Confirm password: ").strip()

        if password != confirm:
            print("❌ Passwords do not match.")
            return
        
        if not password:
            print("❌ Password cannot be empty.")
            return

        full_name = input("Enter full name (optional): ").strip()

        # 3. Check for Existing User
        existing_user = db.query(User).filter(User.username == username).first()

        if existing_user:
            print(f"⚠️  User '{username}' already exists.")
            action = input("Do you want to update the password? (y/n): ").lower()
            if action == 'y':
                existing_user.hashed_password = get_password_hash(password)
                if full_name:
                    existing_user.full_name = full_name
                db.commit()
                print(f"✅ User '{username}' updated successfully.")
            else:
                print("Operation cancelled.")
        else:
            # 4. Create New User
            new_user = User(
                username=username,
                hashed_password=get_password_hash(password),
                full_name=full_name if full_name else None,
                disabled=False,
                email=None 
            )
            db.add(new_user)
            db.commit()
            print(f"✅ User '{username}' created successfully.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
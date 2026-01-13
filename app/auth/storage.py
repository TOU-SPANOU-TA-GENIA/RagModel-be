from typing import Optional
# FIX: Import the module, not the variable, to see updates
import app.db.session as db_session 
from app.db.models import User as UserModel
from app.auth.schemas import UserInDB

def get_user_by_username(username: str) -> Optional[UserInDB]:
    """
    Fetches user credentials directly from the SQLite database.
    """
    # Safety Check: If the session factory isn't ready, initialize it now.
    if db_session.SessionLocal is None:
        db_session.init_db_engine()
        
    # Now access the factory through the module
    db = db_session.SessionLocal()
    
    try:
        user = db.query(UserModel).filter(UserModel.username == username).first()
        
        if user:
            return UserInDB(
                username=user.username,
                hashed_password=user.hashed_password,
                email=user.email,
                full_name=user.full_name,
                disabled=user.disabled
            )
        return None
    except Exception as e:
        print(f"Database Auth Error: {e}")
        return None
    finally:
        db.close()
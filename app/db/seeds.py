from sqlalchemy.orm import Session
from app.db.models import User
from app.config import get_config
from app.auth.hashing import get_password_hash
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def seed_users(db: Session):
    """
    Syncs users from config.json into the SQLite database.
    Ensures at least one user exists.
    """
    config = get_config()
    security_conf = config.get("security", {})
    config_users = security_conf.get("users", [])

    # 1. Sync Config Users to DB
    for u_data in config_users:
        username = u_data.get("username")
        if not username:
            continue

        # Check if user exists in DB
        existing_user = db.query(User).filter(User.username == username).first()
        
        if not existing_user:
            logger.info(f"Seeding user from config: {username}")
            # Use hash from config, or generate default if missing (safety net)
            pwd_hash = u_data.get("hashed_password")
            if not pwd_hash:
                # Fallback if config has user but no hash (rare)
                logger.warning(f"User {username} in config has no hash. Setting default password 'admin'.")
                pwd_hash = get_password_hash("admin")

            new_user = User(
                username=username,
                hashed_password=pwd_hash,
                full_name=u_data.get("full_name"),
                email=u_data.get("email"),
                disabled=u_data.get("disabled", False)
            )
            db.add(new_user)
    
    db.commit()

    # 2. Final Safety Check: Ensure AT LEAST ONE user exists
    count = db.query(User).count()
    if count == 0:
        logger.warning("No users found in Config or DB. Creating default 'admin' user.")
        default_admin = User(
            username="admin",
            hashed_password=get_password_hash("admin"), # Default password
            full_name="System Administrator",
            disabled=False
        )
        db.add(default_admin)
        db.commit()
        logger.info("Default admin created. (User: admin / Pass: admin)")
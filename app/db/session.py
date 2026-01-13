from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base # Updated for SQLAlchemy 2.0+
from app.config import get_config, PATHS
import os

# Create the base class for models
Base = declarative_base()

# Global engine variable
engine = None
SessionLocal = None

def init_db_engine():
    global engine, SessionLocal
    
    # Get config
    config = get_config()
    db_config = config.get("database", {})
    
    # Determine URL (Default to SQLite in data_dir)
    db_url = db_config.get("url")
    if not db_url:
        # Ensure data dir exists
        os.makedirs(PATHS.data_dir, exist_ok=True)
        db_path = os.path.join(PATHS.data_dir, "app.db")
        db_url = f"sqlite:///{db_path}"

    connect_args = {}
    if "sqlite" in db_url:
        connect_args = {"check_same_thread": False}

    engine = create_engine(
        db_url, 
        connect_args=connect_args,
        pool_pre_ping=True
    )
    
    # Create the Session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Return engine in case needed immediately
    return engine

# Helper to get a session (Dependency Injection)
def get_db():
    if SessionLocal is None:
        init_db_engine()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
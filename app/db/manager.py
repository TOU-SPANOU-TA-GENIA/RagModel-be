import app.db.session as db_session  # Import module, not variable
from app.db.models import User

class DatabaseManager:
    """
    Utilities for managing database state (tables, migrations, etc).
    """
    
    @staticmethod
    def create_tables():
        """
        Creates all tables defined in SQLAlchemy models.
        """
        # Access the engine dynamically from the module
        engine = db_session.engine
        
        if engine is None:
            # Fallback: Try to init if missing (safety net)
            print("Warning: DB Engine was None, initializing lazy...")
            db_session.init_db_engine()
            engine = db_session.engine

        if engine is None:
             raise RuntimeError("Database engine could not be initialized.")

        # Access Base from the module to ensure it's the shared instance
        db_session.Base.metadata.create_all(bind=engine)
        print("Database tables created/verified.")

    @staticmethod
    def drop_tables():
        """
        Drops all tables. Use with caution.
        """
        engine = db_session.engine
        if engine:
            db_session.Base.metadata.drop_all(bind=engine)
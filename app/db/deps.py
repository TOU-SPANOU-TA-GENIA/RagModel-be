from typing import Generator
from sqlalchemy.orm import Session
from app.db.session import DatabaseEngine

def get_db() -> Generator[Session, None, None]:
    """
    Dependency to yield a database session per request.
    Closes the session automatically after the request.
    """
    session = DatabaseEngine.get_session()
    try:
        yield session
    finally:
        session.close()
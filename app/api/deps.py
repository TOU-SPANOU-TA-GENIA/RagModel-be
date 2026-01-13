from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

# Re-export core dependencies from their new homes
from app.db.session import get_db
from app.auth.service import get_current_user
from app.db.models import User

# If any file needs the LLM service, they should import 'llm_service' from app.llm
# We do NOT import provider classes here to avoid circular imports.

__all__ = ["get_db", "get_current_user", "User"]
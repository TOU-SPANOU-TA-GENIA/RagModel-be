from typing import Any
from .service import authenticate_user
from .tokens import create_access_token, decode_access_token
from .hashing import get_password_hash
from .storage import get_user_by_username
from app.config import get_config

# 1. Define Constants Explicitly
# We fetch these immediately to ensure they are never None when imported
_security_conf = get_config().get("security", {})

SECRET_KEY = _security_conf.get("secret_key", "dev_secret_key_CHANGE_ME_IN_PROD")
ALGORITHM = _security_conf.get("algorithm", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = _security_conf.get("access_token_expire_minutes", 30)

__all__ = [
    "authenticate_user",
    "create_access_token",
    "decode_access_token",
    "get_password_hash",
    "get_user_by_username",
    "SECRET_KEY",
    "ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_MINUTES"
]
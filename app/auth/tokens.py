from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import jwt, JWTError
from app.config import get_config  # Direct config import

# Load configuration directly here to avoid circular imports with app.auth
_security_conf = get_config().get("security", {})
SECRET_KEY = _security_conf.get("secret_key", "dev_secret_key_CHANGE_ME_IN_PROD")
ALGORITHM = _security_conf.get("algorithm", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = _security_conf.get("access_token_expire_minutes", 10080) # 10080 mins = 7 days

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a JWT access token.
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
    to_encode.update({"exp": expire})
    
    # Now ALGORITHM is guaranteed to be "HS256" (or config value), never None
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[Dict]:
    """
    Decodes and validates a JWT token.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
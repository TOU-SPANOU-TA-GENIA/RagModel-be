# app/auth/auth.py
"""
Authentication and authorization logic.
Handles user registration, login, and JWT token generation.
"""

from datetime import datetime, timedelta
from typing import Optional
import bcrypt
from jose import JWTError, jwt

from app.db.storage import storage
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"  # CHANGE THIS!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class RegistrationError(Exception):
    """Raised when registration fails."""
    pass


# =============================================================================
# Password Utilities (Direct bcrypt usage)
# =============================================================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt directly."""
    # Ensure password is bytes and handle length
    password_bytes = password.encode('utf-8')
    
    # Bcrypt has a 72 byte limit, truncate if necessary
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    # Return as string
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    # Convert to bytes
    password_bytes = plain_password.encode('utf-8')
    
    # Truncate to 72 bytes if necessary
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    
    hashed_bytes = hashed_password.encode('utf-8')
    
    # Verify
    return bcrypt.checkpw(password_bytes, hashed_bytes)


# =============================================================================
# JWT Token Management
# =============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {e}")


# =============================================================================
# User Authentication
# =============================================================================

def register_user(username: str, email: str, password: str) -> dict:
    """
    Register a new user.
    
    Returns:
        User dict with id, username, email
    
    Raises:
        RegistrationError if username/email already exists
    """
    # Validate input
    if len(username) < 3:
        raise RegistrationError("Username must be at least 3 characters")
    
    if len(password) < 6:
        raise RegistrationError("Password must be at least 6 characters")
    
    if len(password) > 72:
        raise RegistrationError("Password cannot be longer than 72 characters")
    
    # Check if user already exists
    existing_user = storage.get_user_by_username(username)
    if existing_user:
        raise RegistrationError("Username already taken")
    
    existing_email = storage.get_user_by_email(email)
    if existing_email:
        raise RegistrationError("Email already registered")
    
    # Hash password and create user
    hashed_password = hash_password(password)
    user_id = storage.create_user(username, email, hashed_password)
    
    logger.info(f"✅ User registered: {username}")
    
    return {
        "id": user_id,
        "username": username,
        "email": email
    }


def authenticate_user(username: str, password: str) -> dict:
    """
    Authenticate a user and return user info.
    
    Returns:
        User dict if authentication successful
    
    Raises:
        AuthenticationError if credentials are invalid
    """
    # Get user from database
    user = storage.get_user_by_username(username)
    if not user:
        raise AuthenticationError("Invalid username or password")
    
    # Verify password
    if not verify_password(password, user["hashed_password"]):
        raise AuthenticationError("Invalid username or password")
    
    # Check if user is active
    if not user.get("is_active", True):
        raise AuthenticationError("Account is disabled")
    
    # Update last login
    storage.update_last_login(user["id"])
    
    logger.info(f"✅ User authenticated: {username}")
    
    # Return user info (without password hash)
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "created_at": user["created_at"],
        "last_login": user["last_login"]
    }


def login_user(username: str, password: str) -> dict:
    """
    Authenticate user and create access token.
    
    Returns:
        Dict with access_token and user info
    """
    user = authenticate_user(username, password)
    
    # Create access token
    token_data = {
        "sub": user["username"],
        "user_id": user["id"]
    }
    access_token = create_access_token(token_data)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }


def get_current_user(token: str) -> dict:
    """
    Get current user from JWT token.
    
    Returns:
        User dict
    
    Raises:
        AuthenticationError if token is invalid
    """
    payload = decode_access_token(token)
    
    username = payload.get("sub")
    if not username:
        raise AuthenticationError("Invalid token payload")
    
    user = storage.get_user_by_username(username)
    if not user:
        raise AuthenticationError("User not found")
    
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"]
    }
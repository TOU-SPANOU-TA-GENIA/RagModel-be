# app/api/auth_routes.py
"""
Authentication API endpoints.
Provides /register, /login, and user management routes.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

from app.auth.auth import (
    register_user,
    login_user,
    get_current_user,
    AuthenticationError,
    RegistrationError
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


# =============================================================================
# Request/Response Models
# =============================================================================

class RegisterRequest(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    """User login request."""
    username: str
    password: str


class UserResponse(BaseModel):
    """User information response."""
    id: int
    username: str
    email: str
    created_at: Optional[str] = None
    last_login: Optional[str] = None


class LoginResponse(BaseModel):
    """Login response with token."""
    access_token: str
    token_type: str
    user: UserResponse


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    success: bool = True


# =============================================================================
# Dependency: Get Current User from Token
# =============================================================================

async def get_current_user_dep(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    FastAPI dependency to extract and validate current user from JWT token.
    
    Usage in routes:
        @router.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user_dep)):
            return {"message": f"Hello {user['username']}"}
    """
    try:
        token = credentials.credentials
        user = get_current_user(token)
        return user
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )


# =============================================================================
# Authentication Endpoints
# =============================================================================

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Register a new user account.
    
    **Request body:**
    - username: 3-50 characters
    - email: Valid email address
    - password: Minimum 6 characters
    
    **Returns:**
    - User information (without password)
    
    **Errors:**
    - 400: Username/email already exists or validation failed
    """
    try:
        user = register_user(
            username=request.username,
            email=request.email,
            password=request.password
        )
        logger.info(f"📝 New user registered: {request.username}")
        return user
    
    except RegistrationError as e:
        logger.warning(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Login with username and password.
    
    **Request body:**
    - username: Your username
    - password: Your password
    
    **Returns:**
    - access_token: JWT token (include in Authorization header as "Bearer <token>")
    - token_type: "bearer"
    - user: User information
    
    **Errors:**
    - 401: Invalid credentials
    """
    try:
        result = login_user(
            username=request.username,
            password=request.password
        )
        logger.info(f"🔐 User logged in: {request.username}")
        return result
    
    except AuthenticationError as e:
        logger.warning(f"Login failed for {request.username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_me(user: dict = Depends(get_current_user_dep)):
    """
    Get current user information from token.
    
    **Headers:**
    - Authorization: Bearer <access_token>
    
    **Returns:**
    - Current user information
    """
    return user


@router.post("/logout", response_model=MessageResponse)
async def logout(user: dict = Depends(get_current_user_dep)):
    """
    Logout current user.
    
    Note: Since we're using stateless JWT tokens, logout is handled
    client-side by deleting the token. This endpoint exists for
    potential future server-side session management.
    """
    logger.info(f"👋 User logged out: {user['username']}")
    return {
        "message": "Logged out successfully",
        "success": True
    }


# =============================================================================
# Health Check (for testing auth setup)
# =============================================================================

@router.get("/health")
async def auth_health():
    """Check if authentication system is working."""
    return {
        "status": "healthy",
        "auth_enabled": True
    }
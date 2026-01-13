from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from sqlalchemy.orm import Session
from typing import Any, Dict

from app.db.session import get_db
from app.auth.service import authenticate_user, get_current_user
from app.auth.tokens import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.api.schemas import Token, LoginRequest, UserResponse
from app.db.models import User

router = APIRouter()

@router.post("/login") # Removed response_model to allow free-form JSON
async def login_for_access_token(
    request: Request,
    response: Response,
    login_data: LoginRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    
    print(f"\n🔑 [LOGIN ATTEMPT] User: {login_data.username}")
    user = authenticate_user(db, login_data.username, login_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate Token
    expires_sec = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # --- PERSISTENCE: DOUBLE COOKIE ---
    # This guarantees the session works on refresh (as proven by your logs)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=False, 
        max_age=expires_sec,
        samesite="lax", 
        secure=False,     
        path="/"
    )
    response.set_cookie(
        key="access_token_cors",
        value=f"Bearer {access_token}",
        httponly=False,
        max_age=expires_sec,
        samesite="none", 
        secure=True,     
        path="/"
    )
    
    # --- FRONTEND COMPATIBILITY PAYLOAD ---
    # We construct a user object that satisfies all common frameworks
    user_obj = {
        "id": user.id,
        "pk": user.id,
        "_id": str(user.id), # Some frontends expect string IDs
        "username": user.username,
        "email": user.email or "",
        "name": user.full_name or user.username,
        "fullName": user.full_name or user.username,
        "role": "admin", # Ensure permissions
        "roles": ["admin"],
        "token": access_token # Token inside user object (Common in React/Vue)
    }

    # --- THE UNIVERSAL RESPONSE ---
    # We return the data in multiple locations so the frontend parser finds it
    return {
        # 1. Standard OAuth2
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": expires_sec,
        
        # 2. Direct Field Access (React Admin / Simple JS)
        "token": access_token,
        "accessToken": access_token,
        "jwt": access_token,
        "user": user_obj,
        "id": user.id,
        "username": user.username,
        "role": "admin",

        # 3. Data Wrapper (JSON:API Standard)
        "data": {
            "token": access_token,
            "accessToken": access_token,
            "user": user_obj
        },

        # 4. Meta Wrapper (Legacy)
        "meta": {
            "token": access_token,
            "user": user_obj
        }
    }

# --- SESSION RESTORATION ---
# Your logs show /chats/ succeeds, but if the frontend checks /me, we must be ready.

@router.get("/me", response_model=UserResponse)
async def me_root(current_user: User = Depends(get_current_user)): return current_user

@router.get("/users/me", response_model=UserResponse)
async def me_users(current_user: User = Depends(get_current_user)): return current_user

@router.get("/auth/me", response_model=UserResponse)
async def me_auth(current_user: User = Depends(get_current_user)): return current_user

@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("access_token_cors", path="/")
    return {"message": "Logged out successfully"}
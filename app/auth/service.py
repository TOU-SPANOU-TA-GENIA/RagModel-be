from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional

from app.db.session import get_db
from app.db.models import User
from app.auth.hashing import verify_password
from app.auth.tokens import decode_access_token

# Define Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme), 
    db: Session = Depends(get_db)
) -> User:
    
    # 1. Try Standard Cookie
    cookie_token = request.cookies.get("access_token")
    
    # 2. Try Cross-Site Cookie (Backup)
    if not cookie_token:
        cookie_token = request.cookies.get("access_token_cors")

    # 3. Use Cookie if Header is missing
    if not token and cookie_token:
        token = cookie_token
        if token.startswith("Bearer "):
            token = token.split(" ")[1]
            
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        # Diagnostic print to help if it fails again
        print(f"⛔ AUTH FAIL: No token. Headers: {len(request.headers)}, Cookies: {list(request.cookies.keys())}")
        raise credentials_exception

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
        
    return user
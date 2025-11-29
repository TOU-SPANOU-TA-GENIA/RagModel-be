# app/main_with_auth.py
"""
FastAPI application with authentication enabled.
This shows how to integrate the auth system into your existing app.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_routes import router as auth_router
from app.api.chat_routes_authenticated import router as chat_router
from app.db.init_db import init_database
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize database on startup
init_database()

# Create FastAPI app
app = FastAPI(
    title="RagModel-be with Authentication",
    description="AI Agent API with user authentication and persistent chat storage",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(chat_router)

# Add your existing routes here
# from app.api.other_routes import router as other_router
# app.include_router(other_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RagModel-be API with Authentication",
        "version": "2.0.0",
        "endpoints": {
            "auth": "/auth",
            "chats": "/chats",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.db.storage import storage
    
    return {
        "status": "healthy",
        "database": "sqlite",
        "redis_available": storage.redis_available
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
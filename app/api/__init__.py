from fastapi import APIRouter
from app.api.routers import agent, auth, config, files, workflows, chats

# Create the main API router
api_router = APIRouter()

# Register sub-routers with standard prefixes and tags
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(config.router, prefix="/config", tags=["config"])
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(chats.router, prefix="/chats", tags=["Chats"]) # Add this line

__all__ = ["api_router"]
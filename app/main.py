import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.config import get_config, SERVER, PATHS
from app.utils.logger import setup_logger
from app.db.session import init_db_engine
from app.db.manager import DatabaseManager
from app.db.seeds import seed_users
from app.api import api_router
from app.services.monitor import monitor_service
from app.llm import llm_service
from app.diagnostics.service import diagnostic_service
import app.rag

# Import Routers
from app.api.routers import auth, chats, agent 

logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... (Keep existing lifespan logic) ...
    logger.info("🚀 Starting RagModel Agent System...")
    init_db_engine()
    if get_config().get("database", {}).get("auto_create_tables", True):
        DatabaseManager.create_tables()

    try:
        from app.db.session import SessionLocal 
        db = SessionLocal()
        seed_users(db)
        db.close()
        logger.info("✅ Auth Database synced.")
    except Exception as e:
        logger.error(f"❌ Database seeding warning: {e}")

    logger.info("⏳ Loading AI Model...")
    llm_service.initialize()
    logger.info("✅ AI Model loaded.")

    monitor_service.start()
    health = diagnostic_service.run_diagnostics()
    if health.overall_status != "pass":
        logger.warning(f"System Check Status: {health.overall_status}")
    
    logger.info("✅ System initialized and ready to serve.")
    yield
    logger.info("🛑 Shutting down...")
    monitor_service.stop()

def create_app() -> FastAPI:
    app_config = get_config()
    server_conf = app_config.get("server", {})
    
    app = FastAPI(
        title="RagModel AI Agent",
        version="2.0.0",
        lifespan=lifespan
    )

    # --- FINAL CORRECT CORS SETUP ---
    # We explicitly list your frontend URL. 
    # This ensures standard FastAPI handling of the OPTIONS request (fixing the 405).
    origins = [
        "http://localhost:4200",    # Your Angular/Frontend port
        "http://127.0.0.1:4200",    # Alternative
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,   # Explicit list is safer than regex for credentials
        allow_credentials=True,  # Allow Cookies
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # --------------------------------

    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(chats.router, prefix="/chats", tags=["Chats"])
    app.include_router(agent.router, prefix="/stream", tags=["Agent Stream"])
    app.include_router(api_router, prefix="/api")
    
    os.makedirs(PATHS.outputs_dir, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=PATHS.outputs_dir), name="outputs")

    @app.get("/")
    async def root():
        return {"status": "online", "version": "2.0.0"}

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=SERVER.host, port=SERVER.port, reload=False)
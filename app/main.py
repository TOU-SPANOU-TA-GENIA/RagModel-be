# app/main.py
"""
FastAPI application entry point with authentication and network filesystem support.
Integrates all routes, middleware, and lifecycle management.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
import mimetypes

# Import API routers
from app.api import config_router
from app.api.auth_routes import router as auth_router
from app.api.chat_routes_authenticated import router as chat_router

# Import models
from app.api import (
    NewChatRequest, MessageRequest,
    ChatSummary, ChatDetail, SourceDocument,
    AgentResponse, UploadResponse, IngestionResponse,
    HealthResponse, StatsResponse
)

# Import configuration
from app.config import (
    SERVER, PATHS, AGENT,
    SYSTEM_INSTRUCTION, config
)

# Import core modules
from app.core import (
    startup_manager,
    ChatNotFoundException,
    RAGException
)

# Import database initialization
from app.db.init_db import init_database
from app.db.storage import storage

# Import utilities
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - startup and shutdown."""
    logger.info("🚀 Starting application...")
    
    # Startup
    try:
        # Initialize database
        logger.info("Initializing database...")
        init_database()
        
        # ========== NETWORK FILESYSTEM INITIALIZATION ==========
        logger.info("Initializing network filesystem...")
        try:
            from app.core.network_filesystem import initialize_network_monitor, NetworkShare
            from app.core.network_rag_integration import initialize_network_rag
            from app.rag.ingestion import ingest_file
            from pathlib import Path
            
            # Get network filesystem configuration
            network_config = config.get_section('network_filesystem')
            
            if network_config and network_config.get('enabled'):
                shares = []
                
                # Parse shares from config
                for share_data in network_config.get('shares', []):
                    if share_data.get('enabled'):
                        share = NetworkShare(
                            name=share_data['name'],
                            mount_path=Path(share_data['mount_path']),
                            share_type=share_data.get('share_type', 'smb'),
                            enabled=True,
                            auto_index=share_data.get('auto_index', True),
                            watch_for_changes=share_data.get('watch_for_changes', True),
                            scan_interval=share_data.get('scan_interval', 300),
                            include_extensions=share_data.get('include_extensions', []),
                            exclude_patterns=share_data.get('exclude_patterns', []),
                            max_file_size_mb=share_data.get('max_file_size_mb', 100)
                        )
                        shares.append(share)
                        logger.info(f"  📁 Configured share: {share.name} at {share.mount_path}")
                
                if shares:
                    # Initialize monitoring
                    monitor = initialize_network_monitor(shares)
                    logger.info(f"✅ Network monitoring initialized for {len(shares)} shares")
                    
                    # Initialize RAG integration
                    if network_config.get('auto_start_monitoring', True):
                        integrator = initialize_network_rag(monitor, ingest_file)
                        integrator.start()
                        logger.info("✅ Network-RAG integration started")
                        
                        # Trigger initial scan
                        logger.info("🔍 Starting initial file discovery...")
                        monitor.scan_all_shares()
                        stats = monitor.get_stats()
                        logger.info(f"📊 Discovery complete: {stats['total_files']} files found")
                        
                        # Show files by share
                        for share_name, count in stats['by_share'].items():
                            logger.info(f"   - {share_name}: {count} files")
                    else:
                        logger.info("Auto-start monitoring disabled")
                else:
                    logger.warning("No enabled network shares configured")
            else:
                logger.info("Network filesystem disabled in configuration")
        
        except Exception as e:
            logger.error(f"Network filesystem initialization failed: {e}")
            import traceback
            traceback.print_exc()
        # ========== END NETWORK FILESYSTEM INITIALIZATION ==========
        
        # Initialize system (RAG, models, etc.)
        logger.info("Initializing system components...")
        await startup_manager.initialize_system()
        
        logger.info("✅ Application ready")
        from pathlib import Path
        db_path = Path(PATHS.data_dir) / 'app.db'
        logger.info(f"   Database: SQLite at {db_path}")
        logger.info(f"   Redis: {'✅ Connected' if storage.redis_available else '⚠️  Not available (running without cache)'}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        # Stop network monitoring
        logger.info("Stopping network monitoring...")
        try:
            from app.core.network_rag_integration import get_network_integrator
            integrator = get_network_integrator()
            if integrator:
                integrator.stop()
                logger.info("Network monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping network monitoring: {e}")
        
        # Save configuration
        from app.config import config_manager
        config_manager.save()
        logger.info("Configuration saved")
    except Exception as e:
        logger.warning(f"Shutdown cleanup failed: {e}")
    
    logger.info("🛑 Shutting down...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="RagModel-be with Authentication",
    description="AI Agent API with RAG capabilities, user authentication, and persistent chat storage",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================================================================
# CORS Middleware
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=SERVER.cors_origins,  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Include Routers
# =============================================================================

# Authentication routes
app.include_router(auth_router)

# Chat routes (authenticated)
app.include_router(chat_router)

# Configuration routes
app.include_router(config_router)


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(ChatNotFoundException)
async def chat_not_found_handler(request, exc):
    """Handle chat not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Chat not found", "detail": str(exc)}
    )


@app.exception_handler(RAGException)
async def rag_exception_handler(request, exc):
    """Handle RAG system errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "System error", "detail": str(exc)}
    )


# =============================================================================
# System Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "RagModel-be API",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "authentication": True,
            "chat_persistence": True,
            "rag": True,
            "file_operations": True,
            "network_filesystem": True
        },
        "endpoints": {
            "auth": "/auth",
            "chats": "/chats",
            "config": "/config",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from app.db.storage import storage
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "sqlite",
            "database_available": True,
            "redis_cache": storage.redis_available,
            "agent": AGENT.mode
        }
    }
    
    # Check if we can access the database
    try:
        import sqlite3
        from pathlib import Path
        db_path = Path(PATHS.data_dir) / "app.db"
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        health_status["services"]["database_status"] = "operational"
    except Exception as e:
        health_status["services"]["database_status"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check network filesystem status
    try:
        from app.core.network_rag_integration import get_network_integrator
        integrator = get_network_integrator()
        if integrator:
            nfs_status = integrator.get_status()
            health_status["services"]["network_filesystem"] = {
                "enabled": True,
                "total_files": nfs_status.get("total_network_files", 0),
                "indexed_files": nfs_status.get("indexed_files", 0),
                "pending_files": nfs_status.get("pending_files", 0)
            }
        else:
            health_status["services"]["network_filesystem"] = {"enabled": False}
    except Exception as e:
        health_status["services"]["network_filesystem"] = {"enabled": False, "error": str(e)}
    
    return health_status


@app.get("/startup-status")
async def get_startup_status():
    """Check startup status and system readiness."""
    status_info = startup_manager.get_status()
    
    from pathlib import Path
    db_path = Path(PATHS.data_dir) / "app.db"
    
    return {
        "startup_complete": status_info.get("complete", False),
        "components": status_info.get("components", {}),
        "errors": status_info.get("errors", []),
        "auth_enabled": True,
        "database": {
            "type": "sqlite",
            "path": str(db_path),
            "cache": "redis" if storage.redis_available else "none"
        }
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    from pathlib import Path
    
    stats = {
        "uptime_seconds": None,
        "total_chats": 0,
        "total_messages": 0,
        "storage": {
            "database": "sqlite",
            "cache": "redis" if storage.redis_available else "none"
        }
    }
    
    # Get startup time
    if startup_manager.start_time:
        import time
        stats["uptime_seconds"] = int(time.time() - startup_manager.start_time)
    
    # Get chat/message counts from database
    try:
        import sqlite3
        db_path = Path(PATHS.data_dir) / "app.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count chats
        cursor.execute("SELECT COUNT(*) FROM chats")
        stats["total_chats"] = cursor.fetchone()[0]
        
        # Count messages
        cursor.execute("SELECT COUNT(*) FROM messages")
        stats["total_messages"] = cursor.fetchone()[0]
        
        conn.close()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
    
    # Get network filesystem stats
    try:
        from app.core.network_rag_integration import get_network_integrator
        integrator = get_network_integrator()
        if integrator:
            nfs_status = integrator.get_status()
            stats["network_filesystem"] = nfs_status
    except Exception:
        pass
    
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
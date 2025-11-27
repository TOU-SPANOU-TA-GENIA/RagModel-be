# app/core/startup.py
"""
Application startup and initialization.
"""

import time
import asyncio
from typing import Dict, Any

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class StartupManager:
    """Manages application startup and component initialization."""
    
    def __init__(self):
        self.start_time = None
        self.components_loaded = {}
        self._is_ready = False
    
    async def initialize_system(self):
        """Initialize system with pre-warming."""
        self.start_time = time.time()
        logger.info("🚀 Starting system initialization...")
        
        try:
            # Start LLM pre-warming
            self._start_llm_prewarming()
            
            # Initialize core components
            await self._initialize_core_components()
            
            self._is_ready = True
            total_time = time.time() - self.start_time
            logger.info(f"✅ Server ready in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    def _start_llm_prewarming(self):
        """Start LLM pre-warming in background."""
        logger.info("🔥 Starting LLM pre-warming...")
        try:
            from app.llm.prewarmed_provider import prewarmed_llm
            prewarmed_llm.initialize()
        except ImportError:
            logger.warning("Pre-warmed LLM provider not available")
    
    async def _initialize_core_components(self):
        """Initialize core components."""
        logger.info("⚡ Initializing core components...")
        
        try:
            from app.config import AGENT
            from app.agent.integration import create_agent
            
            agent = create_agent(mode=AGENT.mode)
            self.components_loaded['agent'] = True
            logger.info("✅ Agent initialized")
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            self.components_loaded['agent'] = False
    
    def is_ready(self) -> bool:
        """Check if system is ready."""
        return self._is_ready
    
    def get_status(self) -> Dict[str, Any]:
        """Get startup status."""
        llm_ready = False
        try:
            from app.llm.prewarmed_provider import prewarmed_llm
            llm_ready = prewarmed_llm.is_ready()
        except ImportError:
            pass
        
        return {
            "ready": self._is_ready,
            "llm_status": "ready" if llm_ready else "pre-warming",
            "startup_time": time.time() - self.start_time if self.start_time else None,
            "components": self.components_loaded,
        }


# Global startup manager
startup_manager = StartupManager()
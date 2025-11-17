# app/startup.py (REPLACE with optimized version)
"""
Ultra-fast startup with pre-warmed LLM.
"""

import time
import asyncio
import threading
from typing import Dict, Any

from app.logger import setup_logger
from app.config import AGENT_MODE, KNOWLEDGE_DIR
from app.agent.integration import create_agent, agent_manager
from app.core.memory_store import memory_db
from app.llm.prewarmed_provider import prewarmed_llm
from app.rag.ingestion import ingest_directory

logger = setup_logger(__name__)

class UltraFastStartupManager:
    """Startup manager optimized for first-response speed."""
    
    def __init__(self):
        self.start_time = None
        self.components_loaded = {}
        self._is_ready = False
    
    async def initialize_system_ultra_fast(self):
        """Ultra-fast initialization focusing on first-response speed."""
        self.start_time = time.time()
        logger.info("🚀 Starting ULTRA-FAST system initialization...")
        
        try:
            # Step 1: Start LLM pre-warming IMMEDIATELY (most important)
            self._start_llm_prewarming()
            
            # Step 2: Quick core initialization
            await self._initialize_core_components_fast()
            
            # Step 3: Mark as ready
            self._is_ready = True
            total_time = time.time() - self.start_time
            logger.info(f"✅ Server ready in {total_time:.2f}s - LLM pre-warming in background")
            
        except Exception as e:
            logger.error(f"❌ Ultra-fast initialization failed: {e}")
            raise
    
    def _start_llm_prewarming(self):
        """Start LLM pre-warming in background immediately."""
        logger.info("🔥 Starting LLM pre-warming...")
        prewarmed_llm.initialize()
    
    async def _initialize_core_components_fast(self):
        """Initialize only what's needed for first response."""
        logger.info("⚡ Initializing core components (ultra-fast)...")
        
        # 1. Create agent with fast LLM provider
        await self._initialize_agent_ultra_fast()
        
        # 2. Ensure knowledge base is ready
        if memory_db.documents:
            logger.info(f"📚 Knowledge base ready: {len(memory_db.documents)} docs")
        else:
            logger.warning("⚠️ No documents in knowledge base")
        
        logger.info("✅ Core components ready")
    
    async def _initialize_agent_ultra_fast(self):
        """Initialize agent optimized for speed."""
        logger.info("🤖 Initializing agent (ultra-fast)...")
        
        # Create agent - this should be fast without model loading
        agent = create_agent(mode=AGENT_MODE)
        logger.info("✅ Agent framework ready")
    
    def is_ready(self) -> bool:
        return self._is_ready
    
    def get_status(self) -> Dict[str, Any]:
        """Get startup status with LLM pre-warm status."""
        llm_status = "ready" if prewarmed_llm.is_ready() else "pre-warming"
        
        return {
            "ready": self._is_ready,
            "llm_status": llm_status,
            "startup_time": time.time() - self.start_time if self.start_time else None,
            "components": self.components_loaded,
        }

# Global startup manager
startup_manager = UltraFastStartupManager()
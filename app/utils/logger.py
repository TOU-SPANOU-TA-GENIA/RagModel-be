import logging
import sys
from typing import Optional

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger instance.
    Reads 'server.log_level' from config if available.
    """
    logger = logging.getLogger(name)
    
    # Singleton-like: Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger
    
    # 1. Determine Log Level
    # We import inside the function to avoid Circular Dependency:
    # ConfigManager -> imports Logger -> imports ConfigManager
    try:
        from app.config import SERVER
        # Use getattr to be safe against schema changes
        default_level = getattr(SERVER, "log_level", "INFO")
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    except ImportError:
        # Fallback for when logger is used before config is fully loaded
        default_level = "INFO"
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    final_level = level or default_level
    
    # 2. Configure Logger
    try:
        logger.setLevel(final_level.upper())
    except ValueError:
        logger.setLevel(logging.INFO)
    
    # 3. Setup Console Handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)
    
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger
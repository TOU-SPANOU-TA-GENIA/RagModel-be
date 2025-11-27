# app/utils/logger.py
"""
Logging configuration and utilities.
"""

import logging
import sys


def setup_logger(name: str, level: str = None, log_format: str = None) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_format: Custom log format string
    """
    # Import here to avoid circular imports during startup
    try:
        from app.config import SERVER
        default_level = SERVER.log_level
        default_format = SERVER.log_format
    except ImportError:
        default_level = "INFO"
        default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    level = level or default_level
    log_format = log_format or default_format
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level))
    
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger
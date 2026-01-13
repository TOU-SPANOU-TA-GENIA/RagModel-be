"""
Legacy entry point.
Authentication is now handled inherently by the main application 
and configured via 'config.json'.

Please use: python -m app.main
"""

import uvicorn
from app.config import get_config
from app.utils.logger import setup_logger
from app.main import app

logger = setup_logger(__name__)

if __name__ == "__main__":
    logger.info("⚠️  NOTE: 'main_with_auth.py' is deprecated. Using 'app/main.py' instead.")
    logger.info("   Authentication is controlled by the API dependencies in 'app/api/deps.py'.")
    
    config = get_config()
    server_conf = config.get("server", {})
    
    uvicorn.run(
        app, 
        host=server_conf.get("host", "0.0.0.0"), 
        port=server_conf.get("port", 8000)
    )
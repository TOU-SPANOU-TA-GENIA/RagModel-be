import uvicorn
import os
import sys
import argparse
from pathlib import Path

# Add project root to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app.config import get_config

def main():
    parser = argparse.ArgumentParser(description="Start the RagModel Agent Server")
    parser.add_argument("--host", default=None, help="Bind host (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (overrides config)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    
    args = parser.parse_args()

    # Initialize Config System manually if custom path provided
    # (Assuming ConfigManager can handle arbitrary paths, otherwise standard init)
    
    print(f"🚀 Starting Server from {ROOT_DIR}")
    
    # Load config to get defaults
    config = get_config()
    server_conf = config.get("server", {})
    
    # Determine settings (Args > Config > Defaults)
    host = args.host or server_conf.get("host", "0.0.0.0")
    port = args.port or server_conf.get("port", 8000)
    reload = args.reload or server_conf.get("debug_mode", False)
    
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload}")
    
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=server_conf.get("log_level", "info").lower()
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")

if __name__ == "__main__":
    main()
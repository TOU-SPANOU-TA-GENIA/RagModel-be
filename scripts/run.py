#!/usr/bin/env python3
# scripts/run.py
"""
Setup and run the RAG system with network filesystem support.

Usage:
    python scripts/run.py --ingest    # Index network files
    python scripts/run.py --run       # Start the server
    python scripts/run.py --full      # Both
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment():
    """Check environment and network filesystem configuration."""
    from app.config import config
    from app.utils.logger import setup_logger
    
    logger = setup_logger(__name__)
    logger.info("Checking environment...")
    
    try:
        # Check for network filesystem configuration
        network_config = config.get_section('network_filesystem')
        
        if network_config and network_config.get('enabled'):
            logger.info("✅ Network filesystem enabled")
            shares = network_config.get('shares', [])
            logger.info(f"Configured shares: {len(shares)}")
            
            for share in shares:
                if share.get('enabled'):
                    mount_path = share.get('mount_path', 'unknown')
                    share_name = share.get('name', 'unnamed')
                    logger.info(f"  📁 {share_name}: {mount_path}")
                    
                    # Check if mount path exists
                    path = Path(mount_path)
                    if path.exists():
                        logger.info(f"     ✅ Path accessible")
                    else:
                        logger.warning(f"     ⚠️  Path not accessible")
        else:
            logger.warning("⚠️ Network filesystem not enabled")
            logger.warning("Add 'network_filesystem' section to config.json")
    
    except Exception as e:
        logger.error(f"Error checking network config: {e}")
        import traceback
        traceback.print_exc()
    
    return True


def run_ingestion(rebuild=True):
    """Run document ingestion from network shares."""
    from app.utils.logger import setup_logger
    
    logger = setup_logger(__name__)
    logger.info("📥 Starting network filesystem ingestion...")
    
    try:
        from app.core.network_rag_integration import get_network_integrator
        
        integrator = get_network_integrator()
        if not integrator:
            logger.error("❌ Network filesystem not initialized")
            logger.error("Make sure the server has been started at least once to initialize network monitoring")
            return False
        
        # Trigger manual indexing of all files
        result = integrator.index_all_now()
        
        if result.get("success"):
            logger.info("✅ Network ingestion completed!")
            logger.info(f"   Files indexed: {result.get('files_indexed', 0)}")
            logger.info(f"   Total network files: {result.get('total_files', 0)}")
            return True
        else:
            logger.error(f"❌ Ingestion failed: {result.get('message', 'Unknown error')}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Ingestion error: {e}")
        import traceback
        traceback.print_exc()
        return False


def start_server(host="localhost", port=8000):
    """Start the FastAPI server."""
    from app.utils.logger import setup_logger
    
    logger = setup_logger(__name__)
    logger.info("🚀 Starting FastAPI server...")
    logger.info(f"   Host: {host}")
    logger.info(f"   Port: {port}")
    logger.info(f"   API docs: http://{host}:{port}/docs")
    
    import uvicorn
    from app.main import app
    
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="Setup and run the RAG system with network filesystem")
    parser.add_argument("--ingest", action="store_true", help="Run network file ingestion")
    parser.add_argument("--run", action="store_true", help="Start server")
    parser.add_argument("--full", action="store_true", help="Ingest and run")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector store (ignored for network)")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    # Always check environment
    check_environment()
    
    if args.full:
        # Start server (which will auto-start network monitoring and indexing)
        start_server(args.host, args.port)
    elif args.ingest:
        # Manual ingestion trigger
        success = run_ingestion(rebuild=args.rebuild)
        sys.exit(0 if success else 1)
    elif args.run:
        start_server(args.host, args.port)
    else:
        parser.print_help()
        print("\n📋 Quick start:")
        print("  1. Ensure config.json has 'network_filesystem' section configured")
        print("  2. Map network drive (e.g., Z: to \\\\192.168.1.227\\SharedDocs)")
        print("  3. Run: python scripts/run.py --full")
        print("\n💡 Network filesystem will auto-discover and index files on startup")


if __name__ == "__main__":
    main()
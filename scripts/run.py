#!/usr/bin/env python3
# scripts/run.py
"""
Setup and run the RAG system.

Usage:
    python scripts/run.py --ingest    # Build vector store
    python scripts/run.py --run       # Start the server
    python scripts/run.py --full      # Both
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment():
    """Check environment and create directories."""
    from app.config import PATHS
    from app.utils.logger import setup_logger
    
    logger = setup_logger(__name__)
    logger.info("Checking environment...")
    
    knowledge_dir = Path(PATHS.knowledge_dir)
    instructions_dir = Path(PATHS.instructions_dir)
    
    issues = []
    
    if not knowledge_dir.exists():
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created knowledge directory: {knowledge_dir}")
    
    if not instructions_dir.exists():
        logger.warning(f"Instructions directory not found: {instructions_dir}")
        issues.append("instructions_dir")
    
    docs = list(knowledge_dir.glob("*.txt")) + list(knowledge_dir.glob("*.md"))
    if not docs:
        logger.warning(f"No documents found in {knowledge_dir}")
        issues.append("no_documents")
    else:
        logger.info(f"Found {len(docs)} documents")
    
    return len(issues) == 0


def run_ingestion(rebuild=True):
    """Run document ingestion."""
    from app.config import PATHS
    from app.rag.ingestion import ingest_directory
    from app.utils.logger import setup_logger
    
    logger = setup_logger(__name__)
    logger.info("Starting ingestion...")
    
    try:
        result = ingest_directory(Path(PATHS.knowledge_dir), rebuild=rebuild)
        
        if result["success"]:
            logger.info("✅ Ingestion completed!")
            logger.info(f"   Documents: {result['documents_loaded']}")
            logger.info(f"   Chunks: {result['chunks_created']}")
            return True
        else:
            logger.error(f"❌ Ingestion failed: {result['message']}")
            return False
    except Exception as e:
        logger.error(f"❌ Ingestion error: {e}")
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
    parser = argparse.ArgumentParser(description="Setup and run the RAG system")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion")
    parser.add_argument("--run", action="store_true", help="Start server")
    parser.add_argument("--full", action="store_true", help="Ingest and run")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector store")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    check_environment()
    
    if args.full:
        if run_ingestion(rebuild=True):
            start_server(args.host, args.port)
        else:
            sys.exit(1)
    elif args.ingest:
        success = run_ingestion(rebuild=args.rebuild)
        sys.exit(0 if success else 1)
    elif args.run:
        start_server(args.host, args.port)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. Add documents to: data/knowledge/")
        print("  2. Run: python scripts/run.py --full")


if __name__ == "__main__":
    main()
"""
Usage:
    python setup_and_run.py --ingest    # Build vector store
    python setup_and_run.py --run       # Start the server
    python setup_and_run.py --full      # Both
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import KNOWLEDGE_DIR, INDEX_DIR, INSTRUCTIONS_DIR
from app.rag.ingestion import ingest_directory
from app.logger import setup_logger

logger = setup_logger(__name__)


def check_environment():
    logger.info("Checking environment...")
    
    issues = []
    
    if not KNOWLEDGE_DIR.exists():
        logger.warning(f"Knowledge directory not found: {KNOWLEDGE_DIR}")
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created knowledge directory: {KNOWLEDGE_DIR}")
    
    if not INSTRUCTIONS_DIR.exists():
        logger.warning(f"Instructions directory not found: {INSTRUCTIONS_DIR}")
        issues.append("instructions_dir")
    
    # Check for documents
    docs = list(KNOWLEDGE_DIR.glob("*.txt")) + list(KNOWLEDGE_DIR.glob("*.md"))
    if not docs:
        logger.warning(f"No documents found in {KNOWLEDGE_DIR}")
        logger.info("Please add .txt or .md files to the knowledge directory")
        issues.append("no_documents")
    else:
        logger.info(f"Found {len(docs)} documents in knowledge directory")
    
    return len(issues) == 0


def run_ingestion(rebuild=True):
    logger.info("Starting ingestion process...")
    
    try:
        result = ingest_directory(KNOWLEDGE_DIR, rebuild=rebuild)
        
        if result["success"]:
            logger.info("Ingestion completed successfully!")
            logger.info(f"   Documents loaded: {result['documents_loaded']}")
            logger.info(f"   Chunks created: {result['chunks_created']}")
            logger.info(f"   Index saved to: {INDEX_DIR}")
            return True
        else:
            logger.error(f" Ingestion failed: {result['message']}")
            return False
    
    except Exception as e:
        logger.error(f" Ingestion error: {e}")
        return False

# Start the FastAPI server.
def start_server(host="localhost", port=8000):
    logger.info(" Starting FastAPI server...")
    logger.info(f"   Host: {host}")
    logger.info(f"   Port: {port}")
    logger.info(f"   API docs: http://{host}:{port}/docs")
    
    import uvicorn
    from app.main import app
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Setup and run the RAG system"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingestion to build vector store"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start the FastAPI server"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run ingestion and then start server"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector store from scratch (use with --ingest)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    if not check_environment():
        logger.warning("  Environment check found issues, but continuing...")
    
    if args.full:
        # Run ingestion then server
        logger.info("Running full setup...")
        if run_ingestion(rebuild=True):
            logger.info("\n" + "="*50)
            start_server(args.host, args.port)
        else:
            logger.error("Ingestion failed, not starting server")
            sys.exit(1)
    
    elif args.ingest:
        # Only run ingestion
        success = run_ingestion(rebuild=args.rebuild)
        sys.exit(0 if success else 1)
    
    elif args.run:
        # Only start server
        if not INDEX_DIR.exists():
            logger.warning("  Vector store not found!")
            logger.warning("The system will work but won't have RAG capabilities.")
            logger.warning("Run with --ingest first to build the vector store.")
            
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        start_server(args.host, args.port)
    
    else:
        # No arguments provided
        parser.print_help()
        print("\n" + "="*50)
        print("Quick start:")
        print("  1. Add documents to: data/knowledge/")
        print("  2. Run: python setup_and_run.py --full")
        print("="*50)


if __name__ == "__main__":
    main()
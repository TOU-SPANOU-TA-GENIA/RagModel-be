#!/usr/bin/env python3
# scripts/download_models.py
"""
Download models for offline use.

Reads model names from configuration, allowing easy testing of different models.

Usage:
    python scripts/download_models.py              # Download models from config
    python scripts/download_models.py --list       # List configured models
    python scripts/download_models.py --llm MODEL  # Override LLM model
    python scripts/download_models.py --force      # Re-download existing models
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_models_from_config() -> dict:
    """Get model names from configuration."""
    try:
        from app.config import LLM, EMBEDDING
        return {
            "llm": LLM.model_name,
            "embedding": EMBEDDING.model_name
        }
    except ImportError:
        # Fallback defaults if config not available
        return {
            "llm": "meta-llama/Llama-3.2-3B-Instruct",
            "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        }


def get_output_dir() -> Path:
    """Get models output directory from config."""
    try:
        from app.config import PATHS
        return Path(PATHS.offline_models_dir)
    except ImportError:
        return PROJECT_ROOT / "offline_models"


def download_llm(model_name: str, output_dir: Path, force: bool = False) -> Path:
    """Download LLM model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Create safe directory name
    safe_name = model_name.replace("/", "--")
    model_dir = output_dir / "llm" / safe_name
    
    # Check if already exists
    if model_dir.exists() and not force:
        print(f"✓ LLM already exists: {model_dir}")
        print("  Use --force to re-download")
        return model_dir
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading LLM: {model_name}")
    print(f"{'='*60}")
    
    # Download tokenizer
    print("📥 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(model_dir)
    print(f"✓ Tokenizer saved")
    
    # Download model
    print("📥 Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto"
    )
    model.save_pretrained(model_dir)
    print(f"✓ Model saved to: {model_dir}")
    
    return model_dir


def download_embedding(model_name: str, output_dir: Path, force: bool = False) -> Path:
    """Download embedding model."""
    from sentence_transformers import SentenceTransformer
    
    # Create safe directory name
    safe_name = model_name.replace("/", "--")
    model_dir = output_dir / "embeddings" / safe_name
    
    # Check if already exists
    if model_dir.exists() and not force:
        print(f"✓ Embedding model already exists: {model_dir}")
        print("  Use --force to re-download")
        return model_dir
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading Embedding: {model_name}")
    print(f"{'='*60}")
    
    print("📥 Downloading embedding model...")
    model = SentenceTransformer(model_name)
    model.save(str(model_dir))
    print(f"✓ Saved to: {model_dir}")
    
    return model_dir


def update_config_paths(llm_path: Path, embedding_path: Path):
    """Update configuration with downloaded model paths."""
    try:
        from app.config import config
        
        config.set("llm", "model_name", str(llm_path))
        config.set("embedding", "model_name", str(embedding_path))
        config.save()
        
        print("\n✓ Configuration updated with local model paths")
    except Exception as e:
        print(f"\n⚠ Could not update config: {e}")
        print("  You may need to manually update config.json")


def list_configured_models():
    """List models from configuration."""
    models = get_models_from_config()
    output_dir = get_output_dir()
    
    print("\n📋 Configured Models:")
    print(f"{'='*60}")
    
    print(f"\nLLM Model:")
    print(f"  Name: {models['llm']}")
    llm_path = output_dir / "llm" / models['llm'].replace("/", "--")
    print(f"  Path: {llm_path}")
    print(f"  Downloaded: {'✓' if llm_path.exists() else '✗'}")
    
    print(f"\nEmbedding Model:")
    print(f"  Name: {models['embedding']}")
    emb_path = output_dir / "embeddings" / models['embedding'].replace("/", "--")
    print(f"  Path: {emb_path}")
    print(f"  Downloaded: {'✓' if emb_path.exists() else '✗'}")
    
    print(f"\nOutput Directory: {output_dir}")


def list_available_models():
    """List some recommended models for testing."""
    print("\n📚 Recommended Models for Testing:")
    print(f"{'='*60}")
    
    llm_models = [
        ("meta-llama/Llama-3.2-1B-Instruct", "1B params, fast, good for testing"),
        ("meta-llama/Llama-3.2-3B-Instruct", "3B params, balanced (default)"),
        ("google/gemma-2b-it", "2B params, alternative option"),
        ("microsoft/phi-2", "2.7B params, efficient"),
    ]
    
    print("\nLLM Models:")
    for name, desc in llm_models:
        print(f"  • {name}")
        print(f"    {desc}")
    
    embedding_models = [
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "Multilingual, 384 dim (default)"),
        ("sentence-transformers/all-MiniLM-L6-v2", "English, fast, 384 dim"),
        ("sentence-transformers/all-mpnet-base-v2", "English, high quality, 768 dim"),
    ]
    
    print("\nEmbedding Models:")
    for name, desc in embedding_models:
        print(f"  • {name}")
        print(f"    {desc}")
    
    print("\nUsage:")
    print("  python scripts/download_models.py --llm MODEL_NAME")
    print("  python scripts/download_models.py --embedding MODEL_NAME")


def main():
    parser = argparse.ArgumentParser(
        description='Download models for offline use',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Download models from config
  %(prog)s --list                       # List configured models
  %(prog)s --available                  # Show recommended models
  %(prog)s --llm meta-llama/Llama-3.2-1B-Instruct  # Use specific LLM
  %(prog)s --force                      # Re-download existing models
  %(prog)s --no-update-config           # Don't update config after download
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='List configured models')
    parser.add_argument('--available', action='store_true',
                       help='Show recommended models')
    parser.add_argument('--llm', type=str,
                       help='Override LLM model name')
    parser.add_argument('--embedding', type=str,
                       help='Override embedding model name')
    parser.add_argument('--force', action='store_true',
                       help='Re-download existing models')
    parser.add_argument('--no-update-config', action='store_true',
                       help='Do not update config with local paths')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory')
    
    args = parser.parse_args()
    
    # Handle info commands
    if args.list:
        list_configured_models()
        return 0
    
    if args.available:
        list_available_models()
        return 0
    
    # Get model names
    models = get_models_from_config()
    llm_model = args.llm or models["llm"]
    embedding_model = args.embedding or models["embedding"]
    
    # Get output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_output_dir()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("MODEL DOWNLOAD FOR OFFLINE USE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"LLM: {llm_model}")
    print(f"Embedding: {embedding_model}")
    print(f"Force re-download: {args.force}")
    
    # Download models
    downloaded = {"llm": None, "embedding": None}
    
    try:
        downloaded["llm"] = download_llm(llm_model, output_dir, args.force)
    except Exception as e:
        print(f"\n❌ Failed to download LLM: {e}")
    
    try:
        downloaded["embedding"] = download_embedding(embedding_model, output_dir, args.force)
    except Exception as e:
        print(f"\n❌ Failed to download embedding model: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    if downloaded["llm"]:
        print(f"✓ LLM: {downloaded['llm']}")
    else:
        print("✗ LLM: Failed")
    
    if downloaded["embedding"]:
        print(f"✓ Embedding: {downloaded['embedding']}")
    else:
        print("✗ Embedding: Failed")
    
    # Update config if requested
    if not args.no_update_config and downloaded["llm"] and downloaded["embedding"]:
        update_config_paths(downloaded["llm"], downloaded["embedding"])
    
    # Next steps
    print("\n📝 Next steps:")
    print("  1. Add documents to: data/knowledge/")
    print("  2. Run: python scripts/run.py --full")
    
    return 0 if all(downloaded.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
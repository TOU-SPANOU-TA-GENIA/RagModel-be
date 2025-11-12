# download_models.py
"""
Download models for offline use.

Run this on a machine WITH internet access, then copy the models 
to your offline military network.

Usage:
    python download_models.py
"""

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Directory to save models
MODELS_DIR = Path("./offline_models")
MODELS_DIR.mkdir(exist_ok=True)

def download_llm(model_name: str):
    """Download LLM and tokenizer."""
    print(f"\n{'='*70}")
    print(f"Downloading LLM: {model_name}")
    print(f"{'='*70}")
    
    model_dir = MODELS_DIR / "llm" / model_name.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(model_dir)
    print(f"✓ Tokenizer saved to: {model_dir}")
    
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto"
    )
    model.save_pretrained(model_dir)
    print(f"✓ Model saved to: {model_dir}")
    
    return model_dir

def download_embeddings(model_name: str):
    """Download embedding model."""
    print(f"\n{'='*70}")
    print(f"Downloading Embeddings: {model_name}")
    print(f"{'='*70}")
    
    model_dir = MODELS_DIR / "embeddings" / model_name.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading embedding model...")
    model = SentenceTransformer(model_name)
    model.save(str(model_dir))
    print(f"✓ Embeddings saved to: {model_dir}")
    
    return model_dir

def main():
    print("="*70)
    print("MODEL DOWNLOAD FOR OFFLINE USE")
    print("="*70)
    print("\nThis will download models to: ./offline_models/")
    print("After download, copy this folder to your offline network.")
    print()
    
    # Models to download
    llm_models = [
        "meta-llama/Llama-3.2-3B-Instruct",  # Recommended
        # "google/gemma-2b-it",  # Alternative
    ]
    
    embedding_models = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ]
    
    downloaded = {
        "llm": [],
        "embeddings": []
    }
    
    # Download LLMs
    for model_name in llm_models:
        try:
            model_dir = download_llm(model_name)
            downloaded["llm"].append(str(model_dir))
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
    
    # Download embeddings
    for model_name in embedding_models:
        try:
            model_dir = download_embeddings(model_name)
            downloaded["embeddings"].append(str(model_dir))
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print("\nDownloaded models:")
    print(f"  LLM models: {len(downloaded['llm'])}")
    for path in downloaded["llm"]:
        print(f"    - {path}")
    print(f"  Embedding models: {len(downloaded['embeddings'])}")
    for path in downloaded["embeddings"]:
        print(f"    - {path}")
    
    print("\nNext steps:")
    print("  1. Copy ./offline_models/ to your offline military network")
    print("  2. Update app/config.py to use local paths")
    print("  3. See 'offline_config_example.py' for configuration")
    
    # Create example config
    create_config_example(downloaded)

def create_config_example(downloaded):
    """Create example config for offline use."""
    config_example = '''# offline_config_example.py
"""
Configuration for offline model usage.

Copy these settings to your app/config.py on the offline network.
"""

from pathlib import Path

# Base directory where models are stored
OFFLINE_MODELS_DIR = Path("/path/to/offline_models")  # UPDATE THIS PATH

# LLM Model - use local path instead of HuggingFace name
LLM_MODEL_NAME = str(OFFLINE_MODELS_DIR / "llm" / "meta-llama--Llama-3.2-3B-Instruct")

# Embedding Model - use local path
EMBEDDING_MODEL_NAME = str(OFFLINE_MODELS_DIR / "embeddings" / "sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2")

# Rest of your config stays the same...
'''
    
    with open(MODELS_DIR / "offline_config_example.py", "w") as f:
        f.write(config_example)
    
    print(f"\n✓ Created config example: {MODELS_DIR}/offline_config_example.py")

if __name__ == "__main__":
    main()
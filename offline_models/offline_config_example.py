# offline_config_example.py
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

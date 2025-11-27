# app/core/offline.py
"""
Offline mode utilities - disables network access for air-gapped deployments.
"""

import os
import socket


def enable_offline_mode():
    """
    Enable offline mode by setting environment variables.
    Call this at application startup for air-gapped deployments.
    """
    # Disable HuggingFace downloads
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    # Set local cache directory for sentence-transformers
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(
        os.path.join(os.getcwd(), "offline_models", "embeddings")
    )
    
    print("✓ Offline mode enabled - network access disabled for models")


def block_network_completely():
    """
    Completely block network access.
    Use with caution - affects all network operations.
    """
    original_socket = socket.socket
    
    def guarded_socket(*args, **kwargs):
        raise RuntimeError("Network access is blocked in offline mode")
    
    socket.socket = guarded_socket
    print("⚠ Network completely blocked")


def is_offline_mode() -> bool:
    """Check if offline mode is enabled."""
    return os.environ.get("HF_HUB_OFFLINE") == "1"
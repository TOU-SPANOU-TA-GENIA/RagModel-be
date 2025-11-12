# app/offline_mode.py
"""
Enforce offline mode - block all network access.
Import this at the start of main.py for security.
"""

import os
import socket

def disable_network():
    """
    Disable network access for security.
    Call this at application startup.
    """
    # Set environment variables to disable HuggingFace downloads
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    # Disable network for sentence-transformers
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(
        os.path.join(os.getcwd(), "offline_models", "embeddings")
    )
    
    print("✓ Offline mode enabled - network access disabled for models")

# Optional: More aggressive network blocking (use with caution)
def block_network_completely():
    """
    Completely block network access (very aggressive).
    Only use if absolutely necessary.
    """
    original_socket = socket.socket
    
    def guarded_socket(*args, **kwargs):
        raise RuntimeError("Network access is blocked in offline mode")
    
    socket.socket = guarded_socket
    print("⚠ Network completely blocked")
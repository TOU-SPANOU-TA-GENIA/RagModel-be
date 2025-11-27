# app/utils/gpu.py
"""
GPU detection and management utilities.
"""

from typing import Dict, Optional

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def _get_torch():
    """Lazy import torch."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def get_gpu_info() -> Optional[Dict]:
    """Get GPU information if available."""
    torch = _get_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    
    try:
        gpu_id = 0
        props = torch.cuda.get_device_properties(gpu_id)
        
        return {
            "name": torch.cuda.get_device_name(gpu_id),
            "total_memory_gb": props.total_memory / 1e9,
            "allocated_memory_gb": torch.cuda.memory_allocated(gpu_id) / 1e9,
            "reserved_memory_gb": torch.cuda.memory_reserved(gpu_id) / 1e9,
            "free_memory_gb": (props.total_memory - torch.cuda.memory_allocated(gpu_id)) / 1e9,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        }
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        return None


def clear_gpu_cache():
    """Clear GPU memory cache."""
    torch = _get_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


def log_gpu_memory():
    """Log current GPU memory usage."""
    torch = _get_torch()
    if torch is not None and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def get_optimal_device() -> str:
    """Determine the optimal compute device."""
    torch = _get_torch()
    if torch is not None and torch.cuda.is_available():
        gpu_info = get_gpu_info()
        if gpu_info:
            logger.info(f"Using GPU: {gpu_info['name']} ({gpu_info['total_memory_gb']:.2f}GB)")
        return "cuda"
    logger.info("No GPU available, using CPU")
    return "cpu"
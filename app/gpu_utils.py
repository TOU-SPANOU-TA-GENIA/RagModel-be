import torch
from typing import Dict, Optional
from .logger import setup_logger

logger = setup_logger(__name__)


def get_gpu_info() -> Optional[Dict]:
    if not torch.cuda.is_available():
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
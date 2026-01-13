from typing import Dict, Any, Optional, Literal
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def get_torch_device_info() -> Dict[str, Any]:
    """
    Returns generic hardware info regardless of whether CUDA is present.
    Safe to call even if torch is missing (returns error dict).
    """
    try:
        import torch
    except ImportError:
        return {
            "available": False, 
            "device": "cpu", 
            "details": "PyTorch not installed"
        }

    info = {
        "available": True,
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "active_device": "cpu",
        "device_name": "CPU",
        "memory": {}
    }

    if torch.cuda.is_available():
        try:
            current = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(current)
            
            info["device_count"] = torch.cuda.device_count()
            info["active_device"] = f"cuda:{current}"
            info["device_name"] = torch.cuda.get_device_name(current)
            
            # Memory Stats (in GB)
            total = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(current) / (1024**3)
            reserved = torch.cuda.memory_reserved(current) / (1024**3)
            
            info["memory"] = {
                "total_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "free_gb": round(total - allocated, 2)
            }
        except Exception as e:
            logger.error(f"Error reading GPU props: {e}")
            info["error"] = str(e)

    return info

def clear_gpu_memory():
    """
    Force garbage collection on GPU cache. 
    Useful between heavy agent workflows.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("GPU memory cache cleared.")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")

def get_optimal_device() -> Literal["cuda", "cpu"]:
    """
    Decides the best device to use based on availability.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
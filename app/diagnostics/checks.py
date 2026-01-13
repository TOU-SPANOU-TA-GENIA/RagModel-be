import sys
import shutil
import subprocess
import os
from typing import Dict, Any, List, Tuple
import importlib.util

from app.diagnostics.schemas import CheckResult, CheckStatus
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class SystemChecks:
    @staticmethod
    def _run_cmd(cmd: str) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0, result.stdout.strip()
        except Exception as e:
            return False, str(e)

    @classmethod
    def check_nvidia_driver(cls) -> CheckResult:
        """Check if NVIDIA driver is installed and responding."""
        success, output = cls._run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        
        status = CheckStatus.PASS if success else CheckStatus.FAIL
        msg = f"Driver found: {output}" if success else "NVIDIA driver not detected"
        recs = []
        
        if not success:
            recs = [
                "Install NVIDIA drivers (e.g., 'sudo apt install nvidia-driver-535')",
                "Download drivers from https://www.nvidia.com/drivers"
            ]

        return CheckResult(
            name="NVIDIA Driver",
            category="gpu",
            status=status,
            message=msg,
            details={"output": output},
            recommendations=recs
        )

    @classmethod
    def check_cuda_toolkit(cls) -> CheckResult:
        """Check CUDA availability."""
        # Method 1: nvcc
        success, output = cls._run_cmd("nvcc --version")
        # Method 2: Check env var
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        
        details = {
            "nvcc_output": output if success else None,
            "cuda_home": cuda_home
        }

        if success:
            return CheckResult(
                name="CUDA Toolkit",
                category="gpu",
                status=CheckStatus.PASS,
                message="CUDA Toolkit detected via nvcc",
                details=details
            )
        elif cuda_home:
             return CheckResult(
                name="CUDA Toolkit",
                category="gpu",
                status=CheckStatus.WARNING,
                message="CUDA_HOME set but nvcc not found in PATH",
                details=details
            )
        else:
            return CheckResult(
                name="CUDA Toolkit",
                category="gpu",
                status=CheckStatus.FAIL,
                message="CUDA Toolkit not found",
                details=details,
                recommendations=["Install CUDA Toolkit from developer.nvidia.com"]
            )

    @classmethod
    def check_pytorch_gpu(cls) -> CheckResult:
        """Check PyTorch GPU access."""
        details = {}
        recs = []
        
        try:
            import torch
            details["version"] = torch.__version__
            details["built_with_cuda"] = torch.version.cuda
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                details["device_count"] = device_count
                details["device_0"] = device_name
                
                return CheckResult(
                    name="PyTorch GPU",
                    category="gpu",
                    status=CheckStatus.PASS,
                    message=f"PyTorch can access {device_count} GPU(s). Primary: {device_name}",
                    details=details
                )
            else:
                msg = "PyTorch installed but CUDA is not available"
                status = CheckStatus.FAIL
                
                if not torch.version.cuda:
                    msg = "PyTorch installed as CPU-only version"
                    recs.append("Reinstall PyTorch with CUDA support (check pytorch.org)")
                else:
                    msg = "PyTorch has CUDA support but cannot see GPU"
                    recs.append("Check NVIDIA driver installation")
                
                return CheckResult(
                    name="PyTorch GPU",
                    category="gpu",
                    status=status,
                    message=msg,
                    details=details,
                    recommendations=recs
                )
                
        except ImportError:
            return CheckResult(
                name="PyTorch GPU",
                category="gpu",
                status=CheckStatus.FAIL,
                message="PyTorch package not installed",
                recommendations=["pip install torch torchvision"]
            )

    @classmethod
    def check_disk_space(cls) -> CheckResult:
        """Check generic system health (example of extensibility)."""
        try:
            total, used, free = shutil.disk_usage(".")
            free_gb = free // (2**30)
            
            status = CheckStatus.PASS
            msg = f"{free_gb} GB free space available"
            recs = []
            
            if free_gb < 5:
                status = CheckStatus.WARNING
                msg = "Low disk space (< 5GB)"
                recs.append("Clear cached models or logs")
                
            return CheckResult(
                name="Disk Space",
                category="system",
                status=status,
                message=msg,
                details={"free_gb": free_gb, "total_gb": total // (2**30)}
            )
        except Exception as e:
            return CheckResult(
                name="Disk Space",
                category="system",
                status=CheckStatus.WARNING,
                message=f"Could not check disk: {e}"
            )
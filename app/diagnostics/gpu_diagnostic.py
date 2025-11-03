#!/usr/bin/env python3
import sys
import subprocess
import os

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def run_command(cmd, description):
    print(f"\n[{description}]")
    print(f"Command: {cmd}")
    print("-" * 40)
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(f"Error output: {result.stderr.strip()}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  Command timed out")
        return False
    except Exception as e:
        print(f" Error running command: {e}")
        return False

def check_nvidia_driver():
    print_section("1. NVIDIA Driver Check")
    
    if run_command("nvidia-smi", "NVIDIA System Management Interface"):
        print("   NVIDIA driver is installed and working")
        return True
    else:
        print(" NVIDIA driver not found or not working")
        print("\nTo install NVIDIA drivers:")
        print("  Ubuntu/Debian: sudo apt install nvidia-driver-XXX")
        print("  Windows: Download from https://www.nvidia.com/drivers")
        return False

def check_cuda():
    print_section("2. CUDA Installation Check")
    
    if run_command("nvcc --version", "CUDA Compiler Version"):
        print(" CUDA toolkit is installed")
        cuda_found = True
    else:
        print(" CUDA toolkit not found")
        print("\nTo install CUDA:")
        print("  Visit: https://developer.nvidia.com/cuda-downloads")
        cuda_found = False
    
    print("\n[CUDA Environment Variables]")
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print(f"CUDA_HOME: {cuda_home}")
    else:
        print("  CUDA_HOME not set")
    
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if 'cuda' in ld_library_path.lower():
        print(f"LD_LIBRARY_PATH includes CUDA: {ld_library_path}")
    else:
        print("  LD_LIBRARY_PATH doesn't include CUDA")
    
    return cuda_found

def check_pytorch():
    print_section("3. PyTorch Configuration")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        
        if torch.cuda.is_available():
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Compute Capability: {props.major}.{props.minor}")
                print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(" PyTorch can use GPU")
            return True
        else:
            print(" PyTorch CUDA support not available")
            print("\nPossible reasons:")
            print("  1. PyTorch installed without CUDA support")
            print("  2. CUDA version mismatch")
            print("  3. No NVIDIA GPU present")
            return False
            
    except ImportError:
        print(" PyTorch not installed")
        return False

def check_pytorch_installation_type():
    print_section("4. PyTorch Installation Type")
    
    try:
        import torch
        
        print("\n[Checking PyTorch build]")
        
        try:
            from torch.utils.cpp_extension import CUDA_HOME
            print(f"PyTorch CUDA_HOME: {CUDA_HOME}")
        except:
            print("  PyTorch CUDA_HOME not accessible")
        
        # Check if this is CPU-only build
        if not hasattr(torch.version, 'cuda') or torch.version.cuda is None:
            print(" This is a CPU-ONLY PyTorch build")
            print("\nTo install PyTorch with CUDA support:")
            print("  Visit: https://pytorch.org/get-started/locally/")
            print("\n  Example for CUDA 11.8:")
            print("    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("\n  Example for CUDA 12.1:")
            print("    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return False
        else:
            print(f" PyTorch built with CUDA {torch.version.cuda}")
            return True
            
    except ImportError:
        print(" PyTorch not installed")
        return False

def check_system_info():
    print_section("5. System Information")
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read()
            if 'microsoft' in version_info.lower():
                print("  Running in WSL (Windows Subsystem for Linux)")
                print("Note: WSL2 with GPU support requires special setup")
    except:
        pass
    
    # Check GPU info on Linux
    if sys.platform.startswith('linux'):
        run_command("lspci | grep -i nvidia", "PCI devices (NVIDIA)")
        run_command("lsmod | grep nvidia", "Loaded NVIDIA kernel modules")

def check_pip_packages():
    print_section("6. Relevant Python Packages")
    
    packages = ['torch', 'torchvision', 'torchaudio', 'transformers', 
                'accelerate', 'bitsandbytes', 'xformers']
    
    for package in packages:
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', package],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Extract version
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(':')[1].strip()
                        print(f"✅ {package}: {version}")
                        break
            else:
                print(f" {package}: Not installed")
        except:
            print(f" {package}: Error checking")

def provide_recommendations(driver_ok, cuda_ok, pytorch_ok, pytorch_cuda_build):
    print_section("RECOMMENDATIONS")
    
    if not driver_ok:
        print("\n Step 1: Install NVIDIA Driver")
        print("   Your system doesn't have NVIDIA drivers installed.")
        print("   Install appropriate drivers for your GPU:")
        print("   - Ubuntu: sudo ubuntu-drivers autoinstall")
        print("   - Or download from: https://www.nvidia.com/drivers")
        print("   - After installation, reboot your system")
        
    elif not cuda_ok:
        print("\n Step 2: Install CUDA Toolkit")
        print("   NVIDIA driver is present but CUDA toolkit is missing.")
        print("   Visit: https://developer.nvidia.com/cuda-downloads")
        print("   Install CUDA toolkit matching your driver version.")
        
    elif not pytorch_cuda_build:
        print("\n Step 3: Reinstall PyTorch with CUDA Support")
        print("   Your PyTorch installation doesn't have CUDA support.")
        print("\n   1. Uninstall current PyTorch:")
        print("      pip uninstall torch torchvision torchaudio")
        print("\n   2. Install PyTorch with CUDA:")
        print("      Visit https://pytorch.org/get-started/locally/")
        print("      and follow instructions for your CUDA version.")
        print("\n   Example commands:")
        print("      # For CUDA 11.8:")
        print("      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n      # For CUDA 12.1:")
        print("      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
    elif not pytorch_ok:
        print("\n Troubleshooting Required")
        print("   Everything seems installed but PyTorch can't access GPU.")
        print("   Try:")
        print("   1. Restart your system")
        print("   2. Check driver compatibility with CUDA version")
        print("   3. Verify GPU is not being used by another process")
        print("   4. Check nvidia-smi output for errors")
        
    else:
        print("\n Everything looks good!")
        print("   GPU should be accessible. If your application still")
        print("   doesn't use GPU, check application-specific settings.")

def main():
    """Run all diagnostic checks."""
    print("="*60)
    print("  GPU DIAGNOSTIC TOOL")
    print("  Checking why GPU is not detected...")
    print("="*60)
    
    # Run all checks
    driver_ok = check_nvidia_driver()
    cuda_ok = check_cuda()
    pytorch_ok = check_pytorch()
    pytorch_cuda_build = check_pytorch_installation_type()
    check_system_info()
    check_pip_packages()
    
    # Provide recommendations
    provide_recommendations(driver_ok, cuda_ok, pytorch_ok, pytorch_cuda_build)
    
    print("\n" + "="*60)
    print("  Diagnostic Complete")
    print("="*60)

if __name__ == "__main__":
    main()
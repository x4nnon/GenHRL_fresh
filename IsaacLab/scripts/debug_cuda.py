#!/usr/bin/env python3
"""
CUDA Compatibility Diagnostic Script
====================================

This script helps diagnose and fix CUDA/PyTorch compatibility issues,
particularly the cuSOLVER error encountered during Isaac Sim training.

Usage:
    python scripts/debug_cuda.py
"""

import os
import sys
import subprocess
import platform

def check_cuda_version():
    """Check CUDA version and driver info."""
    print("=== CUDA System Information ===")
    
    try:
        # Check NVIDIA driver
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    print(f"   {line.strip()}")
                elif 'CUDA Version:' in line:
                    print(f"   {line.strip()}")
        else:
            print("❌ nvidia-smi failed - GPU/driver issue")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not found - NVIDIA drivers not installed")
        return False
    
    return True

def check_pytorch_cuda():
    """Check PyTorch CUDA compatibility."""
    print("\n=== PyTorch CUDA Compatibility ===")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            try:
                version_module = getattr(torch, 'version', None)
                cuda_version = getattr(version_module, 'cuda', 'Not available') if version_module else 'Not available'
                print(f"   CUDA version: {cuda_version}")
            except Exception:
                print("   CUDA version: Not available")
            try:
                print(f"   cuDNN version: {torch.backends.cudnn.version()}")
            except AttributeError:
                print("   cuDNN version: Not available")
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            print(f"   Device name: {torch.cuda.get_device_name()}")
            
            # Test basic CUDA operations
            try:
                x = torch.randn(10, 10).cuda()
                y = torch.randn(10, 10).cuda()
                z = torch.matmul(x, y)
                print("✅ Basic CUDA operations work")
            except Exception as e:
                print(f"❌ Basic CUDA operations failed: {e}")
                return False
                
            # Test cuSOLVER specifically
            try:
                print("\n--- Testing cuSOLVER operations ---")
                
                # Test matrix operations that use cuSOLVER
                A = torch.randn(100, 100, device='cuda', dtype=torch.float32)
                B = torch.randn(100, 100, device='cuda', dtype=torch.float32)
                
                # Test linalg.solve (uses cuSOLVER)
                try:
                    result = torch.linalg.solve(A, B)
                    print("✅ torch.linalg.solve works")
                except Exception as e:
                    print(f"❌ torch.linalg.solve failed: {e}")
                    print("   This is the exact error causing training failure!")
                    return False
                    
                # Test matrix inverse (also uses cuSOLVER)
                try:
                    inv_A = torch.linalg.inv(A)
                    print("✅ torch.linalg.inv works")
                except Exception as e:
                    print(f"❌ torch.linalg.inv failed: {e}")
                    
            except Exception as e:
                print(f"❌ cuSOLVER test setup failed: {e}")
                return False
                
        else:
            print("❌ CUDA not available in PyTorch")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    return True

def check_isaac_sim_environment():
    """Check Isaac Sim specific environment."""
    print("\n=== Isaac Sim Environment ===")
    
    # Check if we're in Isaac Sim environment
    try:
        import omni.kit.app
        print("✅ Isaac Sim environment detected")
    except ImportError:
        print("⚠️  Not in Isaac Sim environment (this is OK for diagnosis)")
    
    # Check environment variables
    important_vars = [
        'CUDA_VISIBLE_DEVICES',
        'PYTORCH_CUDA_ALLOC_CONF', 
        'TORCH_CUDA_ARCH_LIST',
        'ISAACLAB_PATH'
    ]
    
    print("\nEnvironment variables:")
    for var in important_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")

def suggest_fixes():
    """Suggest potential fixes for cuSOLVER issues."""
    print("\n=== Suggested Fixes ===")
    
    print("1. 🔧 Try different PyTorch linear algebra backends:")
    print("   Add to your training script before imports:")
    print("   ```python")
    print("   import torch")
    print("   torch.backends.cuda.preferred_linalg_library('default')")
    print("   # or")
    print("   torch.backends.cuda.preferred_linalg_library('magma')")
    print("   ```")
    
    print("\n2. 🔧 Set CUDA memory management:")
    print("   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
    
    print("\n3. 🔧 Force CPU for problematic operations:")
    print("   Add environment variable:")
    print("   export CUDA_LAUNCH_BLOCKING=1")
    
    print("\n4. 🔧 Update PyTorch/CUDA versions:")
    print("   - Check PyTorch compatibility: https://pytorch.org/get-started/")
    print("   - Consider downgrading PyTorch if using bleeding edge")
    print("   - Update NVIDIA drivers if very old")
    
    print("\n5. 🔧 Isaac Sim specific workarounds:")
    print("   Try running with reduced environments:")
    print("   ./isaaclab.sh -p scripts/.../train.py --num_envs 1024")
    print("   (instead of 4096)")
    
    print("\n6. 🔧 Last resort - CPU fallback:")
    print("   If GPU training impossible, modify configs to use CPU")
    print("   (much slower but will work)")

def run_automatic_fixes():
    """Apply automatic fixes."""
    print("\n=== Applying Automatic Fixes ===")
    
    try:
        import torch
        
        # Try to set the best linear algebra backend
        if hasattr(torch.backends.cuda, 'preferred_linalg_library'):
            backends_to_try = ['default', 'magma', 'cusolver']
            
            for backend in backends_to_try:
                try:
                    torch.backends.cuda.preferred_linalg_library(backend)
                    print(f"✅ Set linear algebra backend to: {backend}")
                    break
                except Exception as e:
                    print(f"❌ Failed to set backend {backend}: {e}")
        
        # Set memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        print("✅ Set CUDA memory allocation config")
        
    except ImportError:
        print("❌ PyTorch not available for automatic fixes")

def main():
    """Run full diagnostic."""
    print("CUDA/PyTorch Diagnostic Tool")
    print("=" * 50)
    
    cuda_ok = check_cuda_version()
    if not cuda_ok:
        print("\n❌ CUDA system check failed - fix GPU/driver issues first")
        return
    
    pytorch_ok = check_pytorch_cuda()
    check_isaac_sim_environment()
    
    if not pytorch_ok:
        print("\n❌ PyTorch CUDA compatibility issues detected")
        suggest_fixes()
        
        answer = input("\nTry automatic fixes? (y/N): ").lower().strip()
        if answer == 'y':
            run_automatic_fixes()
            print("\nRerun this script to test if fixes worked.")
    else:
        print("\n✅ All CUDA/PyTorch tests passed!")
        print("   The cuSOLVER error might be environment-specific.")
        print("   Try the suggested fixes anyway.")
        suggest_fixes()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
"""
Quick test script to verify MPS (Metal Performance Shaders) support on Apple Silicon Macs.
Run this after activating the conda environment to check if GPU acceleration is available.
"""
import torch
from utils import get_device, get_device_name

def main():
    print("=" * 60)
    print("PyTorch GPU Acceleration Test")
    print("=" * 60)
    
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check MPS (Apple Silicon)
    print(f"\nMPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"  MPS built: {torch.backends.mps.is_built()}")
        print("  ? Apple Silicon GPU acceleration is ready!")
    
    # Get recommended device
    device = get_device()
    print(f"\nRecommended device: {get_device_name(device)}")
    
    # Test a simple tensor operation on the recommended device
    print(f"\nTesting tensor operations on {device.type}...")
    try:
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print(f"  ? Successfully created and multiplied tensors on {device.type}")
        print(f"  Result tensor shape: {z.shape}")
    except Exception as e:
        print(f"  ? Error: {e}")
    
    print("\n" + "=" * 60)
    
    if torch.backends.mps.is_available():
        print("\n?? Your M4 Max GPU is ready for acceleration!")
        print("   Use 'from utils import get_device' in your code to automatically use MPS.")
    elif torch.cuda.is_available():
        print("\n?? Your NVIDIA GPU is ready for acceleration!")
    else:
        print("\n??  No GPU acceleration available. Using CPU.")

if __name__ == "__main__":
    main()

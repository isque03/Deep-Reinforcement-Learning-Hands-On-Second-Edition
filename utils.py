"""
Utility functions for device selection and common operations.
"""
import torch


def get_device(cuda: bool = False) -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Priority order:
    1. CUDA (if requested and available)
    2. MPS (Metal Performance Shaders on Apple Silicon Macs)
    3. CPU (fallback)
    
    Args:
        cuda: If True, prefer CUDA when available. Otherwise, prefer MPS on macOS.
        
    Returns:
        torch.device: The best available device.
    """
    if cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    """
    Get a human-readable name for the device.
    
    Args:
        device: The torch device.
        
    Returns:
        str: Human-readable device name.
    """
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device.type == "mps":
        return "MPS (Apple Silicon GPU)"
    else:
        return "CPU"

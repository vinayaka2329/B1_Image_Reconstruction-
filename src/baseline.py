"""
Classical image processing baselines for comparison:
- Raw degraded image (No filtering baseline)
- Gaussian Blur filtering (Linear classical filter)
- Median filtering (Non-linear rank filter, classical S&P baseline)
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def apply_gaussian_filter_batch(images: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Applies classical Gaussian filter across a batch of images.
    """
    arr = images.detach().cpu().numpy()
    filtered = np.zeros_like(arr)
    
    for b in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            filtered[b, c] = gaussian_filter(arr[b, c], sigma=sigma)
            
    filtered = np.clip(filtered, 0.0, 1.0)
    return torch.from_numpy(filtered).float().to(images.device)


def apply_median_filter_batch(images: torch.Tensor, size: int = 3) -> torch.Tensor:
    """
    Applies classical Median filter across a batch of images.
    Gold standard classical baseline for impulse (salt-and-pepper) noise.
    """
    arr = images.detach().cpu().numpy()
    filtered = np.zeros_like(arr)
    
    for b in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            filtered[b, c] = median_filter(arr[b, c], size=size)
            
    filtered = np.clip(filtered, 0.0, 1.0)
    return torch.from_numpy(filtered).float().to(images.device)

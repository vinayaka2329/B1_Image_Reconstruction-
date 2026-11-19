"""
Evaluation metrics: MSE (Mean Squared Error), PSNR (Peak Signal-to-Noise Ratio),
and SSIM (Structural Similarity Index Measure).
Includes high-performance vectorized PyTorch SSIM and scikit-image SSIM.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compute_ssim
from typing import Dict, Tuple


def calculate_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Computes Mean Squared Error (MSE) between two image tensors [0, 1]."""
    return torch.mean((img1 - img2) ** 2).item()


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0, eps: float = 1e-10) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Formula: PSNR = 10 * log10(MAX^2 / (MSE + eps))
    """
    mse = calculate_mse(img1, img2)
    if mse < eps:
        return 100.0  # Identical images
    return 10.0 * math.log10((max_val ** 2) / mse)


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Generates a 1D Gaussian kernel."""
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_2d_window(window_size: int = 11, channel: int = 1) -> torch.Tensor:
    """Creates a 2D Gaussian kernel window for depthwise 2D convolution."""
    _1D_window = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_torch_batch(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    High-performance PyTorch-native SSIM computation on CPU/GPU.
    Significantly faster than per-image NumPy conversion.
    """
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)
        
    channel = img1.size(1)
    window = _create_2d_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def calculate_ssim_tensor(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Computes mean SSIM using fast PyTorch implementation."""
    return ssim_torch_batch(img1, img2)


def calculate_ssim_skimage(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Computes mean SSIM using scikit-image (for standard verification)."""
    arr1 = img1.detach().cpu().numpy()
    arr2 = img2.detach().cpu().numpy()
    
    if arr1.ndim == 4:
        ssims = [compute_ssim(arr1[i, 0], arr2[i, 0], data_range=1.0) for i in range(arr1.shape[0])]
        return float(np.mean(ssims))
    elif arr1.ndim == 3:
        return float(compute_ssim(arr1[0], arr2[0], data_range=1.0))
    elif arr1.ndim == 2:
        return float(compute_ssim(arr1, arr2, data_range=1.0))
    return 0.0


def evaluate_batch(clean: torch.Tensor, degraded_or_reconstructed: torch.Tensor) -> Dict[str, float]:
    """Evaluates MSE, PSNR, and SSIM for a batch of images against clean images."""
    clean_clamped = torch.clamp(clean, 0.0, 1.0)
    eval_clamped = torch.clamp(degraded_or_reconstructed, 0.0, 1.0)
    
    mse = calculate_mse(clean_clamped, eval_clamped)
    psnr = calculate_psnr(clean_clamped, eval_clamped)
    ssim = calculate_ssim_tensor(clean_clamped, eval_clamped)
    
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim
    }

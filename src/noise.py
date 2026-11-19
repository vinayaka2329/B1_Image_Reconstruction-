"""
Noise injection mechanisms: Additive White Gaussian Noise (AWGN),
Salt-and-Pepper (Impulse) Noise, and Mixed Noise degradations.
"""

import torch
from typing import Optional


def add_gaussian_noise(
    images: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.3,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Applies Additive White Gaussian Noise (AWGN) to normalized images [0, 1].
    
    Formula:
        I_noisy = clip(I_clean + N(mean, std^2), 0.0, 1.0)
        
    Args:
        images: Tensor of shape (B, C, H, W) with values in [0, 1].
        mean: Gaussian distribution mean (default: 0.0).
        std: Gaussian noise standard deviation (sigma).
        generator: Optional PyTorch random generator for reproducibility.
        
    Returns:
        Noisy image tensor with values strictly clamped to [0, 1].
    """
    if generator is not None:
        noise = torch.randn(images.shape, generator=generator, device=images.device) * std + mean
    else:
        noise = torch.randn_like(images) * std + mean
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)


def add_salt_and_pepper_noise(
    images: torch.Tensor,
    prob: float = 0.15,
    salt_vs_pepper: float = 0.5,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Applies Salt-and-Pepper (impulse) noise to normalized images [0, 1].
    
    Formula:
        For each pixel with probability `prob`:
            - Set to 1.0 (Salt) with probability `salt_vs_pepper`
            - Set to 0.0 (Pepper) with probability `1 - salt_vs_pepper`
            
    Args:
        images: Tensor of shape (B, C, H, W) with values in [0, 1].
        prob: Total fraction of corrupted pixels (e.g. 0.15).
        salt_vs_pepper: Ratio of salt to pepper noise (default 0.5).
        generator: Optional PyTorch random generator for reproducibility.
        
    Returns:
        Noisy image tensor with impulse corrupted pixels in [0, 1].
    """
    noisy_images = images.clone()
    
    if generator is not None:
        rand_mask = torch.rand(images.shape, generator=generator, device=images.device)
        rand_choice = torch.rand(images.shape, generator=generator, device=images.device)
    else:
        rand_mask = torch.rand_like(images)
        rand_choice = torch.rand_like(images)
        
    # Mask of corrupted pixels
    corrupt_mask = rand_mask < prob
    
    # Within corrupted pixels, decide whether salt or pepper
    salt_mask = corrupt_mask & (rand_choice < salt_vs_pepper)
    pepper_mask = corrupt_mask & (rand_choice >= salt_vs_pepper)
    
    noisy_images[salt_mask] = 1.0
    noisy_images[pepper_mask] = 0.0
    
    return noisy_images


def add_mixed_noise(
    images: torch.Tensor,
    gaussian_std: float = 0.25,
    sp_prob: float = 0.1,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Applies both Gaussian noise and Salt-and-Pepper impulse noise.
    
    Args:
        images: Tensor of shape (B, C, H, W).
        gaussian_std: Standard deviation for Gaussian noise.
        sp_prob: Corruption probability for salt-and-pepper noise.
        generator: Optional PyTorch random generator.
        
    Returns:
        Composite degraded image tensor in [0, 1].
    """
    img_gauss = add_gaussian_noise(images, std=gaussian_std, generator=generator)
    img_mixed = add_salt_and_pepper_noise(img_gauss, prob=sp_prob, generator=generator)
    return img_mixed


def apply_noise_by_type(images: torch.Tensor, noise_type: str, severity: float = 1.0, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Utility dispatcher to apply noise based on string identifier and scaling factor.
    
    noise_type: 'gaussian', 'salt_and_pepper', 'sp', 'mixed'
    """
    nt = noise_type.lower()
    if nt in ("gaussian", "gauss"):
        std = 0.3 * severity
        return add_gaussian_noise(images, std=std, generator=generator)
    elif nt in ("salt_and_pepper", "sp", "salt_pepper"):
        prob = min(0.15 * severity, 0.8)
        return add_salt_and_pepper_noise(images, prob=prob, generator=generator)
    elif nt == "mixed":
        std = 0.25 * severity
        prob = min(0.10 * severity, 0.5)
        return add_mixed_noise(images, gaussian_std=std, sp_prob=prob, generator=generator)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

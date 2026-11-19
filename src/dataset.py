"""
Dataset loaders and utilities for MNIST image reconstruction.
Optimized for PyTorch CPU execution on Windows.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from typing import Tuple, Optional
from src.noise import add_gaussian_noise, add_salt_and_pepper_noise, add_mixed_noise


class OnTheFlyNoisyDataset(Dataset):
    """
    Wraps standard MNIST dataset to return (clean_image, noisy_image, label) on the fly.
    """
    def __init__(self, base_dataset: Dataset, noise_type: str = "gaussian", noise_param: float = 0.3):
        self.base_dataset = base_dataset
        self.noise_type = noise_type
        self.noise_param = noise_param
        
    def __len__(self) -> int:
        return len(self.base_dataset)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        clean_img, label = self.base_dataset[idx]
        # Apply noise
        if self.noise_type == "gaussian":
            noisy_img = add_gaussian_noise(clean_img.unsqueeze(0), std=self.noise_param).squeeze(0)
        elif self.noise_type in ("sp", "salt_and_pepper"):
            noisy_img = add_salt_and_pepper_noise(clean_img.unsqueeze(0), prob=self.noise_param).squeeze(0)
        elif self.noise_type == "mixed":
            noisy_img = add_mixed_noise(clean_img.unsqueeze(0), gaussian_std=self.noise_param, sp_prob=0.1).squeeze(0)
        else:
            noisy_img = clean_img.clone()
        return clean_img, noisy_img, label


def load_mnist_datasets(
    data_dir: str = "./data",
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Loads MNIST and splits training into train and validation sets.
    
    Returns:
        train_dataset (clean), val_dataset (clean), test_dataset (clean)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.ToTensor()
    
    full_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator
    )
    
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders configured for Windows CPU execution.
    `num_workers=0` avoids Windows multiprocessing IPC overhead.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader

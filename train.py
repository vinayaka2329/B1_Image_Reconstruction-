"""
Training script for Convolutional Denoising Autoencoder (CDAE) on MNIST.
Optimized for CPU execution on Windows.
"""

import os
import sys
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ConvDenoisingAutoencoder
from src.dataset import load_mnist_datasets, get_dataloaders
from src.noise import add_gaussian_noise, add_salt_and_pepper_noise, add_mixed_noise
from src.metrics import calculate_mse, calculate_psnr, calculate_ssim_tensor


def inject_training_noise(images: torch.Tensor, noise_type: str) -> torch.Tensor:
    """Applies stochastic noise during training."""
    if noise_type == "gaussian":
        std = float(torch.empty(1).uniform_(0.2, 0.4).item())
        return add_gaussian_noise(images, std=std)
    elif noise_type in ("sp", "salt_and_pepper"):
        prob = float(torch.empty(1).uniform_(0.10, 0.25).item())
        return add_salt_and_pepper_noise(images, prob=prob)
    elif noise_type == "joint":
        mode = torch.randint(0, 3, (1,)).item()
        if mode == 0:
            return add_gaussian_noise(images, std=float(torch.empty(1).uniform_(0.2, 0.4).item()))
        elif mode == 1:
            return add_salt_and_pepper_noise(images, prob=float(torch.empty(1).uniform_(0.1, 0.25).item()))
        else:
            return add_mixed_noise(images, gaussian_std=0.25, sp_prob=0.12)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def inject_val_noise(images: torch.Tensor, noise_type: str, seed: int = 123) -> torch.Tensor:
    """Applies fixed deterministic noise for consistent validation tracking."""
    gen = torch.Generator().manual_seed(seed)
    if noise_type == "gaussian":
        return add_gaussian_noise(images, std=0.3, generator=gen)
    elif noise_type in ("sp", "salt_and_pepper"):
        return add_salt_and_pepper_noise(images, prob=0.15, generator=gen)
    elif noise_type == "joint":
        return add_mixed_noise(images, gaussian_std=0.25, sp_prob=0.10, generator=gen)
    else:
        return images


def train_model(
    noise_type: str = "gaussian",
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    data_dir: str = "./data",
    checkpoint_dir: str = "./checkpoints",
    output_dir: str = "./outputs/tables",
    num_threads: int = 6
):
    torch.set_num_threads(num_threads)
    print(f"[{noise_type.upper()}] CPU Threads configured: {torch.get_num_threads()}", flush=True)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cpu")
    print(f"[{noise_type.upper()}] Target training device: {device}", flush=True)
    
    # Load dataset
    print(f"[{noise_type.upper()}] Loading MNIST dataset...", flush=True)
    train_ds, val_ds, test_ds = load_mnist_datasets(data_dir=data_dir, val_split=0.1, seed=42)
    train_loader, val_loader, _ = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size, num_workers=0
    )
    print(f"[{noise_type.upper()}] Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}", flush=True)
    
    # Initialize model, criterion, optimizer
    model = ConvDenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"[{noise_type.upper()}] Model initialized. Trainable parameters: {model.count_parameters():,}", flush=True)
    
    history = []
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(checkpoint_dir, f"cdae_{noise_type}.pth")
    
    total_start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (clean_imgs, _) in enumerate(train_loader):
            clean_imgs = clean_imgs.to(device)
            noisy_imgs = inject_training_noise(clean_imgs, noise_type)
            
            optimizer.zero_grad()
            reconstructed = model(noisy_imgs)
            loss = criterion(reconstructed, clean_imgs)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * clean_imgs.size(0)
            train_samples += clean_imgs.size(0)
            
        train_loss = running_train_loss / train_samples
        
        # Fast Validation phase
        model.eval()
        running_val_loss = 0.0
        val_samples = 0
        val_psnr_list = []
        val_ssim_list = []
        
        with torch.no_grad():
            for batch_idx, (clean_imgs, _) in enumerate(val_loader):
                clean_imgs = clean_imgs.to(device)
                noisy_imgs = inject_val_noise(clean_imgs, noise_type, seed=1000 + batch_idx)
                
                reconstructed = model(noisy_imgs)
                v_loss = criterion(reconstructed, clean_imgs)
                
                running_val_loss += v_loss.item() * clean_imgs.size(0)
                val_samples += clean_imgs.size(0)
                
                val_psnr_list.append(calculate_psnr(clean_imgs, reconstructed))
                val_ssim_list.append(calculate_ssim_tensor(clean_imgs, reconstructed))
                    
        val_loss = running_val_loss / val_samples
        val_psnr = float(torch.tensor(val_psnr_list).mean().item())
        val_ssim = float(torch.tensor(val_ssim_list).mean().item())
        
        epoch_duration = time.time() - epoch_start
        scheduler.step(val_loss)
        
        # Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "noise_type": noise_type,
            }, checkpoint_path)
            saved_flag = " [*SAVED BEST*]"
        else:
            saved_flag = ""
            
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Time: {epoch_duration:.1f}s | "
              f"Train MSE: {train_loss:.5f} | "
              f"Val MSE: {val_loss:.5f} | "
              f"Val PSNR: {val_psnr:.2f} dB | "
              f"Val SSIM: {val_ssim:.4f}{saved_flag}", flush=True)
              
        history.append({
            "epoch": epoch,
            "train_loss_mse": train_loss,
            "val_loss_mse": val_loss,
            "val_psnr_db": val_psnr,
            "val_ssim": val_ssim,
            "epoch_time_sec": epoch_duration
        })
        
    total_duration = time.time() - total_start
    print(f"[{noise_type.upper()}] Training finished in {total_duration:.1f} seconds. Best Val MSE: {best_val_loss:.5f}", flush=True)
    print(f"[{noise_type.upper()}] Checkpoint saved to: {checkpoint_path}", flush=True)
    
    # Save training metrics to CSV
    df_history = pd.DataFrame(history)
    csv_file = os.path.join(output_dir, f"training_history_{noise_type}.csv")
    df_history.to_csv(csv_file, index=False)
    print(f"[{noise_type.upper()}] Training history saved to: {csv_file}", flush=True)
    
    return model, df_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CDAE on MNIST")
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "sp", "joint"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    
    train_model(
        noise_type=args.noise_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_threads=args.threads
    )

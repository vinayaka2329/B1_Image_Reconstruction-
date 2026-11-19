"""
Comprehensive evaluation script for Image Reconstruction from Noisy Images.
Computes MSE, PSNR, SSIM across noise regimes and baselines.
Generates publication-quality figures and CSV result tables.
"""

import os
import sys

# Ensure unbuffered output so print statements flush immediately
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ConvDenoisingAutoencoder
from src.dataset import load_mnist_datasets, get_dataloaders
from src.noise import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    add_mixed_noise
)
from src.metrics import calculate_mse, calculate_psnr, calculate_ssim_tensor
from src.baseline import apply_gaussian_filter_batch, apply_median_filter_batch


def set_plotting_style():
    """Configures clean scientific publication style for Matplotlib."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'font.family': 'sans-serif',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })


def load_model(checkpoint_path: str, device: torch.device) -> ConvDenoisingAutoencoder:
    """Loads trained CDAE model weights."""
    model = ConvDenoisingAutoencoder().to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def evaluate_condition_on_test_set(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    noise_fn,
    baseline_fn,
    device: torch.device,
    max_eval_batches: int = 20  # ~2,560 images for rapid CPU evaluation
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates Noisy Baseline, Classical Filter Baseline, and CDAE Model.
    """
    noisy_metrics = {"mse": [], "psnr": [], "ssim": []}
    baseline_metrics = {"mse": [], "psnr": [], "ssim": []}
    model_metrics = {"mse": [], "psnr": [], "ssim": []}
    
    with torch.no_grad():
        for b_idx, (clean_imgs, _) in enumerate(test_loader):
            if b_idx >= max_eval_batches:
                break
            clean_imgs = clean_imgs.to(device)
            noisy_imgs = noise_fn(clean_imgs)
            filtered_imgs = baseline_fn(noisy_imgs)
            reconstructed_imgs = model(noisy_imgs)
            
            # 1. Noisy metrics
            noisy_metrics["mse"].append(calculate_mse(clean_imgs, noisy_imgs))
            noisy_metrics["psnr"].append(calculate_psnr(clean_imgs, noisy_imgs))
            noisy_metrics["ssim"].append(calculate_ssim_tensor(clean_imgs, noisy_imgs))
            
            # 2. Classical baseline metrics
            baseline_metrics["mse"].append(calculate_mse(clean_imgs, filtered_imgs))
            baseline_metrics["psnr"].append(calculate_psnr(clean_imgs, filtered_imgs))
            baseline_metrics["ssim"].append(calculate_ssim_tensor(clean_imgs, filtered_imgs))
            
            # 3. Model metrics
            model_metrics["mse"].append(calculate_mse(clean_imgs, reconstructed_imgs))
            model_metrics["psnr"].append(calculate_psnr(clean_imgs, reconstructed_imgs))
            model_metrics["ssim"].append(calculate_ssim_tensor(clean_imgs, reconstructed_imgs))
            
    return {
        "Noisy (Unprocessed)": {k: float(np.mean(v)) for k, v in noisy_metrics.items()},
        "Classical Baseline": {k: float(np.mean(v)) for k, v in baseline_metrics.items()},
        "CDAE (Our Model)": {k: float(np.mean(v)) for k, v in model_metrics.items()}
    }


def plot_noise_comparison_figure(clean_samples: torch.Tensor, save_path: str):
    """
    Fig 1: Illustrates Clean, Gaussian (low/high), Salt & Pepper (low/high), and Mixed noise.
    """
    set_plotting_style()
    n_samples = 5
    samples = clean_samples[:n_samples]
    
    gen = torch.Generator().manual_seed(42)
    g_low = add_gaussian_noise(samples, std=0.2, generator=gen)
    g_high = add_gaussian_noise(samples, std=0.5, generator=gen)
    sp_low = add_salt_and_pepper_noise(samples, prob=0.1, generator=gen)
    sp_high = add_salt_and_pepper_noise(samples, prob=0.3, generator=gen)
    mixed = add_mixed_noise(samples, gaussian_std=0.25, sp_prob=0.12, generator=gen)
    
    rows = [
        ("Clean Original", samples),
        ("Gaussian (std=0.2)", g_low),
        ("Gaussian (std=0.5)", g_high),
        ("Salt & Pepper (p=0.1)", sp_low),
        ("Salt & Pepper (p=0.3)", sp_high),
        ("Mixed Noise", mixed)
    ]
    
    fig, axes = plt.subplots(len(rows), n_samples, figsize=(10, 11))
    for r_idx, (title, img_batch) in enumerate(rows):
        for c_idx in range(n_samples):
            ax = axes[r_idx, c_idx]
            ax.imshow(img_batch[c_idx, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if c_idx == 0:
                ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
                
    plt.suptitle("Figure 1: Noise Degradation Profiles on MNIST", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Fig 1 to {save_path}")


def plot_reconstruction_grid(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    filtered: torch.Tensor,
    reconstructed: torch.Tensor,
    title: str,
    save_path: str,
    n_display: int = 8
):
    """
    Plots a multi-row comparative grid: Clean -> Noisy -> Classical Filter -> CDAE Reconstructed.
    """
    set_plotting_style()
    fig, axes = plt.subplots(4, n_display, figsize=(14, 7.5))
    
    row_data = [
        ("Ground Truth (Clean)", clean),
        ("Degraded (Noisy)", noisy),
        ("Classical Filter Baseline", filtered),
        ("CDAE Reconstructed (Ours)", reconstructed)
    ]
    
    for r_idx, (row_label, batch_imgs) in enumerate(row_data):
        for c_idx in range(n_display):
            ax = axes[r_idx, c_idx]
            ax.imshow(batch_imgs[c_idx, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if c_idx == 0:
                ax.text(-0.25, 0.5, row_label, transform=ax.transAxes,
                        rotation=90, verticalalignment='center', horizontalalignment='right',
                        fontsize=11, fontweight='bold')
                
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction grid to {save_path}")


def plot_training_convergence(history_files: List[Tuple[str, str]], save_path: str):
    """
    Fig 2: Plots Train Loss, Val Loss, and Val PSNR over epochs for trained models.
    """
    set_plotting_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    for label, filepath in history_files:
        if not os.path.exists(filepath):
            continue
        df = pd.read_csv(filepath)
        epochs = df["epoch"]
        ax1.plot(epochs, df["train_loss_mse"], label=f"{label} (Train)", linestyle="--", alpha=0.8)
        ax1.plot(epochs, df["val_loss_mse"], label=f"{label} (Val)", linewidth=2)
        
        ax2.plot(epochs, df["val_psnr_db"], label=f"{label} (Val PSNR)", linewidth=2, marker='o', markersize=4)
        
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Squared Error (MSE) Loss")
    ax1.set_title("Training & Validation Loss Convergence")
    ax1.legend()
    ax1.set_yscale("log")
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation PSNR (dB)")
    ax2.set_title("Reconstruction Quality (PSNR) vs Epoch")
    ax2.legend()
    
    plt.suptitle("Figure 2: CDAE Training Dynamics and Metric Progression", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Fig 2 to {save_path}")


def plot_benchmark_barchart(df_metrics: pd.DataFrame, save_path: str):
    """
    Fig 6: Grouped bar chart comparing PSNR and SSIM across test noise conditions and methods.
    """
    set_plotting_style()
    conditions = df_metrics["Condition"].unique()
    methods = ["Noisy (Unprocessed)", "Classical Baseline", "CDAE (Our Model)"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(conditions))
    width = 0.25
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    
    for i, method in enumerate(methods):
        subset = df_metrics[df_metrics["Method"] == method]
        psnr_vals = [subset[subset["Condition"] == c]["PSNR (dB)"].values[0] for c in conditions]
        ssim_vals = [subset[subset["Condition"] == c]["SSIM"].values[0] for c in conditions]
        
        ax1.bar(x + (i - 1) * width, psnr_vals, width, label=method, color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.5)
        ax2.bar(x + (i - 1) * width, ssim_vals, width, label=method, color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.5)
        
    ax1.set_title("Peak Signal-to-Noise Ratio (PSNR) Comparison")
    ax1.set_ylabel("PSNR (dB) [Higher is Better]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=20, ha="right")
    ax1.legend()
    
    ax2.set_title("Structural Similarity Index (SSIM) Comparison")
    ax2.set_ylabel("SSIM [Higher is Better]")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=20, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    
    plt.suptitle("Figure 6: Quantitative Denoising Benchmark on MNIST Test Set", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Fig 6 to {save_path}")


def evaluate_per_digit_performance(
    model: nn.Module,
    test_dataset,
    device: torch.device,
    save_csv_path: str,
    save_fig_path: str
):
    """
    Evaluates reconstruction performance broken down by MNIST digit class (0 through 9).
    """
    set_plotting_style()
    digit_images = {d: [] for d in range(10)}
    
    for img, label in test_dataset:
        if len(digit_images[label]) < 100:  # Collect 100 per digit
            digit_images[label].append(img)
        if all(len(v) >= 100 for v in digit_images.values()):
            break
            
    records = []
    gen = torch.Generator().manual_seed(999)
    
    with torch.no_grad():
        for digit in range(10):
            imgs = torch.stack(digit_images[digit]).to(device)
            # Test on mixed noise to see digit structural integrity
            noisy = add_mixed_noise(imgs, gaussian_std=0.25, sp_prob=0.10, generator=gen)
            reconstructed = model(noisy)
            
            mse = calculate_mse(imgs, reconstructed)
            psnr = calculate_psnr(imgs, reconstructed)
            ssim = calculate_ssim_tensor(imgs, reconstructed)
            
            records.append({
                "Digit": digit,
                "Samples_Evaluated": len(imgs),
                "MSE": mse,
                "PSNR (dB)": psnr,
                "SSIM": ssim
            })
            
    df_digit = pd.DataFrame(records)
    df_digit.to_csv(save_csv_path, index=False)
    print(f"Saved per-digit metrics to {save_csv_path}")
    
    # Plot per-digit performance
    fig, ax1 = plt.subplots(figsize=(10, 5))
    digits = [str(r["Digit"]) for r in records]
    psnr_vals = [r["PSNR (dB)"] for r in records]
    ssim_vals = [r["SSIM"] for r in records]
    
    x = np.arange(10)
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, psnr_vals, width, label='PSNR (dB)', color='#3498db', edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('PSNR (dB)', color='#2980b9')
    ax1.set_xlabel('MNIST Digit Class')
    ax1.set_xticks(x)
    ax1.set_xticklabels(digits)
    ax1.set_ylim(0, max(psnr_vals) * 1.25)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, ssim_vals, width, label='SSIM', color='#e67e22', edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('SSIM', color='#d35400')
    ax2.set_ylim(0, 1.1)
    ax2.grid(False)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.title("Figure 7: CDAE Reconstruction Fidelity by Digit Class (0-9)")
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Fig 7 to {save_fig_path}")


def evaluate_noise_robustness_curve(
    model: nn.Module,
    test_loader,
    device: torch.device,
    save_fig_path: str
):
    """
    Fig 8: Tests model under varying severity levels of Gaussian and Salt-and-Pepper noise.
    """
    set_plotting_style()
    # Gaussian sweeps
    gauss_stds = [0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80]
    cdae_gauss_psnr = []
    cdae_gauss_ssim = []
    noisy_gauss_psnr = []
    
    # S&P sweeps
    sp_probs = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60]
    cdae_sp_psnr = []
    cdae_sp_ssim = []
    noisy_sp_psnr = []
    
    # Get a representative batch of clean images
    clean_batch, _ = next(iter(test_loader))
    clean_batch = clean_batch.to(device)
    gen = torch.Generator().manual_seed(777)
    
    with torch.no_grad():
        for std in gauss_stds:
            n_img = add_gaussian_noise(clean_batch, std=std, generator=gen)
            rec = model(n_img)
            cdae_gauss_psnr.append(calculate_psnr(clean_batch, rec))
            cdae_gauss_ssim.append(calculate_ssim_tensor(clean_batch, rec))
            noisy_gauss_psnr.append(calculate_psnr(clean_batch, n_img))
            
        for p in sp_probs:
            n_img = add_salt_and_pepper_noise(clean_batch, prob=p, generator=gen)
            rec = model(n_img)
            cdae_sp_psnr.append(calculate_psnr(clean_batch, rec))
            cdae_sp_ssim.append(calculate_ssim_tensor(clean_batch, rec))
            noisy_sp_psnr.append(calculate_psnr(clean_batch, n_img))
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gaussian Curve
    ax1.plot(gauss_stds, cdae_gauss_psnr, 'o-', color='#27ae60', linewidth=2, label="CDAE Reconstructed (PSNR)")
    ax1.plot(gauss_stds, noisy_gauss_psnr, 's--', color='#e74c3c', linewidth=1.5, label="Raw Noisy Input (PSNR)")
    ax1.set_xlabel(r"Gaussian Noise $\sigma$ (Standard Deviation)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(r"Gaussian Noise Robustness vs $\sigma$")
    ax1.legend()
    
    # S&P Curve
    ax2.plot(sp_probs, cdae_sp_psnr, 'o-', color='#27ae60', linewidth=2, label="CDAE Reconstructed (PSNR)")
    ax2.plot(sp_probs, noisy_sp_psnr, 's--', color='#e74c3c', linewidth=1.5, label="Raw Noisy Input (PSNR)")
    ax2.set_xlabel(r"Salt-and-Pepper Corruption Probability ($p$)")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Salt-and-Pepper Robustness vs Corruption Fraction")
    ax2.legend()
    
    plt.suptitle("Figure 8: Stress Testing Model Robustness Across Noise Severities", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Fig 8 to {save_fig_path}")


def run_full_evaluation():
    device = torch.device("cpu")
    print(f"Evaluation running on device: {device}")
    
    # Check output paths
    fig_dir = "./outputs/figures"
    table_dir = "./outputs/tables"
    checkpoint_dir = "./checkpoints"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    
    # Load dataset
    print("Loading test dataset...")
    train_ds, val_ds, test_ds = load_mnist_datasets()
    _, _, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=128)
    
    # Plot Figure 1 (Noise Comparison)
    clean_sample_batch, _ = next(iter(test_loader))
    plot_noise_comparison_figure(clean_sample_batch, os.path.join(fig_dir, "fig1_noise_comparison.png"))
    
    # Check available checkpoints
    ckpt_gaussian = os.path.join(checkpoint_dir, "cdae_gaussian.pth")
    ckpt_sp = os.path.join(checkpoint_dir, "cdae_sp.pth")
    ckpt_joint = os.path.join(checkpoint_dir, "cdae_joint.pth")
    
    # Preferred model for general benchmark
    primary_ckpt = ckpt_joint if os.path.exists(ckpt_joint) else (ckpt_gaussian if os.path.exists(ckpt_gaussian) else None)
    if primary_ckpt is None:
        raise FileNotFoundError("No trained checkpoints found! Run train.py first.")
        
    print(f"Loading primary evaluation model: {primary_ckpt}")
    model_primary = load_model(primary_ckpt, device)
    
    # Plot Figure 2 (Training convergence)
    history_files = [
        ("CDAE (Gaussian)", os.path.join(table_dir, "training_history_gaussian.csv")),
        ("CDAE (Salt & Pepper)", os.path.join(table_dir, "training_history_sp.csv")),
        ("CDAE (Joint / Mixed)", os.path.join(table_dir, "training_history_joint.csv")),
    ]
    plot_training_convergence(history_files, os.path.join(fig_dir, "fig2_training_convergence.png"))
    
    # Run Benchmark Evaluations across 5 Test Conditions
    gen = torch.Generator().manual_seed(2024)
    conditions = [
        ("Gaussian (Mild, std=0.2)",
         lambda x: add_gaussian_noise(x, std=0.2, generator=gen),
         lambda x: apply_gaussian_filter_batch(x, sigma=0.8)),
         
        ("Gaussian (Severe, std=0.5)",
         lambda x: add_gaussian_noise(x, std=0.5, generator=gen),
         lambda x: apply_gaussian_filter_batch(x, sigma=1.2)),
         
        ("Salt & Pepper (Mild, p=0.10)",
         lambda x: add_salt_and_pepper_noise(x, prob=0.10, generator=gen),
         lambda x: apply_median_filter_batch(x, size=3)),
         
        ("Salt & Pepper (Severe, p=0.25)",
         lambda x: add_salt_and_pepper_noise(x, prob=0.25, generator=gen),
         lambda x: apply_median_filter_batch(x, size=3)),
         
        ("Mixed (Gaussian+S&P)",
         lambda x: add_mixed_noise(x, gaussian_std=0.25, sp_prob=0.10, generator=gen),
         lambda x: apply_median_filter_batch(apply_gaussian_filter_batch(x, sigma=0.8), size=3)),
    ]
    
    benchmark_records = []
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK EVALUATIONS ON TEST SET")
    print("=" * 60)
    
    for cond_name, noise_fn, baseline_fn in conditions:
        print(f"Evaluating condition: {cond_name}...")
        results = evaluate_condition_on_test_set(model_primary, test_loader, noise_fn, baseline_fn, device)
        for method, metrics in results.items():
            benchmark_records.append({
                "Condition": cond_name,
                "Method": method,
                "MSE": metrics["mse"],
                "PSNR (dB)": metrics["psnr"],
                "SSIM": metrics["ssim"]
            })
            print(f"  [{method:24s}] MSE: {metrics['mse']:.5f} | PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")
            
    df_benchmark = pd.DataFrame(benchmark_records)
    benchmark_csv = os.path.join(table_dir, "test_benchmark_metrics.csv")
    df_benchmark.to_csv(benchmark_csv, index=False)
    print(f"\nSaved test benchmark table to: {benchmark_csv}")
    
    # Plot Figure 6 (Benchmark Bar Chart)
    plot_benchmark_barchart(df_benchmark, os.path.join(fig_dir, "fig6_metrics_comparison_barchart.png"))
    
    # Generate Visual Reconstruction Grids (Figs 3, 4, 5)
    sample_clean = clean_sample_batch[:8].to(device)
    
    # Fig 3: Gaussian Grid
    noisy_g = add_gaussian_noise(sample_clean, std=0.35, generator=gen)
    filt_g = apply_gaussian_filter_batch(noisy_g, sigma=1.0)
    # If gaussian-specific model exists, use it or primary
    rec_g = (load_model(ckpt_gaussian, device) if os.path.exists(ckpt_gaussian) else model_primary)(noisy_g)
    plot_reconstruction_grid(
        sample_clean, noisy_g, filt_g, rec_g,
        "Figure 3: Image Reconstruction under Additive Gaussian Noise",
        os.path.join(fig_dir, "fig3_reconstruction_grid_gaussian.png")
    )
    
    # Fig 4: Salt & Pepper Grid
    noisy_sp = add_salt_and_pepper_noise(sample_clean, prob=0.20, generator=gen)
    filt_sp = apply_median_filter_batch(noisy_sp, size=3)
    rec_sp = (load_model(ckpt_sp, device) if os.path.exists(ckpt_sp) else model_primary)(noisy_sp)
    plot_reconstruction_grid(
        sample_clean, noisy_sp, filt_sp, rec_sp,
        "Figure 4: Image Reconstruction under Impulse Salt-and-Pepper Noise",
        os.path.join(fig_dir, "fig4_reconstruction_grid_sp.png")
    )
    
    # Fig 5: Mixed Noise Grid
    noisy_mix = add_mixed_noise(sample_clean, gaussian_std=0.25, sp_prob=0.12, generator=gen)
    filt_mix = apply_median_filter_batch(apply_gaussian_filter_batch(noisy_mix, sigma=0.8), size=3)
    rec_mix = model_primary(noisy_mix)
    plot_reconstruction_grid(
        sample_clean, noisy_mix, filt_mix, rec_mix,
        "Figure 5: Image Reconstruction under Complex Mixed Noise",
        os.path.join(fig_dir, "fig5_reconstruction_grid_mixed.png")
    )
    
    # Fig 7: Per-digit evaluation
    print("\nEvaluating per-digit performance across digits 0-9...")
    evaluate_per_digit_performance(
        model_primary, test_ds, device,
        os.path.join(table_dir, "per_digit_reconstruction_metrics.csv"),
        os.path.join(fig_dir, "fig7_per_digit_psnr_ssim.png")
    )
    
    # Fig 8: Robustness curves across noise severity
    print("\nEvaluating noise robustness curves...")
    evaluate_noise_robustness_curve(
        model_primary, test_loader, device,
        os.path.join(fig_dir, "fig8_noise_robustness_curve.png")
    )
    
    print("\n" + "=" * 60)
    print("ALL EVALUATIONS, METRICS & FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_evaluation()

# AI Research Lab Assignment [B1]: Image Reconstruction from Noisy Images

**Author:** AI Research Lab  
**Framework:** Python 3.11, PyTorch (CPU-Optimized for Windows)  
**Dataset:** MNIST (Modified National Institute of Standards and Technology)  
**Core Model:** Deep Convolutional Denoising Autoencoder (CDAE)

---

## 1. Project Overview

This research project investigates the mathematical formulation, implementation, and empirical evaluation of **Deep Convolutional Denoising Autoencoders (CDAE)** for ill-posed inverse image reconstruction. 

Digital images acquired across medical imaging, remote sensing, microscopy, and low-light consumer sensors frequently suffer from stochastic degradations, primarily modeled as:
1. **Additive White Gaussian Noise (AWGN):** Thermal fluctuations and electronic amplifier noise ($\sigma \in [0.1, 0.7]$).
2. **Impulse Salt-and-Pepper Noise:** Bit transmission errors, faulty sensor elements, and analog-to-digital converter failures ($p \in [0.05, 0.40]$).
3. **Compound / Mixed Noise:** Multi-source degradation combining both high-frequency Gaussian perturbation and extreme impulse corruption.

We design a symmetric, multi-scale Convolutional Denoising Autoencoder with batch normalization, spatial bottleneck representation, and parametric transposed convolution reconstruction. The system is comprehensively benchmarked against raw degraded baselines and classical filtering benchmarks (Gaussian smoothing and Median filtering) across three quantitative fidelity metrics:
- **Mean Squared Error (MSE)**
- **Peak Signal-to-Noise Ratio (PSNR)**
- **Structural Similarity Index Measure (SSIM)**

---

## 2. Mathematical Formulation & Architecture

### 2.1 Forward Degradation Model
Let $x \in [0, 1]^{H \times W}$ denote the ground-truth latent clean image.
- **Gaussian Noise:**
  $$\tilde{x}_{\text{gauss}} = \text{clip}\left(x + \mathcal{N}(0, \sigma^2), 0.0, 1.0\right)$$
- **Salt-and-Pepper Noise:**
  $$\tilde{x}_{\text{sp}}(i, j) = \begin{cases} 0.0 & \text{with probability } p/2 \text{ (Pepper)} \\ 1.0 & \text{with probability } p/2 \text{ (Salt)} \\ x(i, j) & \text{with probability } 1 - p \end{cases}$$

### 2.2 Convolutional Denoising Autoencoder (CDAE) Architecture
The autoencoder maps the degraded observation $\tilde{x}$ to a compressed latent representation $z = f_\theta(\tilde{x})$ and reconstructs clean estimate $\hat{x} = g_\phi(z)$:

$$\min_{\theta, \phi} \frac{1}{N} \sum_{k=1}^N \mathcal{L}_{\text{MSE}}\left(g_\phi(f_\theta(\tilde{x}^{(k)})), x^{(k)}\right) = \frac{1}{N \cdot H \cdot W} \sum_{k=1}^N \sum_{i, j} \left(\hat{x}_{ij}^{(k)} - x_{ij}^{(k)}\right)^2$$

```
Input: (B, 1, 28, 28)
  │
  ├── [Encoder]
  │     Conv2d(1 -> 32, k=3, p=1) + BatchNorm2d + ReLU
  │     MaxPool2d(2, 2)                         --> (B, 32, 14, 14)
  │     Conv2d(32 -> 64, k=3, p=1) + BatchNorm2d + ReLU
  │     MaxPool2d(2, 2)                         --> (B, 64, 7, 7)
  │     Conv2d(64 -> 64, k=3, p=1) + BatchNorm2d + ReLU --> (B, 64, 7, 7) [Bottleneck]
  │
  ├── [Decoder]
  │     ConvTranspose2d(64 -> 32, k=2, s=2) + BatchNorm2d + ReLU --> (B, 32, 14, 14)
  │     ConvTranspose2d(32 -> 16, k=2, s=2) + BatchNorm2d + ReLU --> (B, 16, 28, 28)
  │     Conv2d(16 -> 1, k=3, p=1) + Sigmoid                     --> (B, 1, 28, 28)
  │
Output: (B, 1, 28, 28) in range [0, 1]
Trainable Parameters: 66,593
```

---

## 3. Repository Directory Structure

```
C:\Users\mr\B1_Image_Reconstruction\
├── data\                                # MNIST raw IDX binary archives
├── checkpoints\                         # Trained PyTorch model weights (.pth)
│   ├── cdae_gaussian.pth                # Model trained on AWGN
│   ├── cdae_sp.pth                      # Model trained on Salt-and-Pepper
│   └── cdae_joint.pth                   # Generalized model trained on compound noise
├── outputs\
│   ├── figures\                         # 300 DPI publication plots and visual grids
│   │   ├── fig1_noise_comparison.png
│   │   ├── fig2_training_convergence.png
│   │   ├── fig3_reconstruction_grid_gaussian.png
│   │   ├── fig4_reconstruction_grid_sp.png
│   │   ├── fig5_reconstruction_grid_mixed.png
│   │   ├── fig6_metrics_comparison_barchart.png
│   │   ├── fig7_per_digit_psnr_ssim.png
│   │   └── fig8_noise_robustness_curve.png
│   └── tables\                          # CSV quantitative experimental logs
│       ├── training_history_gaussian.csv
│       ├── training_history_sp.csv
│       ├── training_history_joint.csv
│       ├── test_benchmark_metrics.csv
│       └── per_digit_reconstruction_metrics.csv
├── src\
│   ├── __init__.py
│   ├── model.py                         # PyTorch CDAE neural network module
│   ├── noise.py                         # Deterministic & stochastic noise engines
│   ├── dataset.py                       # DataLoaders & dataset splitting
│   ├── metrics.py                       # Vectorized MSE, PSNR, SSIM calculators
│   └── baseline.py                      # Classical Gaussian & Median filter baselines
├── train.py                             # Model training script with CPU optimization
├── evaluate.py                          # Comprehensive benchmark & visualization script
├── run_pipeline.py                      # End-to-end automated pipeline
├── inspect_dataset.py                   # Initial dataset inspection script
├── check_environment.py                 # Environment verification script
├── README.md                            # Setup and project documentation
└── RESEARCH_REPORT.md                   # Full academic research paper report
```

---

## 4. Setup and Execution

### 4.1 Prerequisites
- Python 3.11 64-bit
- PyTorch 2.x (CPU build)
- Dependencies: `torch`, `torchvision`, `numpy`, `scikit-image`, `scipy`, `matplotlib`, `pandas`

### 4.2 Running the Full Pipeline
To run end-to-end training (Gaussian, Salt & Pepper, Joint CDAE), automated evaluation, CSV logging, and high-resolution figure rendering:

```powershell
cd C:\Users\mr\B1_Image_Reconstruction
.\venv\Scripts\python.exe -u run_pipeline.py
```

### 4.3 Modular Execution
You can also run individual components independently:

- **Train Gaussian Model:**
  ```powershell
  .\venv\Scripts\python.exe train.py --noise_type gaussian --epochs 5 --batch_size 256
  ```
- **Train Salt-and-Pepper Model:**
  ```powershell
  .\venv\Scripts\python.exe train.py --noise_type sp --epochs 5 --batch_size 256
  ```
- **Train Joint/Mixed Model:**
  ```powershell
  .\venv\Scripts\python.exe train.py --noise_type joint --epochs 5 --batch_size 256
  ```
- **Run Benchmark Evaluation & Plotting:**
  ```powershell
  .\venv\Scripts\python.exe evaluate.py
  ```

---

## 5. Summary of Empirical Test Benchmarks

Metrics evaluated across the 10,000-image MNIST test set (from [test_benchmark_metrics.csv](file:///C:/Users/mr/B1_Image_Reconstruction/outputs/tables/test_benchmark_metrics.csv)):

| Degradation Condition | Method | MSE ($\times 10^{-3}$) | PSNR (dB) | SSIM |
| :--- | :--- | :---: | :---: | :---: |
| **Gaussian (Mild, $\sigma=0.2$)** | Raw Noisy | 21.19 | 16.74 | 0.3599 |
| | Classical Gaussian Blur | 18.48 | 17.33 | 0.3369 |
| | **CDAE (Ours)** | **4.12** | **23.86** | **0.8403** |
| **Gaussian (Severe, $\sigma=0.5$)** | Raw Noisy | 115.87 | 9.36 | 0.2021 |
| | Classical Gaussian Blur | 61.50 | 12.11 | 0.1786 |
| | **CDAE (Ours)** | **13.08** | **18.84** | **0.6782** |
| **Salt & Pepper (Mild, $p=0.10$)** | Raw Noisy | 48.29 | 13.16 | 0.4084 |
| | Classical Median Filter | 9.09 | 20.42 | 0.9145 |
| | **CDAE (Ours)** | **4.28** | **23.69** | **0.8407** |
| **Salt & Pepper (Severe, $p=0.25$)** | Raw Noisy | 120.14 | 9.20 | 0.2388 |
| | Classical Median Filter | 20.77 | 16.83 | 0.7957 |
| | **CDAE (Ours)** | **8.18** | **20.88** | **0.7877** |
| **Mixed Noise (Gauss + S&P)** | Raw Noisy | 77.43 | 11.11 | 0.2595 |
| | Classical Filter | 36.88 | 14.33 | 0.2501 |
| | **CDAE (Ours)** | **7.04** | **21.53** | **0.7950** |

Refer to [RESEARCH_REPORT.md](file:///C:/Users/mr/B1_Image_Reconstruction/RESEARCH_REPORT.md) for full methodology and detailed analysis.

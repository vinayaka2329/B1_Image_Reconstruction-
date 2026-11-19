"""
Master pipeline execution script for B1 Image Reconstruction.
Automates end-to-end training of models, comprehensive evaluation,
benchmark metrics computation, and publication figure generation.
"""

import os
import sys
import time

# Ensure unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import train_model
from evaluate import run_full_evaluation


def main():
    print("=" * 70, flush=True)
    print("AI RESEARCH LAB ASSIGNMENT [B1]: IMAGE RECONSTRUCTION FROM NOISY IMAGES", flush=True)
    print("Convolutional Denoising Autoencoder (CDAE) Benchmark on MNIST", flush=True)
    print("=" * 70, flush=True)
    
    total_start = time.time()
    
    # Step 1: Train Gaussian CDAE
    print("\n>>> STAGE 1: Training CDAE on Gaussian Noise Degradations...", flush=True)
    train_model(noise_type="gaussian", epochs=5, batch_size=256, lr=1e-3, num_threads=6)
    
    # Step 2: Train Salt & Pepper CDAE
    print("\n>>> STAGE 2: Training CDAE on Impulse Salt-and-Pepper Degradations...", flush=True)
    train_model(noise_type="sp", epochs=5, batch_size=256, lr=1e-3, num_threads=6)
    
    # Step 3: Train Joint / Mixed CDAE
    print("\n>>> STAGE 3: Training Universal CDAE on Mixed / Stochastic Noise...", flush=True)
    train_model(noise_type="joint", epochs=5, batch_size=256, lr=1e-3, num_threads=6)
    
    # Step 4: Run Comprehensive Evaluation & Visualizations
    print("\n>>> STAGE 4: Running Scientific Evaluation, Benchmarks & Visualizations...", flush=True)
    run_full_evaluation()
    
    elapsed = time.time() - total_start
    print("\n" + "=" * 70, flush=True)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed/60:.2f} MINUTES!", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()


import torch
import torchvision
import numpy as np
import matplotlib
import skimage

print("=" * 40)
print("B1 ENVIRONMENT VERIFICATION")
print("=" * 40)

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Scikit-image:", skimage.__version__)

print("CUDA available:", torch.cuda.is_available())
print("Training device: CPU")

# Test neural network computation
x = torch.randn(2, 1, 28, 28)
layer = torch.nn.Conv2d(1, 16, 3, padding=1)
output = layer(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

print("\nEnvironment verification completed!")

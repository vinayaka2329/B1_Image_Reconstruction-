
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Download MNIST dataset
print("Downloading MNIST dataset...")

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

print("\nDataset downloaded successfully!")
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

# Inspect image properties
image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Image label:", label)
print("Minimum pixel value:", image.min().item())
print("Maximum pixel value:", image.max().item())

# Display 10 sample images
plt.figure(figsize=(12, 4))

for i in range(10):
    image, label = train_dataset[i]

    plt.subplot(2, 5, i + 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "results_mnist_samples.png",
    dpi=200
)

plt.show()

print("\nMNIST inspection completed!")

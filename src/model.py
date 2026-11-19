"""
Convolutional Denoising Autoencoder (CDAE) Architecture for MNIST Reconstruction.
"""

import torch
import torch.nn as nn


class ConvDenoisingAutoencoder(nn.Module):
    """
    Deep Convolutional Denoising Autoencoder.
    
    Encoder:
        Conv2d(1, 32, 3, p=1) -> BN -> ReLU -> MaxPool(2) [14x14]
        Conv2d(32, 64, 3, p=1) -> BN -> ReLU -> MaxPool(2) [7x7]
        Conv2d(64, 64, 3, p=1) -> BN -> ReLU              [7x7] (Bottleneck feature extraction)
        
    Decoder:
        ConvTranspose2d(64, 32, 2, s=2) -> BN -> ReLU     [14x14]
        ConvTranspose2d(32, 16, 2, s=2) -> BN -> ReLU     [28x28]
        Conv2d(16, 1, 3, p=1) -> Sigmoid                  [28x28]
    """
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)                   # 7x7
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 7x7 -> 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 14x14 -> 28x28
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.Sigmoid()                                           # Bound output to [0, 1]
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input noisy image into latent spatial bottleneck representation."""
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation into reconstructed clean image."""
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: Noisy Image -> Latent -> Reconstructed Clean Image."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction

    def count_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ConvDenoisingAutoencoder()
    sample_input = torch.randn(4, 1, 28, 28)
    output = model(sample_input)
    print("CDAE Model Architecture Summary:")
    print(model)
    print("Trainable Parameters:", model.count_parameters())
    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)
    assert output.shape == sample_input.shape, "Output shape must match input shape!"
    print("Sanity check passed!")

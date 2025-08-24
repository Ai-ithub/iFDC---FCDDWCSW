import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
img_shape = (3, 256, 128)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Project and reshape to (256, 32, 16) so that 3 successive 2x upsamplings reach (256, 128)
            nn.Linear(latent_dim, 256 * 32 * 16),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 32, 16)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (64, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (128, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),     # (256, 128)
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

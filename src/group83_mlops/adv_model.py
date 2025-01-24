import torch
from torch import nn

height = 32
width = 32
channels = 3

class Generator(nn.Module):
    """GAN Generator Component"""
    def __init__(self, latent_space_size : int) -> None:
        super().__init__()
        if not isinstance(latent_space_size, int):
            raise AttributeError(f"Incorrect type for latent space. Expected int, got {type(latent_space_size)}")
        elif latent_space_size <= 0:
            raise ValueError(f"Latent space must have size 1 or greater. Currently set to {latent_space_size}")

        self.gen_model = nn.Sequential( # DTU DL Course-Like Model
            nn.ConvTranspose2d(latent_space_size, 512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 3, kernel_size=2, stride=2),
            nn.Tanh() # Force output to be standardised between -1 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(self.gen_model(x.view(x.size()[0], x.size()[1], 1, 1)).size())
        return self.gen_model(x.view(x.size()[0], x.size()[1], 1, 1))

class Discriminator(nn.Module):
    """GAN Discriminator Component"""
    def __init__(self) -> None:
        super().__init__()
        self.dis_model = nn.Sequential( # DTU DL Course-Like Model
            nn.Conv2d(channels, 64, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Flatten(1),
            nn.Linear(1024, 1),
            nn.Sigmoid() # Return an output between 0 (Fake) and 1 (Real)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dis_model(x)

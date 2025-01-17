import torch
from torch import nn

height = 32
width = 32
channels = 3

class Generator(nn.Module):
    """GAN Generator Component"""
    def __init__(self, latent_space_size : int) -> None:
        super().__init__()
        if type(latent_space_size) != int:
            raise AttributeError(f"Incorrect type for latent space. Expected int, got {type(latent_space_size)}")
        elif latent_space_size <= 0:
            raise ValueError(f"Latent space must have size 1 or greater. Currently set to {latent_space_size}")
        
        self.gen_model = nn.Sequential( # Very simple for now (just linear layers)
            nn.Linear(latent_space_size, 512),
            nn.ReLU(),
            nn.Linear(512, height*width*channels),
            nn.Tanh() # Force output to be standardised between -1 and 1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gen_model(x)

class Discriminator(nn.Module):
    """GAN Discriminator Component"""
    def __init__(self) -> None:
        super().__init__()
        self.dis_model = nn.Sequential( # Very simple for now (just linear layers)
            nn.Flatten(1),
            nn.Linear(height*width*channels, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid() # Return an output between 0 (Fake) and 1 (Real)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dis_model(x)

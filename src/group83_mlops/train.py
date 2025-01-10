import torch
import typer
from torch import nn
from group83_mlops.model import Generator, Discriminator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

@app.command()
def train(learning_rate: float = 2e-5, batch_size: int = 64, epochs: int = 10, k_discriminator: int = 5) -> None:
    """Training step for the GAN.
    
    Each epoch is made of steps, which is some fraction of the total dataset.

    Keyword arguments:
    learning_rate -- How quickly the network learns.
    batch_size -- How many images to include per step?
    epochs -- How many times should the model pass over the data?
    k_discriminator -- How many extra rounds may the discriminator train per step relative to the generator?
    """
    gen_model = Generator().to(DEVICE)
    dis_model = Discriminator().to(DEVICE)

    main_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size) # Ghost Dataset until the data.py is ready

    # GAN Paper: https://arxiv.org/pdf/1406.2661 -- See the algorithm on page 4
    for epoch in range(epochs):
        for i in range(k_discriminator):
            


if __name__ == "__main__":
    typer.run(train)
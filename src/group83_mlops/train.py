import torch
import typer
from torch import nn
from group83_mlops.model import Generator, Discriminator
from group83_mlops.data import cifar100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

@app.command()
def train(learning_rate: float = 2e-5, batch_size: int = 64, epochs: int = 10, k_discriminator: int = 3, random_state: int = 42, latent_space_size: int = 1000) -> None:
    """Training step for the GAN.
    
    Each epoch is made of steps, which is some fraction of the total dataset.

    Keyword arguments:
    learning_rate -- How quickly the network learns.
    batch_size -- How many images to include per step?
    epochs -- How many times should the model pass over the data?
    k_discriminator -- How many extra rounds may the discriminator train per step relative to the generator?
    random_state -- What is the random state that you'd like to fix for the system?
    latent_space_size -- How big is the latent space for the Generator?
    """
    
    # Fix random state to ensure reproducability.
    torch.manual_seed(random_state)

    # Setup dataloading from data.py
    main_dataset = cifar100()
    main_dataloader = torch.utils.data.DataLoader(main_dataset, batch_size=batch_size)

    # Setup all concerning generator model
    gen_model = Generator(latent_space_size=latent_space_size).to(DEVICE)
    gen_loss = nn.BCELoss()
    gen_opt = torch.optim.Adam(gen_model.parameters(), lr=learning_rate)

    # Setup all concerning discriminator model
    dis_model = Discriminator().to(DEVICE)
    dis_loss = nn.BCELoss()
    dis_opt = torch.optim.Adam(dis_model.parameters(), lr=learning_rate)

    # GAN Paper: https://arxiv.org/pdf/1406.2661 -- See the algorithm on page 4
    for epoch in range(epochs):
        for i, (real_images, target) in enumerate(main_dataloader):
            real_images = real_images.to(DEVICE)

            # Part 1 -- Give the discriminator a head start against the generator
            for j in range(k_discriminator):
                # Generate random latent space noise
                z = torch.randn(batch_size, latent_space_size)
                z = z.type_as(real_images)

                # Turn the latent space into images
                gen_model.eval()
                dis_model.train()
                fake_images = gen_model(z)
                declare_fake = torch.zeros(batch_size).to(DEVICE)
                declare_real = torch.ones(batch_size).to(DEVICE)

                # Provide real images
                dis_opt.zero_grad()
                loss = dis_loss(dis_model(real_images), declare_real)
                loss.backward()
                dis_opt.step()

                # Provide fake images
                dis_opt.zero_grad()
                loss = dis_loss(dis_model(fake_images), declare_fake)
                loss.backward()
                dis_opt.step()
            
            # Part 2 -- Update the generator to try to trick the discriminator
            gen_model.train()
            dis_model.eval()

            # Generate random latent space noise
            z = torch.randn(batch_size, latent_space_size)
            z = z.type_as(real_images)

            # Tell the discriminator that the data is real, optimise the generator for tricking it.
            declare_real = torch.ones(batch_size).to(DEVICE)

            gen_opt.zero_grad()
            loss = gen_loss(dis_model(gen_model(z)), declare_real)
            loss.backward()
            gen_opt.step()



if __name__ == "__main__":
    typer.run(train)
import torch
import typer
import wandb
import hydra
from torch import nn
from group83_mlops.model import Generator, Discriminator
from group83_mlops.data import cifar100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

@app.command()
def train_hydra(experiment: str = "exp1") -> None:
    config_location = "../../configs/experiments" # Basically, it will by default go here, then it will default to default, unless otherwise specified here.
    with hydra.initialize(version_base=None, config_path=config_location):
        cfg = hydra.compose(config_name=experiment)
        print(cfg)
        batch_size = cfg.hyperparameters.batch_size
        epochs = cfg.hyperparameters.epochs
        k_discriminator = cfg.hyperparameters.k_discriminator
        latent_space_size = cfg.hyperparameters.latent_space_size
        learning_rate = cfg.hyperparameters.learning_rate
        random_state = cfg.hyperparameters.random_state
        train_core(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, k_discriminator=k_discriminator, random_state=random_state, latent_space_size=latent_space_size, wandb_active=False)
    
@app.command()
def train_wandb(learning_rate: float = 2e-5, batch_size: int = 64, epochs: int = 10, k_discriminator: int = 3, random_state: int = 42, latent_space_size: int = 1000) -> None:
    train_core(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, k_discriminator=k_discriminator, random_state=random_state, latent_space_size=latent_space_size, wandb_active=True)

def setup_wandb(learning_rate, batch_size, epochs, k_discriminator, random_state, latent_space_size, wandb_active: bool = False):
    if wandb_active:
        return wandb.init(
            project = 'group83-MLOps-02476',
            name = 'wandb with model logging',
            config = {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "k_discriminator": k_discriminator,
                "random_state": random_state,
                "latent_space_size": latent_space_size
            }
        )
    else:
        return None

def finalise_wandb(wandb_active, run, tg, td, gencol, discol):
    if wandb_active:
        art_gen = wandb.Artifact(
            name = "Simple_Generators",
            type = "model",
            description = "Our first very simple model",
            metadata = dict(run.config)
        )
        art_gen.add_file(local_path = tg)
        run.link_artifact(art_gen, f"s203768-dtu-org/wandb-registry-MLOps_Project_Models/{gencol}")

        art_dis = wandb.Artifact(
                name = "SimpleDiscirminator",
                type = "model",
                description = "Our first very simple model",
                metadata = dict(run.config)
        )
        art_dis.add_file(local_path = td)
        run.link_artifact(
            art_dis, f"s203768-dtu-org/wandb-registry-MLOps_Project_Models/{discol}"
        )
        run.finish()

def logging_loss(wandb_active, dictlog : dict[any, any]):
    if wandb_active:
        wandb.log(dictlog)

def train_core(learning_rate: float = 2e-5, batch_size: int = 64, epochs: int = 10, k_discriminator: int = 3, random_state: int = 42, latent_space_size: int = 1000, gencol:str = "Simple_Generators", discol:str = "Simple_Discirminators", wandb_active: bool = False) -> None:
    """Training step for the GAN.
    
    Each epoch is made of steps, which is some fraction of the total dataset.

    Keyword arguments:
    learning_rate -- How quickly the network learns.
    batch_size -- How many images to include per step?
    epochs -- How many times should the model pass over the data?
    k_discriminator -- How many extra rounds may the discriminator train per step relative to the generator?
    random_state -- What is the random state that you'd like to fix for the system?
    latent_space_size -- How big is the latent space for the Generator?
    wandb_active -- Is wandb logging to be used or not?
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

    run = setup_wandb(learning_rate, batch_size, epochs, k_discriminator, random_state, latent_space_size, wandb_active)

    # GAN Paper: https://arxiv.org/pdf/1406.2661 -- See the algorithm on page 4
    for epoch in range(epochs):
        for i, real_images in enumerate(main_dataloader):
            real_images = real_images.to(DEVICE)
            temp_batch_size = len(real_images)

            # Part 1 -- Give the discriminator a head start against the generator
            for j in range(k_discriminator):
                # Generate random latent space noise
                z = torch.randn(temp_batch_size, latent_space_size)
                z = z.type_as(real_images)

                # Turn the latent space into images
                gen_model.eval()
                dis_model.train()
                fake_images = gen_model(z)
                declare_fake = torch.zeros(temp_batch_size, 1).to(DEVICE)
                declare_real = torch.ones(temp_batch_size, 1).to(DEVICE)

                # Provide real images
                dis_opt.zero_grad()
                real_loss = dis_loss(dis_model(real_images), declare_real)
                
                # Provide fake images
                fake_loss = dis_loss(dis_model(fake_images), declare_fake)
                loss = (real_loss + fake_loss) / 2
                loss.backward()
                dis_opt.step()
                logging_loss(wandb_active, {"Discriminator_real_loss": loss.item()})
                logging_loss(wandb_active, {"Discriminator_fake_loss": loss.item()})
                logging_loss(wandb_active, {"Discriminator_loss": loss.item()})

                # Get idea of loss
                if i % 100 == 0 and j == k_discriminator - 1:
                    print(f"Epoch {epoch}, iter {i}, dis loss: {loss.item()}")
            
            # Part 2 -- Update the generator to try to trick the discriminator
            gen_model.train()
            dis_model.eval()

            # Generate random latent space noise
            z = torch.randn(temp_batch_size, latent_space_size)
            z = z.type_as(real_images)

            # Tell the discriminator that the data is real, optimise the generator for tricking it.
            declare_real = torch.ones(temp_batch_size, 1).to(DEVICE)

            gen_opt.zero_grad()
            loss = gen_loss(dis_model(gen_model(z)), declare_real)
            loss.backward()
            gen_opt.step()  
            logging_loss(wandb_active, {"Generator_loss": loss.item()})

            # Get idea of loss
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, gen loss: {loss.item()}")
                z_for_logging = torch.randn(1, latent_space_size)
                z_for_logging = z_for_logging.type_as(real_images)
                image_for_logging = gen_model(z_for_logging)
                # view image in 2d
                image_for_logging = image_for_logging.view(3, 32, 32).detach().cpu().numpy()
                logging_loss(wandb_active, {"Generated_image": [wandb.Image(image_for_logging)]})


    trained_path = "models"
    trained_generator_name = "simple_generator.pth"
    trained_discriminator_name = "simple_discriminator.pth"
    tg = trained_path + "/" + trained_generator_name
    td = trained_path + "/" + trained_discriminator_name
    torch.save(gen_model.state_dict(), tg)
    torch.save(dis_model.state_dict(), td)

    print(f"Saved generator to {trained_path} as {trained_generator_name}")
    print(f"Saved discriminator to {trained_path} as {trained_discriminator_name}")
    finalise_wandb(wandb_active, run, tg, td, gencol, discol)

if __name__ == "__main__":
    app()
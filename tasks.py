import os
from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "group83_mlops"
PYTHON_VERSION = "3.11"

# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )

@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)

@task
def train_hydra(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py train-hydra", echo=True, pty=not WINDOWS)

@task
def train_wandb(
    ctx: Context,
    gencol: str = "Simple_Generators",
    discol: str = "Simple_Discriminators",
    lr: float = 2e-5,
    batch_size: int = 64,
    epochs: int = 10,
    k_discriminator: int = 3,
    random_state: int = 42,
    latent_space_size: int = 1000
) -> None:
    """Train model.

    Args:
        gencol (str): The name of the generator model collection to use for training.
        discol (str): The name of the discriminator model collection to use for training.
        lr (float): Learning rate for the training process.
        batch_size (int): Number of samples per training batch.
        epochs (int): Total number of training epochs.
        k_discriminator (int): Number of discriminator steps per generator step.
        random_state (int): Seed for reproducibility.
        latent_space_size (int): Dimensionality of the latent space for generator inputs.
    """
    command = (
        f"python src/{PROJECT_NAME}/train.py train-wandb "
        f"--gencol {gencol} "
        f"--discol {discol} "
        f"--learning-rate {lr} "
        f"--batch-size {batch_size} "
        f"--epochs {epochs} "
        f"--k-discriminator {k_discriminator} "
        f"--random-state {random_state} "
        f"--latent-space-size {latent_space_size}"
    )
    ctx.run(command, echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, n_images: int = 1000, gen_col: str = "Simple_Generators", alias: str = "latest") -> None:
    """Evaluate using CNNdetect"""
    """
    n_images:       Number of images to generate and evaluate with CNNDetect
    gen-col:        Generator Collection to pull model from.
                    Possible options for collection:
                    - "Simple_Generators"
                    - "Generators"
    alias:          Model alias. Example v1 for version 1 or "latest" for the latest model in
                    this colection
    """
    ctx.run(f"python src/{PROJECT_NAME}/evaluate.py --n-images {n_images} --gen-col {gen_col} --alias {alias}", echo=True, pty=not WINDOWS)

    import torch
    use_cpu_flag = "--use_cpu" if not torch.cuda.is_available() else ""

    cnn_model = "CNNDetection/weights/blur_jpg_prob0.5.pth"

    if not os.path.exists(cnn_model):
        print(f"Model {cnn_model} not found. Downloading weights...")
        ctx.run("bash CNNDetection/weights/download_weights.sh", echo=True, pty=not WINDOWS)

    # Run CNN detection
    ctx.run(
        f"python CNNDetection/demo_dir.py -d CNNDetection/tmp "
        f"-m {cnn_model} {use_cpu_flag}",
        echo=True,
        pty=not WINDOWS,
    )

    # Cleanup
    import shutil
    folder = "CNNDetection/tmp"
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            shutil.rmtree(subfolder_path)



@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)

@task
def cnn_detect(ctx: Context) -> None:
    """Run CNN detection."""
    ctx.run("python CNNDetection/demo.py -f CNNDetection/examples/real.png -m CNNDetection/weights/blur_jpg_prob0.5.pth --use_cpu", echo=True, pty=not WINDOWS)

@task
def cnn_detect_dir(ctx: Context) -> None:
    """Run CNN detection."""
    ctx.run("python CNNDetection/demo_dir.py -d CNNDetection/examples/realfakedir -m CNNDetection/weights/blur_jpg_prob0.5.pth --use_cpu", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

@task
def docker_run(ctx: Context) -> None:
    """Run docker container."""
    try:
        ctx.run(
            'docker run -it --rm -v $(pwd)/data/processed:/data/processed -v $(pwd)/models:/models --env-file .env train:latest',
            echo=True,
            pty=not WINDOWS
        )
    except Exception:
        ctx.run(
            'docker run -it --rm -v $(pwd)/data/processed:/data/processed -v $(pwd)/models:/models train:latest',
            echo=True,
            pty=not WINDOWS
        )
# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)

import wandb
import torch
import typer
import os
import warnings
import random

from torchvision.transforms import ToPILImage
from group83_mlops.data import cifar100

to_pil = ToPILImage()

# Create output dirs for images
tmp = "CNNDetection/tmp"
os.makedirs(tmp, exist_ok=True)
output_dir_fake = "CNNDetection/tmp/1_fake"
os.makedirs(output_dir_fake, exist_ok=True)
output_dir_real = "CNNDetection/tmp/0_real"
os.makedirs(output_dir_real, exist_ok=True)

from group83_mlops.get_model_from_artifact import get_model_from_artifact
latent_space_size = 1000

def generate_images(gen_col:str = 'Simple_Generators', alias:str = 'latest', n_images:int = 1000):
    warnings.filterwarnings("ignore", category=FutureWarning)

    model = get_model_from_artifact(collection = gen_col, alias = alias)

    latent_space_size = 1000
    z = torch.randn(n_images, latent_space_size)
    for i in range(n_images):
        gen_img = model.forward(z[i,:])
        gen_img = gen_img.view(3, 32, 32)
        gen_img = 0.5 * (gen_img + 1)
        
        image = to_pil(gen_img)
        output_path = os.path.join(output_dir_fake, f"fake_{i}.png")
        image.save(output_path)

    # Sample real images as well
    real_images =torch.load("data/processed/test_images.pt")
    real_images = 0.5 * (real_images + 1)
    real_images = real_images.permute(0, 3, 1, 2)
    
    n_real_images = len(real_images)

    # If n_images is larger than the number of real test images, just take all of them
    # else take n_images
    if n_images > n_real_images:
        for i in range(n_real_images):
            image = to_pil(real_images[i,:,:,:])
            output_path = os.path.join(output_dir_real, f"real_{i}.png")
            image.save(output_path)
    else:
        random_idxs = random.sample(range(n_real_images), n_images)
        for i, idx in enumerate(random_idxs):
            # print(real_images.shape)
            image = to_pil(real_images[idx,:,:,:])
            output_path = os.path.join(output_dir_real, f"real_{i}.png")
            image.save(output_path)
            
if __name__ == '__main__':
    typer.run(generate_images)
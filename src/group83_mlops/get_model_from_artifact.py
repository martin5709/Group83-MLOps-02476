import wandb
import torch
from group83_mlops.model import Generator, Discriminator

REGISTRY = 'MLOps_Project_Models'

# Working collections are:
# Simple_Generators
# Simple_Discriminators
# Generators
# Discriminators

run = wandb.init(
    entity = 'group83-MLOps-02476',
    project = 'group83-MLOps-02476'
)  
def get_model_from_artifact(collection:str = 'Simple_Generators', alias:str = 'latest'):
    artifact_name = f"wandb-registry-{REGISTRY}/{collection}:{alias}"

    #fetched_artifact = run.use_artifact(artifact_or_name = artifact_name)  
    downloaded_model_path = run.use_model(name= artifact_name)
    model = Generator(latent_space_size = 1000)
    model.load_state_dict(torch.load(downloaded_model_path))

    return model

get_model_from_artifact()    


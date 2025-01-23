import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torchvision.datasets as datasets
import pandas as pd
import torch
from google.cloud import storage
import os
import matplotlib.pyplot as plt

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset,TargetDriftPreset

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

from torchvision.transforms import ToPILImage

to_pil = ToPILImage()

DATA_BUCKET = "1797480b-392d-46d1-be40-af7e3b95936b"
FILE_NAME = "train_images.pt"
FILE_PATH = "data/processed"
BACKUP_BUCKET = "e3d6e328-42fd-4297-b9e5-71375e160dc0"
BACKUP_NAME = "train_images_CIFAR10.pt"


def download_data():
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(DATA_BUCKET, prefix="data/processed/",versions=True)
    blobs = [blob for blob in blobs if blob.name.endswith('train_images.pt')]

    for blob in blobs:
        print(f"{blob.name},{blob.generation}")

    blobs.sort(key=lambda x: x.generation, reverse=True)

    # Download the latest version
    latest_blob = blobs[0]
    latest_blob.download_to_filename("new.pt")
    print(f"Downloaded latest version: {latest_blob.name}, generation: {latest_blob.generation}")

    # Download the previous version if it exists
    if len(blobs) > 1:
        previous_blob = blobs[1]
        previous_blob.download_to_filename("old.pt")
        print(f"Downloaded previous version: {previous_blob.name}, generation: {previous_blob.generation}")
    else:
        backup_blob = storage_client.bucket(BACKUP_BUCKET).blob(BACKUP_NAME)
        backup_blob.download_to_filename("old.pt")
        print(f"Your versioning is fucked, dowloaded CIFAR10 dataset from backup instead")

    print(f"Downloaded old and new data")
    return None

download_data()
old_data =torch.load("old.pt", weights_only = False).float()
new_data =torch.load("new.pt", weights_only = False).float()

# new_mean = torch.tensor([0.5071, 0.4865, 0.4409])
# new_mean = new_mean[None,None,None,:]
# new_std = torch.tensor([0.2673, 0.2564, 0.2762])
# new_std = new_std[None,None,None,:]

# old_data =  old_data * new_std + new_mean
# new_data = new_data * new_std + new_mean

old_data = old_data.permute( (0, 3, 1, 2))
new_data = new_data.permute( (0, 3, 1, 2))

print("Data downloaded successfully")

df_old = pd.DataFrame(columns=[f"feature_{i}" for i in range(512)])
df_new = pd.DataFrame(columns=[f"feature_{i}" for i in range(512)])

n = 100
for i in range(n):
    old = old_data[i,:,:,:]
    new = new_data[i,:,:,:]
    inputs = processor(text=None, images=old, return_tensors="pt", padding=True)

    img_features = model.get_image_features(inputs["pixel_values"])
    df_old.loc[i] = img_features[0].detach().numpy()

    inputs = processor(text=None, images=new, return_tensors="pt", padding=True)

    img_features = model.get_image_features(inputs["pixel_values"])
    df_new.loc[i] = img_features[0].detach().numpy()


report = Report(metrics=[DataDriftPreset(), DataQualityPreset,TargetDriftPreset])
report.run(reference_data=df_old, current_data=df_new)
report.save_html('CLIP_report.html')

# Cleanup
print("Hej")
os.remove("old.pt")
os.remove("new.pt")

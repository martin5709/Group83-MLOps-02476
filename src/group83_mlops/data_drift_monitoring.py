import json
import os
from pathlib import Path

import anyio
import nltk
import pandas as pd
from evidently.metric_preset import TargetDriftPreset, DataDriftPreset, DataQualityPreset,TargetDriftPreset
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage

import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
from torchvision.transforms import ToPILImage
import random

nltk.download("words")
nltk.download("wordnet")
nltk.download("omw-1.4")


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



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

    blobs.sort(key=lambda x: x.generation, reverse=True)

    # Download the latest version
    latest_blob = blobs[0]
    latest_blob.download_to_filename("new.pt")
    print(f"Downloaded latest version of your data")

    # Download the previous version if it exists
    if len(blobs) > 1:
        previous_blob = blobs[1]
        previous_blob.download_to_filename("old.pt")
        print("WOW! Versioning acually fucking works. I have downloaded two versions of your data :D")
    else:
        backup_blob = storage_client.bucket(BACKUP_BUCKET).blob(BACKUP_NAME)
        backup_blob.download_to_filename("old.pt")
        print(f"Versioning is fucked again, dowloaded CIFAR10 dataset from backup instead")
    return None

def data_2_csvs(n_images_check):
    old_data =torch.load("old.pt", weights_only = False).float()
    new_data =torch.load("new.pt", weights_only = False).float()

    # For transforming data back to interval [0,1], for PIL
    new_mean = torch.tensor([0.5071, 0.4865, 0.4409])
    new_mean = new_mean[None,None,None,:]
    new_std = torch.tensor([0.2673, 0.2564, 0.2762])
    new_std = new_std[None,None,None,:]

    old_data =  old_data * new_std + new_mean
    new_data = new_data * new_std + new_mean

    old_data = old_data.permute( (0, 3, 1, 2))
    new_data = new_data.permute( (0, 3, 1, 2))


    df_old = pd.DataFrame(columns=[f"feature_{i}" for i in range(512)])
    df_new = pd.DataFrame(columns=[f"feature_{i}" for i in range(512)])

    num_images = old_data.shape[0]
    indices = random.sample(range(num_images), n_images_check)
    if num_images < n_images_check:
        indices = range(num_images)

    
    for i, idx in enumerate(indices):
        old = old_data[idx,:,:,:]
        inputs = processor(text=None, images=old, return_tensors="pt", padding=True)

        img_features = model.get_image_features(inputs["pixel_values"])
        df_old.loc[i] = img_features[0].detach().numpy()

    num_images = new_data.shape[0]
    indices = random.sample(range(num_images), n_images_check)
    if num_images < n_images_check:
        indices = range(num_images)

    for i, idx in enumerate(indices):
        new = new_data[idx,:,:,:]
        inputs = processor(text=None, images=new, return_tensors="pt", padding=True)

        img_features = model.get_image_features(inputs["pixel_values"])
        df_new.loc[i] = img_features[0].detach().numpy()
    
    return df_new, df_old

def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global df_old, df_new
    download_data()
    yield

    del df_old, df_new


app = FastAPI(lifespan=lifespan)


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 100):
    """Generate and return the report."""
    print(n)
    df_old, df_new = data_2_csvs(n_images_check = n)
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(),TargetDriftPreset()])
    report.run(reference_data=df_old, current_data=df_new)
    report.save_html('report.html')

    async with await anyio.open_file("report.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)


import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torchvision.datasets as datasets
import pandas as pd
import torch
from google.cloud import storage
import os

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset,TargetDriftPreset

DATA_BUCKET = "1797480b-392d-46d1-be40-af7e3b95936b"
FILE_NAME = "train_images.pt"
FILE_PATH = "data/processed"


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
        print("No previous version found.")

    print(f"Downloaded old and new data")
    return None

download_data()
old_data =torch.load("old.pt")
new_data =torch.load("new.pt")
print(old_data.shape)
print("Data downloaded successfully")

# Cleanup
os.remove("old.pt")
os.remove("new.pt")

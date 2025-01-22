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
  """Downloads the previous version of an object from Google Cloud Storage.

  Args:
    bucket_name: The name of the bucket.
    object_name: The name of the object.

  Returns:
    The path to the downloaded file or None if no previous version exists.
  """

  storage_client = storage.Client()
  blobs = storage_client.list_blobs(DATA_BUCKET, prefix="data/processed/",versions=True)
  blobs = [blob for blob in blobs if blob.name.endswith('.pt')]

  for blob in blobs:
    print(f"{blob.name},{blob.generation}")


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

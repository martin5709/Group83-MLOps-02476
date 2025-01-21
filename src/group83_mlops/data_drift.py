import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torchvision.datasets as datasets
import pandas as pd
import torch
from google.cloud import storage

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset,TargetDriftPreset

DATA_BUCKET = "1797480b-392d-46d1-be40-af7e3b95936b"

def load_data_from_cloud(file):
    storage_client = storage.Client()
    bucket = storage_client.bucket(DATA_BUCKET)
    blob = bucket.blob(file)
    blob.download_to_filename(file)
    print(f"Model {file} downloaded from {DATA_BUCKET}.")
    return

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

load_data_from_cloud("data/processed/train_images.pt")
load_data_from_cloud("data/processed/new_images.pt")

training_data = torch.load("data/processed/train_images.pt")
svhn = datasets.SVHN(root='data', download=True)

df_cifar = pd.DataFrame(columns=[f"feature_{i}" for i in range(512)])
df_svhn = pd.DataFrame(columns=[f"feature_{i}" for i in range(512)])

n = 100
for i in range(n):
    cifar_data = cifar.data[i]
    inputs = processor(text=None, images=cifar_data, return_tensors="pt", padding=True)

    img_features = model.get_image_features(inputs["pixel_values"])
    df_cifar.loc[i] = img_features[0].detach().numpy()

    svhn_data = svhn.data[i]
    inputs = processor(text=None, images=svhn_data, return_tensors="pt", padding=True)

    img_features = model.get_image_features(inputs["pixel_values"])
    df_svhn.loc[i] = img_features[0].detach().numpy()


report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_cifar, current_data=df_svhn)
report.save_html('reports/CLIP_report.html')
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google.cloud import storage
import datetime
from group83_mlops.model import Generator
from io import BytesIO

from torchvision.transforms import ToPILImage

to_pil = ToPILImage()

# Define model and device configuration
MODEL_NAME = "simple-generator"
MODEL_FILE = "simple_generator.pth"
BUCKET_IMG = "0e6b97bf-6590-4dc4-b464-08f9e1cb2ae7"
BUCKET_MODEL = "mlops-model-repo"
LATENT_SPACE_SIZE = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Input(BaseModel):
    """Define input data structure for the endpoint."""

    request: str

def load_model_from_cloud():
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_MODEL)
    blob = bucket.blob(MODEL_FILE)
    blob.download_to_filename(MODEL_FILE)
    print(f"Model {MODEL_FILE} downloaded from {BUCKET_MODEL}.")
    return

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and tokenizer when the app starts and clean up when the app stops."""
    global model
    model = Generator(LATENT_SPACE_SIZE)
    load_model_from_cloud()
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model = model.to(device)
    model.eval()
    print("Generator loaded successfully.")

    yield

    del model

def upload_to_cloud(image):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_IMG)
    time = datetime.datetime.now(tz=datetime.UTC)

    # Save the PIL image to a byte stream
    byte_stream = BytesIO()
    image.save(byte_stream, format='PNG')
    byte_stream.seek(0)

    # Prepare the blob and upload the image
    blob = bucket.blob(f"/images/image_{time}.png")
    blob.upload_from_file(byte_stream, content_type='image/png')
    print("Prediction saved to GCP bucket.")
    return

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


# Prediction endpoint
@app.post("/generate")
async def predict_sentiment(review_input: Input):
    try:
        z = torch.randn(1, LATENT_SPACE_SIZE).to(device)
        with torch.no_grad():
            gen_img = model.forward(z)
            gen_img = gen_img.view(3, 32, 32)
            gen_img = 0.5 * (gen_img + 1)

        image = to_pil(gen_img)

        upload_to_cloud(image)

        byte_stream = BytesIO()
        image.save(byte_stream, format='PNG')
        byte_stream.seek(0)

        return StreamingResponse(byte_stream, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

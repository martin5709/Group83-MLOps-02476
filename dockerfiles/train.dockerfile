# Base image
FROM nvcr.io/nvidia/pytorch:24.12-py3 AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc wget && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY CNNDetection CNNDetection/
COPY configs configs/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose
RUN wget https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=0 -O CNNDetection/weights/blur_jpg_prob0.5.pth

ENTRYPOINT ["python", "src/group83_mlops/train.py", "train-hydra", "--vertex"]

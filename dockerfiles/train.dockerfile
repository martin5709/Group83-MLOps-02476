# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get install -y --no-install-recommends wget

COPY src src/
COPY CNNDetection CNNDetection/
COPY configs configs/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose
RUN wget https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=0 -O CNNDetection/weights/blur_jpg_prob0.5.pth

ENTRYPOINT ["python", "src/group83_mlops/train.py", "train-hydra", "--vertex"]

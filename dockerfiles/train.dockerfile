# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY CNNDetection CNNDetection/
COPY configs configs/
COPY .dvc .dvc/
COPY .dvcignore .dvcignore
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_dev.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose
RUN dvc pull --no-run-cache

ENTRYPOINT ["python", "src/group83_mlops/train.py train-hydra"]

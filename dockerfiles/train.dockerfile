# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY ./ ./

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_dev.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose
RUN dvc config core.no_scm true
RUN dvc pull --no-run-cache

ENTRYPOINT ["python", "src/group83_mlops/train.py train-hydra"]

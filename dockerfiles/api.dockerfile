# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

WORKDIR /app
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install -r requirements_cloud.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose
RUN pip install uvicorn

EXPOSE $PORT

# ENTRYPOINT ["uvicorn", "src/group83_mlops/api:app", "--host", "0.0.0.0", "--port", {$PORT}]
CMD exec uvicorn "src.group83_mlops.api:app" --port $PORT --host 0.0.0.0 --workers 1


# europe-west1-docker.pkg.dev/mlops-project-group83/docker-images

# docker build -t api_image -f dockerfiles/api.dockerfile .

# docker tag api_image europe-west1-docker.pkg.dev/mlops-project-group83/docker-images/api_image:latest

# docker push europe-west1-docker.pkg.dev/mlops-project-group83/docker-images/api_image:latest



gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name=training-run-gpu \
    --config=configs/vertex/gpu_config.yaml \
    --command 'python' \
    --args="src/group83_mlops/train.py","train-hydra","--experiment","exp2","--vertex"

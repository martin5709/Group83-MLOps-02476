gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name=training-run \
    --config=configs/vertex/cpu_config.yaml \
    --command 'python' \
    --args="src/group83_mlops/train.py","train-hydra","--experiment","exp1","--vertex"

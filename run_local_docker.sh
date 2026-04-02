#!/bin/bash
set -euo pipefail

IMAGE_NAME="ds-core-cn-omra-trainer:local"

echo "=== Building Docker image ==="
docker build -t "${IMAGE_NAME}" -f Dockerfile .

echo "=== Running training container ==="
docker run --rm -it \
  --gpus all \
  --shm-size=64g \
  -v /mnt/ephemeral/blip_dataset:/data \
  -v "$(pwd)/universal_pix_t2i_workdirs:/app/workdirs" \
  -v "${HOME}/.aws:/root/.aws:ro" \
  -e WANDB_MODE=offline \
  -e WANDB_DIR=/tmp/wandb \
  -e AWS_DEFAULT_REGION=us-east-1 \
  "${IMAGE_NAME}" \
  python main.py fit \
    -c ./configs_t2i/pretraining_res256_selfflow.yaml \
    --trainer.default_root_dir=/app/workdirs \
    --trainer.devices=8 \
    --trainer.max_steps=100 \
    --data.train_batch_size=4 \
    --data.train_num_workers=2 \
    --data.train_dataset.init_args.urls=[/data/BLIP3o-60k]

#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "$0")/.." && pwd)
gpu_args=(--gpus all)
for opt in "$@"; do
    if [[ "$opt" == "--validate-data" || "$opt" == "--check-implementation" ]]; then gpu_args=(); fi
done
docker run --rm --network none "${gpu_args[@]}" --shm-size=8g \
    --user "$(id -u):$(id -g)" -v "$project_dir:/project" -w /project \
    -e PYTHONPATH=/project/.runtime/paper_training:/project/.runtime/official_compat:/project/.runtime/official_entry:/project/.runtime/official_entry_extra:/project/.runtime/official_iqa:/project/.runtime/python \
    -e WANDB_MODE=disabled -e XDG_CACHE_HOME=/tmp/cache -e MPLCONFIGDIR=/tmp/matplotlib \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 \
    -e CUDA_CACHE_PATH=/project/.runtime/retrain_cuda_cache -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --entrypoint python deeppro-official-eval:20261009 paper_reproduction/train.py "$@"

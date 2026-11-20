#!/usr/bin/env bash
set -euo pipefail
case_dir=$(cd -- "$(dirname -- "$0")"/../.. && pwd)
docker run --rm --gpus all --network none --hostname localhost --shm-size=8g --user "$(id -u):$(id -g)" \
    -v "$case_dir:/project" -w /project --entrypoint python \
    -e OMP_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -e CUDA_CACHE_PATH=/project/.runtime/retrain_cuda_cache -e CUDA_CACHE_MAXSIZE=1073741824 -e MPLCONFIGDIR=/tmp/matplotlib -e XDG_CACHE_HOME=/tmp/cache -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e PYTHONPATH=/project/.runtime/official_compat:/project/.runtime/official_entry:/project/.runtime/official_entry_extra:/project/.runtime/official_iqa:/project/.runtime/python \
    deeppro-official-eval:20261009 train_local_baseline.py "$@"

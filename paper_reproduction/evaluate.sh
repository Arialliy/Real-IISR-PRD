#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "$0")/.." && pwd)
gpu_args=()
if [[ ${1:-} == infer ]]; then gpu_args=(--gpus "device=${FLIR_EVAL_GPU:-0}"); fi
docker run --rm --network none "${gpu_args[@]}" --shm-size=2g \
    --user "$(id -u):$(id -g)" -v "$project_dir:/project" -w /project \
    -e PYTHONPATH=/project/.runtime/official_compat:/project/.runtime/official_entry:/project/.runtime/official_entry_extra:/project/.runtime/official_iqa:/project/.runtime/python \
    -e XDG_CACHE_HOME=/tmp/cache -e MPLCONFIGDIR=/tmp/mplcache \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 \
    --entrypoint python deeppro-official-eval:20261009 paper_reproduction/evaluate.py "$@"

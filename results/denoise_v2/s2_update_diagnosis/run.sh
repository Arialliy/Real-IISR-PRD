#!/usr/bin/env bash
set -euo pipefail
run_dir=$(cd -- "$(dirname -- "$0")" && pwd)
project=$(cd -- "$run_dir/../../.." && pwd)
task=(diagnose_s2_updates.py "$1")
if [[ $1 == pure ]]; then
    task=(results/denoise_v2/s2_update_diagnosis/pure_first_gradient.py)
fi
docker run --rm --network none --gpus "device=$2" --shm-size=2g \
    --user "$(id -u):$(id -g)" --entrypoint python \
    -e PYTHONPATH=/project/.runtime/official_compat:/project/.runtime/official_entry:/project/.runtime/official_entry_extra:/project/.runtime/official_iqa:/project/.runtime/python \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 \
    -e CUDA_CACHE_PATH=/project/.runtime/retrain_cuda_cache -e CUDA_CACHE_MAXSIZE=1073741824 -e MPLCONFIGDIR=/tmp/mplcache -e XDG_CACHE_HOME=/tmp/cache -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$project:/project" -w /project deeppro-official-eval:20261009 "${task[@]}"

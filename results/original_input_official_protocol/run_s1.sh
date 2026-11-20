#!/usr/bin/env bash
set -euo pipefail
run_dir=$(cd -- "$(dirname -- "$0")" && pwd)
project=$(cd -- "$run_dir/../.." && pwd)
docker run --rm --network none --gpus "device=${1:-1}" --shm-size=2g \
    --user "$(id -u):$(id -g)" --entrypoint python \
    -v "$project:/project" -w /project/results/original_input_official_protocol \
    -e PYTHONPATH=/project/.runtime/official_compat:/project/.runtime/official_entry:/project/.runtime/official_entry_extra:/project/.runtime/official_iqa:/project/.runtime/python \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 \
    -e XDG_CACHE_HOME=/tmp/cache -e MPLCONFIGDIR=/tmp/mplcache \
    deeppro-official-eval:20261009 capture_s1.py

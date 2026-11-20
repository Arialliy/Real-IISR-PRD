#!/usr/bin/env bash
set -euo pipefail
audit_dir=$(cd -- "$(dirname -- "$0")" && pwd)
project=$(cd -- "$audit_dir/../../.." && pwd)
job=capture.py
device=(--gpus "device=${1:-1}")
if [[ ${1:-} == measure ]]; then
    job=summarize.py
    device=()
fi
docker run --rm --network none "${device[@]}" --shm-size=2g \
    --user "$(id -u):$(id -g)" --entrypoint python \
    -v "$project:/project" -w /project/results/official_sirst_train/float_audit \
    -e PYTHONPATH=/project/.runtime/official_compat:/project/.runtime/official_entry:/project/.runtime/official_entry_extra:/project/.runtime/official_iqa:/project/.runtime/python \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=2 \
    -e XDG_CACHE_HOME=/tmp/cache -e MPLCONFIGDIR=/tmp/mplcache \
    deeppro-official-eval:20261009 "$job"

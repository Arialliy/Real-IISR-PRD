#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
op=${1:-report}
gpu=()
if [[ "$op" == run ]]; then gpu=(--gpus "device=${SIRST_SPOT_GPU:-1}"); fi
docker run --rm --network none "${gpu[@]}" --user "$(id -u):$(id -g)" -v "$project:/project" -w /project \
  --entrypoint python -e OMP_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/project/.runtime/python deeppro-official-eval:20261009 diagnose_spot_transfer.py "$op"

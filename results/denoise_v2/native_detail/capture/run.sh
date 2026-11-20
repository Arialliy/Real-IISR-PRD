#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "$0")/../../../.." && pwd)
exec docker run --rm --network none --gpus "device=${SIRST_DETAIL_GPU:-0}" --user "$(id -u):$(id -g)" \
  -v "$project:/project" -w /project --entrypoint python -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/project/.runtime/python deeppro-official-eval:20261009 capture_native_detail_pairs.py "$@"

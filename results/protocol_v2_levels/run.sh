#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "$0")/../.." && pwd)
device=()
case "${1:-}" in published|s1|s2) device=(--gpus "device=${SIRST_EVAL_GPU:-0}");; esac
exec docker run --rm --network none "${device[@]}" --user "$(id -u):$(id -g)" \
  -v "$project:/project" -w /project --entrypoint python -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e MPLCONFIGDIR=/tmp/matplotlib -e PYTHONPATH=/project/.runtime/python \
  deeppro-official-eval:20261009 evaluate_noise_levels.py "$@"

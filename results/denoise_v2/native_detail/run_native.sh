#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
device=()
case "${1:-}" in stripe_only|pixel_only|combined|image|scan) device=(--gpus "device=${SIRST_EVAL_GPU:-0}");; esac
script=evaluate_native_detail.py
if [[ ${1:-} == image ]]; then shift; script=native_detail.py; fi
exec docker run --rm --network none "${device[@]}" --user "$(id -u):$(id -g)" \
  -v "$project:/project" -w /project --entrypoint python -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e MPLCONFIGDIR=/tmp/mpl -e PYTHONPATH=/project/.runtime/python \
  deeppro-official-eval:20261009 "$script" "$@"

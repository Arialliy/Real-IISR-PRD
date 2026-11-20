#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "$0")/../.." && pwd)
device=()
if [[ ${1:-} != --compare ]]; then device=(--gpus "device=${SIRST_EVAL_GPU:-0}"); fi
docker run --rm --network none "${device[@]}" --user "$(id -u):$(id -g)" \
  -v "$project:/project" -w /project --entrypoint python -e OMP_NUM_THREADS=2 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/project/.runtime/python \
  deeppro-official-eval:20261009 evaluate_sirst_endpoints.py "$@"

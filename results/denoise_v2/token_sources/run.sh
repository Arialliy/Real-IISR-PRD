#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
gpu=(--gpus device=2)
case "${1:-}" in
  capture|score) entry=evaluate_e_only.py ;;
  summarize)
    entry=summarize_e_only.py
    gpu=()
    shift
    ;;
  width)
    entry=measure_e_only_native_width.py
    shift
    if [[ ${1:-} == summarize ]]; then gpu=(); fi
    ;;
  *) echo 'usage: bash run.sh {capture|score|summarize|width {s1|s2|summarize}}' >&2; exit 2 ;;
esac
docker run --rm --network none "${gpu[@]}" --user "$(id -u):$(id -g)" \
  -v "$project:/project" -w /project --entrypoint python \
  -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/project/.runtime/python \
  deeppro-official-eval:20261009 "$entry" "$@"

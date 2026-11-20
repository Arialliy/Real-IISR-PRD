#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "$0")/../../../.." && pwd)
exec docker run --rm --network none --user "$(id -u):$(id -g)" -v "$project:/project" -w /project \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONPATH=/project/.runtime/python \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python \
  deeppro-official-eval:20261009 evaluate_native_detail_spots.py

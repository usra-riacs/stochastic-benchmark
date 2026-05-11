#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_job.sh" \
  --graph-type heavy_hex \
  --num-nodes 144 \
  --p-values 5 \
  --train-count 10 \
  --test-count 5 \
  --start-train-index 100 \
  --fa-method-name FA_PP_opt \
  --pt-method-name "" \
  --fa-n-values 10,20,50,75,100 \
  --fa-m-values 10,50,100,200 \
  --q-values 100,500,1000,5000 \
  --t-grid-points 500 \
  --t-grid-scale log \
  --mps-chi 20 \
  --max-parallel-threads 4 \
  --output-root "${SCRIPT_DIR}/results/pss_window_sticker/heavy_hex_144_small" \
  "$@"

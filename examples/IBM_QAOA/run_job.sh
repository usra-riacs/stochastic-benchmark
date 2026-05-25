#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

load_conda() {
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        return
    fi

    local candidates=(
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
    )

    local conda_sh
    for conda_sh in "${candidates[@]}"; do
        if [[ -f "$conda_sh" ]]; then
            # shellcheck disable=SC1090
            source "$conda_sh"
            return
        fi
    done

    echo "Could not locate conda. Update run_job.sh with your conda.sh path." >&2
    exit 1
}

load_conda

if [[ -n "${VIRTUAL_ENV:-}" ]] && type deactivate >/dev/null 2>&1; then
    echo "Deactivating inherited virtualenv: ${VIRTUAL_ENV}"
    deactivate
fi

conda activate QAOA

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Could not find Python inside conda env QAOA at: $PYTHON_BIN" >&2
    exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$NUMBA_CACHE_DIR"

cd "$REPO_ROOT"

echo "Running IBM QAOA PSS preparation from: $REPO_ROOT"
echo "Using conda environment: QAOA"
echo "Using Python interpreter: $PYTHON_BIN"

DEFAULT_REUSE_ROOT="${SCRIPT_DIR}/results/pss_window_sticker/heavy_hex_144_small"
METHOD_NAME="FA_PP_opt"
METHOD_SLUG="${METHOD_NAME/_MPSAer/}"
METHOD_SLUG="${METHOD_SLUG/_MPS/}"
METHOD_SLUG="${METHOD_SLUG/_PP/}"
METHOD_SLUG="${METHOD_SLUG/_SV/}"
P_VALUES="5"

"$PYTHON_BIN" examples/IBM_QAOA/run_prepare_pss_campaign.py \
  --graph-type heavy_hex \
  --num-nodes 144 \
  --p-values "$P_VALUES" \
  --train-count 20 \
  --test-count 10 \
  --start-train-index 100 \
  --fa-method-name "$METHOD_NAME" \
  --pt-method-name "" \
  --fa-n-values 10,20,40,60,80,100,150 \
  --fa-m-values 10,25,50,100,200,500,1000 \
  --q-values 100,250,500,1000,2500,5000,10000 \
  --t-grid-points 1000 \
  --t-grid-scale log \
  --mps-chi 20 \
  --max-parallel-threads 4 \
  --output-root "${SCRIPT_DIR}/results/pss_window_sticker/heavy_hex_144_${METHOD_SLUG}_p${P_VALUES}_expanded" \
  --reuse-output-root "${SCRIPT_DIR}/results/pss_window_sticker/heavy_hex_144" \
  --reuse-output-root "$DEFAULT_REUSE_ROOT" \
  "$@"

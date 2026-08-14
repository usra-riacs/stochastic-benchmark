#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

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
METHOD_NAME="${METHOD_NAME:-FA_PP_opt}"
METHOD_SLUG="${METHOD_NAME/_MPSAer/}"
METHOD_SLUG="${METHOD_SLUG/_MPS/}"
METHOD_SLUG="${METHOD_SLUG/_PP/}"
METHOD_SLUG="${METHOD_SLUG/_SV/}"
METHOD_SLUG="${METHOD_SLUG/_AAA/}"

# Methods with no training (only Q-sweep): route as pt_method_name so they use
# run_pt_pss_exact_points (Q-only, evaluator stripped for FixedAngleConjecture).
# Methods with training (N,M grid): route as fa_method_name.
_ZERO_TRAINING_METHODS="PT_PP_AAA FA_PP_no_opt linear_ramp_no_opt"
if echo "$_ZERO_TRAINING_METHODS" | grep -qw "$METHOD_NAME"; then
    _FA_METHOD_ARG=""
    _PT_METHOD_ARG="$METHOD_NAME"
else
    _FA_METHOD_ARG="$METHOD_NAME"
    _PT_METHOD_ARG=""
fi
P_VALUES="${P_VALUES:-5}"
TRAIN_COUNT="${TRAIN_COUNT:-20}"
TEST_COUNT="${TEST_COUNT:-10}"
START_TRAIN_INDEX="${START_TRAIN_INDEX:-100}"
FA_N_VALUES="${FA_N_VALUES:-10,20,40,60,80,100,150}"
FA_M_VALUES="${FA_M_VALUES:-10,25,50,100,200,500,1000}"
Q_VALUES_WAS_SET="${Q_VALUES+x}"
DEFAULT_Q_VALUES="100,250,500,1000,2500,5000,10000"
PT_Q_GRID_POINTS="${PT_Q_GRID_POINTS:-70}"
T_GRID_POINTS="${T_GRID_POINTS:-1000}"
T_GRID_SCALE="${T_GRID_SCALE:-log}"
MPS_CHI="${MPS_CHI:-20}"
MAX_PARALLEL_THREADS="${MAX_PARALLEL_THREADS:-4}"
PT_TRANSFER_COST="${PT_TRANSFER_COST:-0}"
QAOA_PARAMETER_SETTING_ROOT="${QAOA_PARAMETER_SETTING_ROOT:-${WORKSPACE_ROOT}/QAOA-Parameter-Setting}"
QAOA_TRAINING_PIPELINE_ROOT="${QAOA_TRAINING_PIPELINE_ROOT:-${WORKSPACE_ROOT}/qaoa_training_pipeline}"

if [[ -z "$Q_VALUES_WAS_SET" && "$METHOD_NAME" == PT_* ]]; then
    Q_VALUES="$("$PYTHON_BIN" -c 'import numpy as np, os
points = int(os.environ.get("PT_Q_GRID_POINTS", "70"))
values = np.unique(np.rint(np.geomspace(100, 10000, points)).astype(int))
print(",".join(str(int(value)) for value in values))')"
else
    Q_VALUES="${Q_VALUES:-$DEFAULT_Q_VALUES}"
fi

echo "Using method: $METHOD_NAME"
echo "Using QAOA depth(s): $P_VALUES"
echo "Using Q grid: $Q_VALUES"
echo "Using QAOA-Parameter-Setting root: $QAOA_PARAMETER_SETTING_ROOT"
echo "Using qaoa_training_pipeline root: $QAOA_TRAINING_PIPELINE_ROOT"

"$PYTHON_BIN" examples/IBM_QAOA/run_prepare_pss_campaign.py \
  --graph-type heavy_hex \
  --num-nodes 144 \
  --main-repo "$QAOA_PARAMETER_SETTING_ROOT" \
  --pipeline-repo "$QAOA_TRAINING_PIPELINE_ROOT" \
  --p-values "$P_VALUES" \
  --train-count "$TRAIN_COUNT" \
  --test-count "$TEST_COUNT" \
  --start-train-index "$START_TRAIN_INDEX" \
  --fa-method-name "$_FA_METHOD_ARG" \
  --pt-method-name "$_PT_METHOD_ARG" \
  --fa-n-values "$FA_N_VALUES" \
  --fa-m-values "$FA_M_VALUES" \
  --q-values "$Q_VALUES" \
  --t-grid-points "$T_GRID_POINTS" \
  --t-grid-scale "$T_GRID_SCALE" \
  --mps-chi "$MPS_CHI" \
  --max-parallel-threads "$MAX_PARALLEL_THREADS" \
  --pt-transfer-cost "$PT_TRANSFER_COST" \
  --output-root "${SCRIPT_DIR}/results/pss_window_sticker/heavy_hex_144_${METHOD_SLUG}_p${P_VALUES}_expanded" \
  --reuse-output-root "${SCRIPT_DIR}/results/pss_window_sticker/heavy_hex_144" \
  --reuse-output-root "$DEFAULT_REUSE_ROOT" \
  "$@"

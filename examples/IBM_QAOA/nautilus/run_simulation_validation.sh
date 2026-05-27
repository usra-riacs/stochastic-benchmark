#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPOS_DIR="${WORKDIR}/repos"
DATA_DIR="${WORKDIR}/data"
RESULTS_DIR="${WORKDIR}/results"
SKIP_REPO_UPDATE="${IBM_QAOA_SKIP_REPO_UPDATE:-0}"

STOCHASTIC_BENCHMARK_REPO="${STOCHASTIC_BENCHMARK_REPO:-https://github.com/usra-riacs/stochastic-benchmark.git}"
STOCHASTIC_BENCHMARK_BRANCH="${STOCHASTIC_BENCHMARK_BRANCH:-QAOA_Parameter_Setting_IBM}"
QPS_REPO_URL="${QPS_REPO_URL:-https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting.git}"
QPS_BRANCH="${QPS_BRANCH:-main}"
QAOA_PIPELINE_REPO_URL="${QAOA_PIPELINE_REPO_URL:-https://github.com/qiskit-community/qaoa_training_pipeline.git}"
QAOA_PIPELINE_BRANCH="${QAOA_PIPELINE_BRANCH:-main}"

mkdir -p "${REPOS_DIR}" "${DATA_DIR}" "${RESULTS_DIR}" /tmp/matplotlib /tmp/numba-cache

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    git config --global credential.helper "store --file=/tmp/git-credentials"
    printf 'https://x-access-token:%s@github.com\n' "${GITHUB_TOKEN}" > /tmp/git-credentials
    chmod 0600 /tmp/git-credentials
fi

clone_or_update() {
    local url="$1"
    local branch="$2"
    local dest="$3"

    if [[ "${SKIP_REPO_UPDATE}" == "1" && -d "${dest}/.git" ]]; then
        return
    fi

    if [[ -d "${dest}/.git" ]]; then
        git -C "${dest}" fetch origin "${branch}"
        git -C "${dest}" checkout "${branch}"
        git -C "${dest}" pull --ff-only origin "${branch}"
    else
        git clone --branch "${branch}" --depth 1 "${url}" "${dest}"
    fi
}

clone_or_update_sparse() {
    local url="$1"
    local branch="$2"
    local dest="$3"
    shift 3
    local paths=("$@")

    if [[ "${SKIP_REPO_UPDATE}" == "1" && -d "${dest}/.git" ]]; then
        return
    fi

    if [[ -d "${dest}/.git" ]]; then
        git -C "${dest}" fetch origin "${branch}"
        git -C "${dest}" checkout "${branch}"
        git -C "${dest}" pull --ff-only origin "${branch}"
    else
        git clone --branch "${branch}" --depth 1 --filter=blob:none --sparse "${url}" "${dest}"
    fi
    git -C "${dest}" sparse-checkout set --no-cone "${paths[@]}"
}

clone_or_update "${STOCHASTIC_BENCHMARK_REPO}" "${STOCHASTIC_BENCHMARK_BRANCH}" "${REPOS_DIR}/stochastic-benchmark"
clone_or_update_sparse "${QPS_REPO_URL}" "${QPS_BRANCH}" "${REPOS_DIR}/QAOA-Parameter-Setting" \
  qaoa_parameter_setting \
  methods \
  instances/heavy_hex \
  data/evaluation_times \
  data/minmax_cuts/heavy_hex \
  setup.py \
  requirements.txt \
  VERSION.txt \
  README.md
clone_or_update "${QAOA_PIPELINE_REPO_URL}" "${QAOA_PIPELINE_BRANCH}" "${REPOS_DIR}/qaoa_training_pipeline"

SB_REPO="${REPOS_DIR}/stochastic-benchmark"
QPS_REPO="${REPOS_DIR}/QAOA-Parameter-Setting"
PIPELINE_REPO="${REPOS_DIR}/qaoa_training_pipeline"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${SB_REPO}/requirements.txt" -r "${SB_REPO}/requirements-examples.txt"
python -m pip install -e "${PIPELINE_REPO}" qiskit-aer
python -m pip install -e "${QPS_REPO}"
python -m pip install -e "${SB_REPO}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-cache}"
export QAOA_PARAMETER_SETTING_ROOT="${QPS_REPO}"
export QAOA_TRAINING_PIPELINE_ROOT="${PIPELINE_REPO}"
export IBM_QAOA_INSTANCE_CACHE_ROOT="${DATA_DIR}/generated_instances"
export PYTHONPATH="${SB_REPO}/src:${PYTHONPATH:-}"

cd "${SB_REPO}"

python examples/IBM_QAOA/run_prepare_pss_campaign.py \
  --main-repo "${QPS_REPO}" \
  --pipeline-repo "${PIPELINE_REPO}" \
  --instance-cache-root "${IBM_QAOA_INSTANCE_CACHE_ROOT}" \
  --output-root "${RESULTS_DIR}/pss_window_sticker/heavy_hex_144_small" \
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
  "$@"

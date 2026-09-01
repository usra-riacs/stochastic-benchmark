#!/usr/bin/env bash
# Generates and applies (or just prints) the Nautilus Job manifest for the
# zero-training "Linear Ramp" (linear_ramp_no_opt) window-sticker campaign --
# the plain (non-star) Linear Ramp point in Analysis.ipynb's hardware Pareto
# plot, which has no simulated counterpart yet (only the trained LR_PP_opt
# variant does, in heavy_hex_144_LR_opt_p9_expanded). p=9 is used to match
# that campaign's depth so the two are directly comparable on the same
# window-sticker figure.
#
# Zero-training methods (linear_ramp_no_opt, FA_PP_no_opt, PT_PP_AAA) skip
# COBYLA entirely -- there's no (N, M) grid, only a Q-only sweep -- so this
# runs in minutes, not hours, and doesn't need sharding.
#
# Usage:
#   ./lr_no_opt_job.sh [--apply]
#
# Without --apply, prints the manifest to stdout (kubectl apply -f - yourself).

set -euo pipefail

APPLY="${1:-}"

NAMESPACE="usra-expedition"
JOB_NAME="ibm-qaoa-lr-no-opt-p9"
OUTPUT_TAG="heavy_hex_144_LR_no_opt_p9_expanded"
OUTPUT_ROOT="/workspace/results/pss_window_sticker/${OUTPUT_TAG}"

# Same Q grid actually used (verified from the FA_PP_no_opt/PT_PP_AAA Qext
# run logs) for the other zero-training methods, so all three are on the
# same resource axis: 100-point geomspace from 100 to 100,000 shots.
Q_VALUES="$(python3 -c '
import numpy as np
values = np.unique(np.rint(np.geomspace(100, 100000, 100)).astype(int))
print(",".join(str(int(v)) for v in values))
')"

STOCHASTIC_BENCHMARK_REPO="https://github.com/usra-riacs/stochastic-benchmark.git"
# Defaults to main since this branch is deleted once PR #84 merges; pass an
# override for testing against an unmerged branch.
STOCHASTIC_BENCHMARK_BRANCH="${STOCHASTIC_BENCHMARK_BRANCH:-main}"
QPS_REPO_URL="https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting.git"
QAOA_PIPELINE_REPO_URL="https://github.com/qiskit-community/qaoa_training_pipeline.git"
QAOA_PIPELINE_BRANCH="v0.1.0"

emit() {
cat <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 3
  activeDeadlineSeconds: 7200
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: runner
          image: python:3.11-bookworm
          imagePullPolicy: IfNotPresent
          command:
            - /bin/bash
            - -lc
          args:
            - |
              set -euo pipefail
              export DEBIAN_FRONTEND=noninteractive
              apt-get update
              apt-get install -y --no-install-recommends git build-essential
              rm -rf /var/lib/apt/lists/*

              if [[ -n "\${GITHUB_TOKEN:-}" ]]; then
                git config --global credential.helper "store --file=/tmp/git-credentials"
                printf 'https://x-access-token:%s@github.com\n' "\${GITHUB_TOKEN}" > /tmp/git-credentials
                chmod 0600 /tmp/git-credentials
              fi

              export WORKDIR="/workspace/workdirs/lr-no-opt-p9"
              mkdir -p "\${WORKDIR}" /workspace/results /workspace/data
              rm -rf "\${WORKDIR}/repos"
              mkdir -p "\${WORKDIR}/repos"
              export REPOS_DIR="\${WORKDIR}/repos"
              export DATA_DIR="/workspace/data"
              export RESULTS_DIR="/workspace/results"

              # Same transient-network retry as interp_full_recursion_job.sh: GitHub
              # clones from some federated Nautilus nodes intermittently fail with
              # "RPC failed; curl 55/56 ...".
              git_clone_retry() {
                local branch="\$1" url="\$2" dest="\$3" attempt
                for attempt in 1 2 3 4 5; do
                  rm -rf "\${dest}"
                  if git clone --branch "\${branch}" --depth 1 "\${url}" "\${dest}"; then
                    return 0
                  fi
                  echo "git clone attempt \${attempt} failed for \${dest}, retrying in \$((attempt * 5))s..." >&2
                  sleep "\$((attempt * 5))"
                done
                echo "git clone failed after 5 attempts for \${dest}" >&2
                return 1
              }

              git_fetch_retry() {
                local dir="\$1" ref="\$2" attempt
                for attempt in 1 2 3 4 5; do
                  if git -C "\${dir}" fetch --depth 1 --filter=blob:none origin "\${ref}"; then
                    return 0
                  fi
                  echo "git fetch attempt \${attempt} failed in \${dir}, retrying in \$((attempt * 5))s..." >&2
                  sleep "\$((attempt * 5))"
                done
                echo "git fetch failed after 5 attempts in \${dir}" >&2
                return 1
              }

              git_clone_retry "\${STOCHASTIC_BENCHMARK_BRANCH}" "\${STOCHASTIC_BENCHMARK_REPO}" "\${REPOS_DIR}/stochastic-benchmark"

              # Pin QAOA-Parameter-Setting to 50a17c6, the commit before a 2026-06-29
              # rename (methods/I_MPSAer.json -> methods/I_MPSAer_opt.json) that
              # QPS_BRANCH=main would otherwise pick up, breaking load_method_config.
              mkdir -p "\${REPOS_DIR}/QAOA-Parameter-Setting"
              git -C "\${REPOS_DIR}/QAOA-Parameter-Setting" init -q
              git -C "\${REPOS_DIR}/QAOA-Parameter-Setting" remote add origin "\${QPS_REPO_URL}"
              git_fetch_retry "\${REPOS_DIR}/QAOA-Parameter-Setting" 50a17c63bbac754c95df48acd3f4d824d8707e9e
              git -C "\${REPOS_DIR}/QAOA-Parameter-Setting" sparse-checkout init --no-cone
              git -C "\${REPOS_DIR}/QAOA-Parameter-Setting" sparse-checkout set --no-cone \\
                qaoa_parameter_setting methods instances/heavy_hex data/evaluation_times \\
                data/minmax_cuts/heavy_hex setup.py requirements.txt VERSION.txt README.md
              git -C "\${REPOS_DIR}/QAOA-Parameter-Setting" checkout FETCH_HEAD
              export IBM_QAOA_SKIP_REPO_UPDATE=1

              bash "\${REPOS_DIR}/stochastic-benchmark/examples/IBM_QAOA/nautilus/run_simulation_validation.sh" \\
                --output-root ${OUTPUT_ROOT} \\
                --instance-cache-root /workspace/data/generated_instances \\
                --graph-type heavy_hex \\
                --num-nodes 144 \\
                --p-values 9 \\
                --train-count 20 \\
                --test-count 10 \\
                --start-train-index 100 \\
                --fa-method-name "" \\
                --pt-method-name linear_ramp_no_opt \\
                --q-values ${Q_VALUES} \\
                --t-grid-points 1000 \\
                --t-grid-scale log \\
                --mps-chi 20 \\
                --max-parallel-threads 4
          env:
            - name: STOCHASTIC_BENCHMARK_REPO
              value: ${STOCHASTIC_BENCHMARK_REPO}
            - name: STOCHASTIC_BENCHMARK_BRANCH
              value: ${STOCHASTIC_BENCHMARK_BRANCH}
            - name: QPS_REPO_URL
              value: ${QPS_REPO_URL}
            - name: QAOA_PIPELINE_REPO_URL
              value: ${QAOA_PIPELINE_REPO_URL}
            - name: QAOA_PIPELINE_BRANCH
              value: ${QAOA_PIPELINE_BRANCH}
            - name: GITHUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: github-credentials
                  key: token
                  optional: true
          resources:
            requests:
              cpu: "4"
              memory: 16Gi
            limits:
              cpu: "8"
              memory: 32Gi
          volumeMounts:
            - name: workspace
              mountPath: /workspace
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: ibm-qaoa-workspace
YAML
}

emit

if [[ "${APPLY}" == "--apply" ]]; then
  emit | kubectl apply -f -
fi

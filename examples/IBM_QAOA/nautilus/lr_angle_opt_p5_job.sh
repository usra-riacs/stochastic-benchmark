#!/usr/bin/env bash
# Generates and applies (or just prints) the Nautilus Job manifest for the
# LR_PP_angle_opt ("Linear Ramp*") window-sticker campaign at p=5.
#
# We already have LR_PP_opt (heavy_hex_144_LR_opt_p5/p9_expanded) -- TQATrainer
# with a real evaluator, optimizing only the 2 ramp slope parameters. That's
# the no-superscript "Linear Ramp" tier, not the starred one, despite the
# earlier "_opt" name suggesting otherwise. LR_PP_angle_opt chains the same
# ramp-slope optimization into a second ScipyTrainer stage that does full
# angle-by-angle COBYLA refinement on top -- that's the actual starred
# "Linear Ramp*" tier (method-parameter + full angle optimization), matching
# the real hardware LR_PP_angleOpt* training files and never run before now.
#
# Usage:
#   ./lr_angle_opt_p5_job.sh <setup|shards|finalize|cleanup> [--apply]
#
# Without --apply, prints the manifest to stdout (kubectl apply -f - yourself).
# With --apply, pipes directly into `kubectl apply -f -`.
#
# Sequence:
#   ./lr_angle_opt_p5_job.sh setup --apply
#   ./lr_angle_opt_p5_job.sh shards --apply
#   kubectl get pods -n usra-expedition -l job-name=ibm-qaoa-lr-angle-opt-p5-shards -w
#   # once all 10 shard pods complete:
#   ./lr_angle_opt_p5_job.sh finalize --apply

set -euo pipefail

STAGE="${1:?usage: lr_angle_opt_p5_job.sh <setup|shards|finalize|cleanup> [--apply]}"
APPLY="${2:-}"

NAMESPACE="usra-expedition"
JOB_PREFIX="ibm-qaoa-lr-angle-opt-p5"
OUTPUT_TAG="heavy_hex_144_LR_angle_opt_p5_expanded"
OUTPUT_ROOT="/workspace/results/pss_window_sticker/${OUTPUT_TAG}"

# Same N, M, Q grid used by every other FA-family window-sticker campaign
# (FA_PP_opt, I_MPSAer, LR_PP_opt) so all strategies are directly comparable
# on the same resource axis.
FA_N_VALUES="10,20,40,60,80,100,150"
FA_M_VALUES="10,25,50,100,200,500,1000"
Q_VALUES="100,250,500,1000,2500,5000,10000"

# Same compute level as the other trained (non-zero-training) shard jobs --
# LR_PP_angle_opt has a real (N, M) grid to train, unlike the zero-training
# LR_PP_opt-only variant, so this needs the full per-shard allocation.
SHARD_CPU_REQUEST="16"
SHARD_CPU_LIMIT="32"
SHARD_MEM_REQUEST="64Gi"
SHARD_MEM_LIMIT="128Gi"

emit_setup() {
cat <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_PREFIX}-setup
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 3
  activeDeadlineSeconds: 1800
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

              clone_or_update() {
                local url="\$1" branch="\$2" dest="\$3"
                if [[ -d "\${dest}/.git" ]]; then
                  git -C "\${dest}" fetch origin "\${branch}"
                  git -C "\${dest}" checkout "\${branch}"
                  git -C "\${dest}" pull --ff-only origin "\${branch}"
                else
                  git clone --branch "\${branch}" --depth 1 "\${url}" "\${dest}"
                fi
              }

              mkdir -p /workspace/repos
              clone_or_update "\${STOCHASTIC_BENCHMARK_REPO}" "\${STOCHASTIC_BENCHMARK_BRANCH}" /workspace/repos/stochastic-benchmark
              clone_or_update "\${QPS_REPO_URL}" "\${QPS_BRANCH}" /workspace/repos/QAOA-Parameter-Setting
              clone_or_update "\${QAOA_PIPELINE_REPO_URL}" "\${QAOA_PIPELINE_BRANCH}" /workspace/repos/qaoa_training_pipeline

              rm -rf ${OUTPUT_ROOT}
              echo "Setup complete: repos updated and old output cleaned."
              git -C /workspace/repos/stochastic-benchmark log --oneline -3
          env:
            - name: STOCHASTIC_BENCHMARK_REPO
              value: https://github.com/usra-riacs/stochastic-benchmark.git
            - name: STOCHASTIC_BENCHMARK_BRANCH
              value: IBM_QAOA_audit_recursion_and_paper_draft
            - name: QPS_REPO_URL
              value: https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting.git
            - name: QPS_BRANCH
              value: main
            - name: QAOA_PIPELINE_REPO_URL
              value: https://github.com/qiskit-community/qaoa_training_pipeline.git
            - name: QAOA_PIPELINE_BRANCH
              value: v0.1.0
            - name: GITHUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: github-credentials
                  key: token
                  optional: true
          resources:
            requests:
              cpu: "1"
              memory: 2Gi
            limits:
              cpu: "2"
              memory: 4Gi
          volumeMounts:
            - name: workspace
              mountPath: /workspace
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: ibm-qaoa-workspace
YAML
}

emit_shards() {
cat <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_PREFIX}-shards
  namespace: ${NAMESPACE}
spec:
  completions: 10
  parallelism: 10
  completionMode: Indexed
  # See interp_full_recursion_job.sh for why: backoffLimitPerIndex gives each
  # shard its own retry budget instead of one shared across all 10 (a single
  # flaky shard used to be able to kill the whole job); maxFailedIndexes lets
  # up to 2 shards permanently fail without taking the other 8 down.
  backoffLimitPerIndex: 4
  maxFailedIndexes: 2
  backoffLimit: 20
  activeDeadlineSeconds: 172800
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

              export WORKDIR="/workspace/workdirs/lr-angle-opt-p5-shard-\${JOB_COMPLETION_INDEX}"
              mkdir -p "\${WORKDIR}" /workspace/results /workspace/data
              rm -rf "\${WORKDIR}/repos"
              mkdir -p "\${WORKDIR}/repos"
              export REPOS_DIR="\${WORKDIR}/repos"
              export DATA_DIR="/workspace/data"
              export RESULTS_DIR="/workspace/results"

              # See interp_full_recursion_job.sh for why these retry: transient
              # GitHub egress flakiness ("RPC failed; curl 55/56 ...") from some
              # federated Nautilus nodes, not just clone contention.
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

              # Desynchronize the 10 shards' git clones (see interp_full_recursion_job.sh).
              sleep "\$(( JOB_COMPLETION_INDEX * 6 + RANDOM % 5 ))"

              git_clone_retry "\${STOCHASTIC_BENCHMARK_BRANCH}" "\${STOCHASTIC_BENCHMARK_REPO}" "\${REPOS_DIR}/stochastic-benchmark"

              # Pin QAOA-Parameter-Setting to 50a17c6, the commit before a 2026-06-29 rename
              # (methods/I_MPSAer.json -> methods/I_MPSAer_opt.json) that QPS_BRANCH=main would
              # otherwise pick up, breaking load_method_config.
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
                --shards-root ${OUTPUT_ROOT}/shards \\
                --exact-only \\
                --restart \\
                --shard-count 10 \\
                --shard-index "\${JOB_COMPLETION_INDEX}" \\
                --graph-type heavy_hex \\
                --num-nodes 144 \\
                --p-values 5 \\
                --train-count 20 \\
                --test-count 10 \\
                --start-train-index 100 \\
                --fa-method-name LR_PP_angle_opt \\
                --pt-method-name "" \\
                --fa-n-values ${FA_N_VALUES} \\
                --fa-m-values ${FA_M_VALUES} \\
                --q-values ${Q_VALUES} \\
                --t-grid-points 1000 \\
                --t-grid-scale log \\
                --mps-chi 20 \\
                --max-parallel-threads ${SHARD_CPU_REQUEST}
          env:
            - name: JOB_COMPLETION_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
            - name: STOCHASTIC_BENCHMARK_REPO
              value: https://github.com/usra-riacs/stochastic-benchmark.git
            - name: STOCHASTIC_BENCHMARK_BRANCH
              value: IBM_QAOA_audit_recursion_and_paper_draft
            - name: QPS_REPO_URL
              value: https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting.git
            - name: QPS_BRANCH
              value: main
            - name: QAOA_PIPELINE_REPO_URL
              value: https://github.com/qiskit-community/qaoa_training_pipeline.git
            - name: QAOA_PIPELINE_BRANCH
              value: v0.1.0
            - name: GITHUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: github-credentials
                  key: token
                  optional: true
          resources:
            requests:
              cpu: "${SHARD_CPU_REQUEST}"
              memory: ${SHARD_MEM_REQUEST}
            limits:
              cpu: "${SHARD_CPU_LIMIT}"
              memory: ${SHARD_MEM_LIMIT}
          volumeMounts:
            - name: workspace
              mountPath: /workspace
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: ibm-qaoa-workspace
YAML
}

emit_finalize() {
cat <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_PREFIX}-finalize
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 14400
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

              export WORKDIR="/workspace/workdirs/lr-angle-opt-p5-finalize"
              mkdir -p "\${WORKDIR}" /workspace/results /workspace/data
              rm -rf "\${WORKDIR}/repos"
              mkdir -p "\${WORKDIR}/repos"
              export REPOS_DIR="\${WORKDIR}/repos"
              export DATA_DIR="/workspace/data"
              export RESULTS_DIR="/workspace/results"

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
                --reuse-output-root ${OUTPUT_ROOT} \\
                --instance-cache-root /workspace/data/generated_instances \\
                --shards-root ${OUTPUT_ROOT}/shards \\
                --finalize-only \\
                --graph-type heavy_hex \\
                --num-nodes 144 \\
                --p-values 5 \\
                --train-count 20 \\
                --test-count 10 \\
                --start-train-index 100 \\
                --fa-method-name LR_PP_angle_opt \\
                --pt-method-name "" \\
                --fa-n-values ${FA_N_VALUES} \\
                --fa-m-values ${FA_M_VALUES} \\
                --q-values ${Q_VALUES} \\
                --t-grid-points 1000 \\
                --t-grid-scale log \\
                --mps-chi 20 \\
                --max-parallel-threads 4
          env:
            - name: STOCHASTIC_BENCHMARK_REPO
              value: https://github.com/usra-riacs/stochastic-benchmark.git
            - name: STOCHASTIC_BENCHMARK_BRANCH
              value: IBM_QAOA_audit_recursion_and_paper_draft
            - name: QPS_REPO_URL
              value: https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting.git
            - name: QPS_BRANCH
              value: main
            - name: QAOA_PIPELINE_REPO_URL
              value: https://github.com/qiskit-community/qaoa_training_pipeline.git
            - name: QAOA_PIPELINE_BRANCH
              value: v0.1.0
            - name: GITHUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: github-credentials
                  key: token
                  optional: true
          resources:
            requests:
              cpu: "2"
              memory: 8Gi
            limits:
              cpu: "4"
              memory: 16Gi
          volumeMounts:
            - name: workspace
              mountPath: /workspace
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: ibm-qaoa-workspace
YAML
}

emit_cleanup() {
cat <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_PREFIX}-cleanup
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 300
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: cleanup
          image: busybox
          command:
            - /bin/sh
            - -c
            - |
              rm -rf ${OUTPUT_ROOT}
              echo "Cleanup done"
          resources:
            requests:
              cpu: "100m"
              memory: 128Mi
            limits:
              cpu: "500m"
              memory: 256Mi
          volumeMounts:
            - name: workspace
              mountPath: /workspace
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: ibm-qaoa-workspace
YAML
}

case "${STAGE}" in
  setup) MANIFEST="$(emit_setup)" ;;
  shards) MANIFEST="$(emit_shards)" ;;
  finalize) MANIFEST="$(emit_finalize)" ;;
  cleanup) MANIFEST="$(emit_cleanup)" ;;
  *) echo "Unknown stage: ${STAGE} (expected setup|shards|finalize|cleanup)" >&2; exit 1 ;;
esac

if [[ "${APPLY}" == "--apply" ]]; then
  echo "${MANIFEST}" | kubectl apply -f -
else
  echo "${MANIFEST}"
fi

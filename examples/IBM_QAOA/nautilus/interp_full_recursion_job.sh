#!/usr/bin/env bash
# Generates and applies (or just prints) the Nautilus Job manifest for one
# stage of the Interp* full-recursion campaign, i.e. the (N, M, Q) grid now
# applied at every depth p=1..target (see the RecursionTrainer fix in
# src/simulation_validation.py's build_sampled_training_config), not just the
# p=1 warm-start stage as in the earlier heavy_hex_144_I_p7_expanded run.
#
# Writes to a NEW output root (heavy_hex_144_I_full_p<P>_expanded) so the
# earlier p=1-only-grid I_p7 data is left untouched for comparison.
#
# Usage:
#   ./interp_full_recursion_job.sh <p> <setup|shards|finalize|cleanup> [--apply]
#
# Without --apply, prints the manifest to stdout (kubectl apply -f - yourself).
# With --apply, pipes directly into `kubectl apply -f -`.
#
# Example, full sequence for depth p=3:
#   ./interp_full_recursion_job.sh 3 setup --apply
#   ./interp_full_recursion_job.sh 3 shards --apply
#   kubectl get pods -n usra-expedition -l job-name=ibm-qaoa-i-full-p3-shards -w
#   # once all 10 shard pods complete:
#   ./interp_full_recursion_job.sh 3 finalize --apply
#
# Repeat for p in 1 2 3 4 5 6 7.

set -euo pipefail

P="${1:?usage: interp_full_recursion_job.sh <p> <setup|shards|finalize|cleanup> [--apply]}"
STAGE="${2:?usage: interp_full_recursion_job.sh <p> <setup|shards|finalize|cleanup> [--apply]}"
APPLY="${3:-}"

NAMESPACE="usra-expedition"
JOB_PREFIX="ibm-qaoa-i-full-p${P}"
OUTPUT_TAG="heavy_hex_144_I_full_p${P}_expanded"
OUTPUT_ROOT="/workspace/results/pss_window_sticker/${OUTPUT_TAG}"

# Same N, M, Q grid as the earlier heavy_hex_144_I_p7_expanded run
# (nautilus/simulation-validation-i-pp-p7-shards-r5.yaml), now applied to
# every recursion depth instead of only p=1.
FA_N_VALUES="10,20,40,60,80,100,150"
FA_M_VALUES="10,25,50,100,200,500,1000"
Q_VALUES="100,250,500,1000,2500,5000,10000"

# Bumped up from the earlier run's 4/8 CPU, 32/64Gi (adjust to your actual
# namespace ResourceQuota; these are a starting point for "max out", not a
# verified ceiling since I have no cluster access to check your quota).
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
              echo "Setup complete for p=${P}: repos updated and old output cleaned."
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
  # backoffLimit alone is a job-WIDE retry budget shared across all 10
  # indices -- shard 6 hit 3 transient "git clone" network failures
  # (RPC failed; curl 55 Send failure: Broken pipe) over a 48h run and
  # that alone exceeded backoffLimit: 2, killing the whole job including
  # 5+ shards that were >40% through their work. backoffLimitPerIndex
  # gives each shard its own retry budget instead; maxFailedIndexes lets
  # up to 2 shards permanently fail without taking the other 8 down with
  # them; backoffLimit is raised to act only as an extreme global safety
  # valve, decoupled from any single index's flakiness.
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

              export WORKDIR="/workspace/workdirs/i-full-p${P}-shard-\${JOB_COMPLETION_INDEX}"
              mkdir -p "\${WORKDIR}" /workspace/results /workspace/data
              rm -rf "\${WORKDIR}/repos"
              mkdir -p "\${WORKDIR}/repos"
              export REPOS_DIR="\${WORKDIR}/repos"
              export DATA_DIR="/workspace/data"
              export RESULTS_DIR="/workspace/results"

              # Desynchronize the 10 shards' git clones: an earlier run had multiple
              # shards hit "RPC failed; curl 18 Transferred a partial file" during the
              # stochastic-benchmark clone, consistent with 10 simultaneous shallow
              # clones of the same repo landing in the same instant. Stagger by index
              # (0, 6, 12, ..., 54s) plus a little jitter so they don't all fire together.
              sleep "\$(( JOB_COMPLETION_INDEX * 6 + RANDOM % 5 ))"

              git clone --branch "\${STOCHASTIC_BENCHMARK_BRANCH}" --depth 1 \\
                "\${STOCHASTIC_BENCHMARK_REPO}" "\${REPOS_DIR}/stochastic-benchmark"

              # Pin QAOA-Parameter-Setting to 50a17c6, the commit before a 2026-06-29 rename
              # (methods/I_MPSAer.json -> methods/I_MPSAer_opt.json) that QPS_BRANCH=main would
              # otherwise pick up, breaking load_method_config. QPS has no tags and doesn't
              # support shallow-clone-by-SHA via --branch, so fetch+checkout it directly with
              # the same sparse paths run_simulation_validation.sh would normally set up, then
              # tell it to leave this checkout alone via SKIP_REPO_UPDATE.
              mkdir -p "\${REPOS_DIR}/QAOA-Parameter-Setting"
              git -C "\${REPOS_DIR}/QAOA-Parameter-Setting" init -q
              git -C "\${REPOS_DIR}/QAOA-Parameter-Setting" remote add origin "\${QPS_REPO_URL}"
              git -C "\${REPOS_DIR}/QAOA-Parameter-Setting" fetch --depth 1 --filter=blob:none origin 50a17c63bbac754c95df48acd3f4d824d8707e9e
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
                --p-values ${P} \\
                --train-count 20 \\
                --test-count 10 \\
                --start-train-index 100 \\
                --fa-method-name I_MPSAer \\
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

              export WORKDIR="/workspace/workdirs/i-full-p${P}-finalize"
              mkdir -p "\${WORKDIR}" /workspace/results /workspace/data
              rm -rf "\${WORKDIR}/repos"
              mkdir -p "\${WORKDIR}/repos"
              export REPOS_DIR="\${WORKDIR}/repos"
              export DATA_DIR="/workspace/data"
              export RESULTS_DIR="/workspace/results"

              git clone --branch "\${STOCHASTIC_BENCHMARK_BRANCH}" --depth 1 \\
                "\${STOCHASTIC_BENCHMARK_REPO}" "\${REPOS_DIR}/stochastic-benchmark"

              # Pin QAOA-Parameter-Setting to 50a17c6, the commit before a 2026-06-29 rename
              # (methods/I_MPSAer.json -> methods/I_MPSAer_opt.json) that QPS_BRANCH=main would
              # otherwise pick up, breaking load_method_config. QPS has no tags and doesn't
              # support shallow-clone-by-SHA via --branch, so fetch+checkout it directly with
              # the same sparse paths run_simulation_validation.sh would normally set up, then
              # tell it to leave this checkout alone via SKIP_REPO_UPDATE.
              mkdir -p "\${REPOS_DIR}/QAOA-Parameter-Setting"
              git -C "\${REPOS_DIR}/QAOA-Parameter-Setting" init -q
              git -C "\${REPOS_DIR}/QAOA-Parameter-Setting" remote add origin "\${QPS_REPO_URL}"
              git -C "\${REPOS_DIR}/QAOA-Parameter-Setting" fetch --depth 1 --filter=blob:none origin 50a17c63bbac754c95df48acd3f4d824d8707e9e
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
                --p-values ${P} \\
                --train-count 20 \\
                --test-count 10 \\
                --start-train-index 100 \\
                --fa-method-name I_MPSAer \\
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
              echo "Cleanup done for p=${P}"
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

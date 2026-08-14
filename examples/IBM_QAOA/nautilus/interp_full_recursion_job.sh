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
              value: https://github.com/anurag-r20/stochastic-benchmark.git
            - name: STOCHASTIC_BENCHMARK_BRANCH
              value: QAOA_Parameter_Setting_IBM
            - name: QPS_REPO_URL
              value: https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting.git
            - name: QPS_BRANCH
              value: main
            - name: QAOA_PIPELINE_REPO_URL
              value: https://github.com/qiskit-community/qaoa_training_pipeline.git
            - name: QAOA_PIPELINE_BRANCH
              value: main
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
  backoffLimit: 2
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

              git clone --branch "\${STOCHASTIC_BENCHMARK_BRANCH}" --depth 1 \\
                "\${STOCHASTIC_BENCHMARK_REPO}" "\${REPOS_DIR}/stochastic-benchmark"

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
                --max-parallel-threads 4
          env:
            - name: JOB_COMPLETION_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
            - name: STOCHASTIC_BENCHMARK_REPO
              value: https://github.com/anurag-r20/stochastic-benchmark.git
            - name: STOCHASTIC_BENCHMARK_BRANCH
              value: QAOA_Parameter_Setting_IBM
            - name: QPS_REPO_URL
              value: https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting.git
            - name: QPS_BRANCH
              value: main
            - name: QAOA_PIPELINE_REPO_URL
              value: https://github.com/qiskit-community/qaoa_training_pipeline.git
            - name: QAOA_PIPELINE_BRANCH
              value: main
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
              value: https://github.com/anurag-r20/stochastic-benchmark.git
            - name: STOCHASTIC_BENCHMARK_BRANCH
              value: QAOA_Parameter_Setting_IBM
            - name: QPS_REPO_URL
              value: https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting.git
            - name: QPS_BRANCH
              value: main
            - name: QAOA_PIPELINE_REPO_URL
              value: https://github.com/qiskit-community/qaoa_training_pipeline.git
            - name: QAOA_PIPELINE_BRANCH
              value: main
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

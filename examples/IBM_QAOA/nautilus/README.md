# Nautilus IBM QAOA Run

These manifests run the IBM QAOA simulation-validation campaign on Nautilus in
the `usra-expedition` namespace. VS Code remains the control surface, but the
code executes inside a Kubernetes pod. The pod clones the full
`stochastic-benchmark` repository into `/workspace/repos/stochastic-benchmark`;
the Nautilus helper files are only launch instructions.

## Files

- `pvc.yaml` creates a shared PVC mounted at `/workspace`.
- `simulation-validation-job.yaml` launches the batch run.
- `simulation-validation-sharded-job.yaml` launches the Nautilus-only sharded
  exact-point run.
- `simulation-validation-finalize-job.yaml` merges shard outputs and writes the
  final frontier files after the sharded job completes.
- `dev-pod.yaml` launches an idle pod you can attach to from VS Code.
- `run_simulation_validation.sh` clones/updates the full `stochastic-benchmark`
  repo, sparse-checks out the large `QAOA-Parameter-Setting` dependency paths
  needed by the run, installs Python dependencies, and runs
  `examples/IBM_QAOA/run_prepare_pss_campaign.py`.

## Before Running

Commit and push the branch that Nautilus should execute:

```bash
git push upstream QAOA_Parameter_Setting_IBM
```

If your namespace spelling differs from `usra-expedition`, update the
`metadata.namespace` field in the YAML files.

If any dependency repo is private, create a GitHub token secret in Nautilus.
Use a fine-grained token with read access to:

- `usra-riacs/stochastic-benchmark`
- `Quantum-Working-Groups/QAOA-Parameter-Setting`
- `qiskit-community/qaoa_training_pipeline`

Create the secret without echoing the token:

```bash
read -rsp "GitHub token: " GITHUB_TOKEN
kubectl create secret generic github-credentials \
  -n usra-expedition \
  --from-literal=token="${GITHUB_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -
unset GITHUB_TOKEN
```

## Create Storage

```bash
kubectl apply -f examples/IBM_QAOA/nautilus/pvc.yaml
kubectl get pvc -n usra-expedition
```

The default storage class is `rook-cephfs`. If Nautilus reports that this class
does not exist, replace it with the storage class available in your namespace.

## Submit The Batch Job

The regular manifest runs the same single-process workflow as the local script:

```bash
kubectl apply -f examples/IBM_QAOA/nautilus/simulation-validation-job.yaml
kubectl get pods -n usra-expedition -w
```

Follow logs:

```bash
kubectl logs -n usra-expedition -f job/ibm-qaoa-fa-opt-p5
```

Results are written under:

```text
/workspace/results/pss_window_sticker/heavy_hex_144_FA_opt_p5_expanded
```

## Submit The Sharded Run

Use this only on Nautilus. It is an opt-in parallel path that calls
`run_prepare_pss_campaign.py` with `--exact-only`, `--shard-index`, and
`--shard-count`. Laptop/local scripts do not use these flags.

Start ten exact-point shards:

```bash
kubectl apply -f examples/IBM_QAOA/nautilus/simulation-validation-sharded-job.yaml
kubectl get pods -n usra-expedition -l job-name=ibm-qaoa-fa-opt-p5-shards -w
```

Follow one shard:

```bash
kubectl logs -n usra-expedition -f job/ibm-qaoa-fa-opt-p5-shards
```

After all shard pods complete, merge cached root rows plus all shard rows and
write the canonical final frontier files:

```bash
kubectl apply -f examples/IBM_QAOA/nautilus/simulation-validation-finalize-job.yaml
kubectl logs -n usra-expedition -f job/ibm-qaoa-fa-opt-p5-finalize
```

Shard outputs are written under:

```text
/workspace/results/pss_window_sticker/heavy_hex_144_FA_opt_p5_expanded/shards/shard-XX
```

The generated instance/minmax cache is stored under:

```text
/workspace/data/generated_instances
```

## VS Code Interactive Pod

Start the dev pod:

```bash
kubectl apply -f examples/IBM_QAOA/nautilus/dev-pod.yaml
```

Then attach VS Code to `ibm-qaoa-dev` using the Kubernetes extension or exec
into it:

```bash
kubectl exec -n usra-expedition -it pod/ibm-qaoa-dev -- bash
```

Inside the pod, you can run:

```bash
cd /workspace/repos/stochastic-benchmark
bash examples/IBM_QAOA/nautilus/run_simulation_validation.sh
```

## Cleanup

Delete the job or dev pod when finished:

```bash
kubectl delete -f examples/IBM_QAOA/nautilus/simulation-validation-job.yaml
kubectl delete -f examples/IBM_QAOA/nautilus/simulation-validation-sharded-job.yaml
kubectl delete -f examples/IBM_QAOA/nautilus/simulation-validation-finalize-job.yaml
kubectl delete -f examples/IBM_QAOA/nautilus/dev-pod.yaml
```

Do not delete the PVC unless you want to remove cached instances and results:

```bash
kubectl delete -f examples/IBM_QAOA/nautilus/pvc.yaml
```

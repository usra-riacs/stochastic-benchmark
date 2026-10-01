# Nautilus IBM QAOA Run

These manifests run the IBM QAOA simulation-validation campaign on Nautilus in
the `usra-expedition` namespace. VS Code remains the control surface, but the
code executes inside a Kubernetes pod. The pod clones the full
`stochastic-benchmark` repository into `/workspace/repos/stochastic-benchmark`;
the Nautilus helper files are only launch instructions.

## Files

- `pvc.yaml` creates a shared PVC mounted at `/workspace`.
- `simulation-validation-lr-opt-p7-shards.yaml` runs a training campaign
  (here Linear Ramp, `LR_PP_opt`, at p=7) as ten exact-point shards.
- `simulation-validation-lr-opt-p7-finalize-job.yaml` merges those shards and
  writes the final frontier files after they complete.
- `simulation-validation-fa-no-opt-p7-job.yaml` runs a zero-training campaign
  (Fixed Angles-dagger, `FA_PP_no_opt`, at p=7) as a single pod; a Q-only
  sweep is cheap enough not to need shards.
- `dev-pod.yaml` launches an idle pod you can attach to from VS Code.
- `run_simulation_validation.sh` clones/updates the full `stochastic-benchmark`
  repo, sparse-checks out the large `QAOA-Parameter-Setting` dependency paths
  needed by the run, installs Python dependencies, and runs
  `examples/IBM_QAOA/run_prepare_pss_campaign.py`.

These three are the templates for any new campaign: copy one, then change the
job name, the `--output-root`, `--p-values`, and the method flags
(`--fa-method-name` for a family that trains, `--pt-method-name` for a
zero-training one; `run_job.sh` shows which is which).

## Before Running

The manifests clone `STOCHASTIC_BENCHMARK_BRANCH` from this repository, so
push the branch you want run first:

```bash
git push upstream your-feature-branch
```

They also pin the two dependency repositories to exact commits through the
`QPS_COMMIT` and `QAOA_PIPELINE_COMMIT` env vars. Keep those pins. Both
upstreams have moved on incompatibly since the campaigns on disk were run
(`qaoa_training_pipeline` no longer exports the `TRAINERS` registry that
`src/simulation_validation.py` imports), so a manifest that tracks their
`main` fails at import time, and a new campaign should in any case come from
the same code as the ones it will share a Pareto frontier with.

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

## Submit A Single-Pod Campaign

For a zero-training family the whole sweep fits in one pod:

```bash
kubectl apply -f examples/IBM_QAOA/nautilus/simulation-validation-fa-no-opt-p7-job.yaml
kubectl get pods -n usra-expedition -w
kubectl logs -n usra-expedition -f job/ibm-qaoa-fa-no-opt-p7
```

Results are written under:

```text
/workspace/results/pss_window_sticker/heavy_hex_144_FA_no_opt_p7_expanded
```

## Submit A Sharded Campaign

A family that trains runs an (N, M, Q) grid per instance and takes hours, so
it is split ten ways by instance. This is a Nautilus-only path that calls
`run_prepare_pss_campaign.py` with `--exact-only`, `--shard-index`, and
`--shard-count`; laptop/local scripts do not use these flags.

Start ten exact-point shards:

```bash
kubectl apply -f examples/IBM_QAOA/nautilus/simulation-validation-lr-opt-p7-shards.yaml
kubectl get pods -n usra-expedition -l batch.kubernetes.io/job-name=ibm-qaoa-lr-opt-p7-shards -w
```

After all ten complete, merge the shards and write the canonical final
frontier files:

```bash
kubectl apply -f examples/IBM_QAOA/nautilus/simulation-validation-lr-opt-p7-finalize-job.yaml
kubectl logs -n usra-expedition -f job/ibm-qaoa-lr-opt-p7-finalize
```

Shard outputs are written under:

```text
/workspace/results/pss_window_sticker/heavy_hex_144_LR_opt_p7_expanded/shards/shard-XX
```

The merged `strategy_raw_points.pkl` carries a per-shot `counts` histogram
and is ~2.3 GB; everything downstream needs only the other columns, so pull a
counts-free copy rather than the file itself (`run_latency_recost.py`'s
`SLIM_COLUMNS` lists what is needed).

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
kubectl delete -f examples/IBM_QAOA/nautilus/simulation-validation-lr-opt-p7-shards.yaml
kubectl delete -f examples/IBM_QAOA/nautilus/simulation-validation-lr-opt-p7-finalize-job.yaml
kubectl delete -f examples/IBM_QAOA/nautilus/simulation-validation-fa-no-opt-p7-job.yaml
kubectl delete -f examples/IBM_QAOA/nautilus/dev-pod.yaml
```

Do not delete the PVC unless you want to remove cached instances and results:

```bash
kubectl delete -f examples/IBM_QAOA/nautilus/pvc.yaml
```

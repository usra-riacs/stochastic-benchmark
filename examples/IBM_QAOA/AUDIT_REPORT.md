# IBM QAOA Codebase Audit Report

**Date:** 2026-06-25
**Scope:** `src/simulation_validation.py`, `src/utils.py`, `run_prepare_pss_campaign.py`, `run_job.sh`, `nautilus/run_simulation_validation.sh`

---

## Bugs

**Bug 1 — Dead duplicate function `_hardware_result_depths`**
- **Location:** `simulation_validation.py:206–245`
- **Description:** Verbatim copy of the nested `_result_depths` closure inside `estimate_hardware_time_per_shot`. Never called anywhere in the codebase. If the inner closure gets updated, this silent copy diverges.
- **Impact:** Dead code / maintenance hazard.

---

**Bug 2 — `FA_PP_no_opt` in shell's zero-training list but not Python's**
- **Location:** `run_job.sh:71` vs `simulation_validation.py:62`
- **Description:** `run_job.sh` routes `FA_PP_no_opt` → `run_pt_pss_exact_points` (which strips evaluators for `FixedAngleConjecture`). But Python's `ZERO_TRAINING_METHODS` set only contains `PT_PP_AAA` and `linear_ramp_no_opt`. Any code path that checks the Python set won't match `FA_PP_no_opt` — the two lists are out of sync.
- **Fix:** Add `FA_PP_no_opt` to `ZERO_TRAINING_METHODS` in `simulation_validation.py`.

---

**Bug 3 — Asymmetric random-weight matrix for `line_to_full` graphs**
- **Location:** `simulation_validation.py:837–840`
- **Description:**
  ```python
  adjacency = adjacency * rng.choice([1, -1], (n, n))
  ```
  Generates an independent sign per `(i,j)` entry so the matrix is not symmetric. `nx.from_numpy_array` only reads the upper triangle, so half the random draws are wasted and edge signs are determined purely by upper-triangular entries.
- **Fix:** Symmetrize before multiplying:
  ```python
  signs = rng.choice([1, -1], (n, n))
  signs = np.triu(signs) + np.triu(signs, 1).T
  adjacency = adjacency * signs
  ```

---

**Bug 4 — Double-accumulation before dedup in `generate_pss_exact_points`**
- **Location:** `simulation_validation.py:3637–3642`
- **Description:** `cached_exact_df` is already grown by `_append_exact_checkpoint` and contains every row from `exact_frames`. The code then concatenates both before deduplicating, so every newly computed row appears twice before `_deduplicate_exact_points` runs.
- **Fix:** Omit `exact_frames` from the concat when `cached_exact_df` is already current.

---

**Bug 5 — Fragile `result_df.get("error")` pattern**
- **Location:** `simulation_validation.py:2039`
- **Description:** `pd.DataFrame.get(key)` returns `None` (not a Series) when the key is absent. The `"error" in result_df.columns` guard currently prevents the crash, but if the condition is ever inverted or reordered, `.isna()` on `None` raises `AttributeError` with no useful message.
- **Fix:** Use `result_df["error"].isna()` with an explicit column guard.

---

**Bug 6 — `sem_total` aligned via `.values` across two independent groupbys**
- **Location:** `utils.py:~1524`
- **Description:**
  ```python
  agg["sem_total"] = df_rows.groupby(...)["brick_total"].sem().values
  ```
  Strips the MultiIndex and relies on both groupbys having identical row ordering. Silently misaligns if group keys duplicate or pandas sort behavior changes — error bars get drawn on the wrong bars.
- **Fix:** Use a merge instead of `.values` assignment:
  ```python
  sem_df = df_rows.groupby(["job_p", "method_base"])["brick_total"].sem().rename("sem_total").reset_index()
  agg = agg.merge(sem_df, on=["job_p", "method_base"])
  ```

---

**Bug 7 — `fig.canvas.draw()` called inside annotation offset loop**
- **Location:** `utils.py:~2210–2260`
- **Description:** Called once per (frontier-point × offset-candidate) combination — up to 112 full figure renders per `plot_ibm_qaoa_recommendation` call. Beyond being slow, if renderer state is stale after `candidate.remove()`, collision detection queries stale bounding boxes and overlapping annotations can get placed.
- **Fix:** Call `fig.canvas.draw()` once before the outer loop, cache `renderer = fig.canvas.get_renderer()`, and pass it to all `get_window_extent(renderer=renderer)` calls inside the loop.

---

**Bug 8 — Single-qubit Z sign convention may diverge between evaluators**
- **Location:** `simulation_validation.py:1284–1292`
- **Description:**
  ```python
  if len(indices) == 1:
      energy += -coeff if sample[indices[0]] else coeff
  ```
  If the sign convention in `MPSAerSampleEvaluator.energy()` differs from `maxcut_energy_from_bitstring`, then `expected_energy_eval` and `expected_energy_from_counts` diverge silently — looks like a measurement error rather than a code bug.
- **Impact:** Silent numerical inconsistency for graphs with single-qubit Z terms.

---

**Bug 9 — `RecursionTrainer` nested chi override silently skipped**
- **Location:** `simulation_validation.py:1759–1767`
- **Description:** The outer evaluator's shot count is always overwritten with `sample_config["chi"]`, but the nested evaluator's bond dimension uses `setdefault` — so if the config JSON already defines `matrix_product_state_max_bond_dimension`, the CLI `--mps-chi` override is silently ignored for the inner evaluator.
- **Impact:** Could cause OOM or incorrect chi at scale; use of wrong chi is undetected.
- **Fix:** Replace `setdefault` with explicit assignment, matching the outer evaluator's behavior.

---

**Bug 10 — `latest_result[result_key]` subscript on unverified type**
- **Location:** `simulation_validation.py:~1127–1139`
- **Description:** `trainer.train()` return value is subscripted as a dict without verifying the type. If the trainer doesn't implement `__getitem__`, raises `TypeError` with no diagnostic message. Only triggered when `train_kwargs` contains a `"result"` key for chained stages.
- **Fix:** Add a type check or `isinstance` guard with an informative error message.

---

## Performance

**Perf 1 — O(N²) pickle I/O: full DataFrame rewritten on every checkpoint**
- **Location:** `simulation_validation.py:3544–3562`
- **Description:** `_append_exact_checkpoint` is called after every individual result. The k-th call writes k rows, so total bytes written ≈ O(N²/2). For 30 instances × 5 depths × 49 FA points (~7,350 total rows), the last checkpoint alone writes the full DataFrame.
- **Fix:** Batch checkpoints every K=10 completions, or append incrementally to a log file and rewrite the canonical pkl only at depth-level boundaries.

---

**Perf 2 — 49 full AerSimulator results materialized simultaneously in memory**
- **Location:** `simulation_validation.py:~2739–2746`
- **Description:** At 10,000 shots × 144-char bitstrings per call across a 7×7 (N,M) grid, peak memory ≈ 70 MB in bitstring streams alone, plus intermediate DataFrames. All 49 results exist in RAM simultaneously during iteration.
- **Fix:** Reuse sample streams when angles haven't changed between adjacent grid points (warm-start sampling), or process each stream via generator.

---

**Perf 3 — `expand_metric_rows_to_shots` makes one dict copy per shot**
- **Location:** `simulation_validation.py:1450–1456`
- **Description:**
  ```python
  expanded_rows.extend([row.copy() for _ in range(count)])
  ```
  Up to 10,000 dict copies with ~6 keys each per evaluation. The downstream KS test only needs a weighted array of `approximation_ratio`.
- **Fix:** Replace with vectorized expansion:
  ```python
  values = metric_rows["approximation_ratio"].values
  counts = metric_rows["count"].astype(int).values
  expanded = np.repeat(values, counts)
  ```
  Estimated speedup: 10–100×.

---

**Perf 4 — `AerSimulator` instantiated fresh on every `sample_bound_circuit_*` call**
- **Location:** `simulation_validation.py:~1372, 1396`
- **Description:**
  ```python
  AerSimulator(**options).run(...)
  ```
  Rebuilds the Aer C++ backend from scratch 49 times per `run_fa_pss_exact_points` call (once per N,M pair). Constructor overhead is non-trivial.
- **Fix:** Instantiate once outside the (N,M) loop and reuse across calls.

---

**Perf 5 — 112 full figure renders in `plot_ibm_qaoa_recommendation`**
- **Location:** `utils.py:~2210–2260` (see also Bug 7)
- **Description:** `fig.canvas.draw()` triggers a complete matplotlib figure render for every (frontier-point, offset-candidate) pair. At 8 frontier points × 14 candidate offsets = 112 full renders per call.
- **Fix:** Call `fig.canvas.draw()` once before the outer loop, cache the renderer, and pass it to all `get_window_extent()` calls inside.

---

**Perf 6 — `normalize_sb_pickle_files` reads and rewrites every bootstrap pkl unconditionally**
- **Location:** `simulation_validation.py:~4509–4513`
- **Description:** Only converts `pd.StringDtype` columns to `object` — a trivially detectable condition. Reads and re-serializes every file regardless of whether normalization is needed.
- **Fix:** Check dtypes before loading the full DataFrame; skip re-serialization if all columns are already compatible.

---

**Perf 7 — Quadratic dedup work in `_append_exact_checkpoint` inner accumulation**
- **Location:** `simulation_validation.py:3550–3556`
- **Description:**
  ```python
  cached_exact_df = _deduplicate_exact_points(
      pd.concat([cached_exact_df, df_new], ignore_index=True)
  )
  ```
  O(n_accumulated) concat + dedup on every new result. For 150+ spec/reps combinations, total dedup work is O(150²/2) ≈ 11,250 operations on progressively larger frames.
- **Fix:** Accumulate into a list, concat only before cache-completeness checks, dedup only at checkpoint-write boundaries.

---

**Perf 8 — `build_budget_frontier` deep-copies the `counts` column unnecessarily**
- **Location:** `simulation_validation.py:~3008`
- **Description:**
  ```python
  work = exact_df.copy()
  ```
  The `counts` column contains per-bitstring count dicts (thousands of entries per row) and is deep-copied by `.copy()`, then dropped later in `build_pss_frontier_outputs` after the copy cost is already paid.
- **Fix:** Drop `counts` before passing to `build_budget_frontier`:
  ```python
  exact_df_no_counts = exact_df.drop(columns=["counts"], errors="ignore")
  ```

---

**Perf 9 — `_load_exact_cache_frames` loads both canonical and legacy pickles unconditionally**
- **Location:** `run_prepare_pss_campaign.py:56–75`
- **Description:** Loads all four filenames (canonical + legacy) then concatenates, doubling peak memory when both exist.
- **Fix:** Check for canonical files first; fall back to legacy filenames only when canonical files are absent.

---

**Perf 10 — Nested `df.get()` always evaluates all branches eagerly**
- **Location:** `simulation_validation.py:~4467–4490`
- **Description:**
  ```python
  df["response_metric"] = df.get(
      "best_found_value",
      df.get("BestApproximationRatio", df.get("Approximation_Ratio")),
  )
  ```
  `pd.DataFrame.get(key, default)` evaluates `default` before checking if `key` exists. If no matching column is found, `None` is silently assigned as the entire column with no warning.
- **Fix:** Use explicit `if "col" in df.columns` guards with a `raise KeyError` fallback.

---

*Highest-priority items: **Perf 1 + 7** (quadratic pkl writes), **Perf 3** (dict copies per shot — easy win), **Perf 4** (AerSimulator reuse), **Bug 2** (Python/shell `ZERO_TRAINING_METHODS` sync).*

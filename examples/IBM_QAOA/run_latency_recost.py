#!/usr/bin/env python3
"""Rebuild the Window Sticker campaign outputs under a circuit-preparation charge.

The published proxy resource model bills only shot time,

    T_proxy = t_preprocessing + N M t_shot + Q t_shot

which assumes the shots are the whole cost of getting results off the device.
Hardware measurements disagree: one submitted circuit takes on the order of
ten seconds to come back whether it asks for ten shots or ten thousand,
because compilation, transfer, queueing and control-electronics load dominate.
Charging every submitted job for that fixed cost gives

    T_proxy = t_preprocessing + n_evals (t_prep + M t_shot) + (t_prep + Q t_shot)

where ``n_evals`` is the number of objective evaluations the optimizer
actually ran, which is somewhat more than the requested COBYLA ``maxiter``.

**No simulation is re-run.** The (N, M, Q) -> approximation-ratio mapping is
physics and does not depend on the cost model, so this reads the existing
exact points, re-costs them, and redoes only the cheap post-processing
(budget frontier plus the stochastic-benchmark prescription pipeline).

Both the baseline and the charged variant are regenerated through this same
code path, so the two differ by the preparation charge alone rather than by
months of accumulated code drift.

Usage
-----
    python run_latency_recost.py                      # baseline + 13.87 s
    python run_latency_recost.py --circuit-prep-time 10
    python run_latency_recost.py --tags heavy_hex_144_PT_p5_expanded
    python run_latency_recost.py --q-cap 10000        # fairness sensitivity

Writes ``<results-base>/<tag>__<label>/`` per campaign, laid out exactly like
the original roots so the notebook's ``load_multi_strategy_summaries`` reads
them with no changes.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for path in (REPO_ROOT / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.simulation_validation import (  # noqa: E402
    build_pss_proxy_costs,
    build_resource_frontier_from_exact_points,
    build_strategy_budget_summary,
    estimate_hardware_time_per_shot,
    recost_exact_points_with_circuit_prep,
    run_stochastic_benchmark_pss,
)

# The strategy roots Fig. 12 compares (COMPARISON_RESULT_TAGS in the notebook).
DEFAULT_TAGS = [
    "heavy_hex_144_FA_PP_opt_p5_expanded",
    "heavy_hex_144_FA_PP_opt_p6_expanded",
    "heavy_hex_144_FA_PP_opt_p7_expanded",
    "heavy_hex_144_LR_opt_p5_expanded",
    "heavy_hex_144_LR_angle_opt_p5_expanded",
    "heavy_hex_144_I_full_p7_expanded",
    "heavy_hex_144_PT_p7_expanded",
    "heavy_hex_144_PT_p6_expanded",
    "heavy_hex_144_PT_p5_expanded",
    "heavy_hex_144_PT_p3_expanded",
    "heavy_hex_144_PT_p2_expanded",
    "heavy_hex_144_FA_no_opt_p2_expanded",
    "heavy_hex_144_FA_no_opt_p3_expanded",
    "heavy_hex_144_FA_no_opt_p4_expanded",
    "heavy_hex_144_FA_no_opt_p6_expanded",
]

RAW_POINT_FILENAMES = [
    ("strategy_raw_points.pkl", "fa_raw_points.pkl"),
    ("transfer_raw_points.pkl", "pt_raw_points.pkl"),
]

# Everything the frontier and prescription pipeline needs. The raw pickles
# also carry a per-shot `counts` histogram that makes them ~2 GB; dropping it
# is what lets the slim cache be a couple of megabytes.
SLIM_COLUMNS = [
    "graph_type", "num_nodes", "instance", "instance_id", "split", "source", "p",
    "strategy", "strategy_family", "strategy_runtime_label", "simulation_method",
    "N", "M", "Q", "classical_setup_cost", "training_cost", "sampling_cost",
    "T_exact", "num_objective_evaluations", "total_training_shots",
    "runtime_train_wallclock", "runtime_sample_wallclock",
    "BestApproximationRatio", "Approximation_Ratio",
]


def load_exact_points(root: Path, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Load a campaign's exact points, caching a counts-free copy alongside."""
    cache_path = cache_dir / f"{root.name}.pkl"
    if cache_path.exists() and not refresh:
        return pd.read_pickle(cache_path)

    frames = []
    for primary, legacy in RAW_POINT_FILENAMES:
        path = root / primary
        if not path.exists():
            path = root / legacy
        if path.exists():
            frame = pd.read_pickle(path)
            frames.append(frame.loc[:, [c for c in SLIM_COLUMNS if c in frame.columns]].copy())
            del frame
            gc.collect()
    if not frames:
        raise FileNotFoundError(f"No raw exact-point pickle under {root}")

    exact_df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    cache_dir.mkdir(parents=True, exist_ok=True)
    exact_df.to_pickle(cache_path)
    return exact_df


def hardware_shot_times(exact_df: pd.DataFrame, hardware_root: Path, graph_type: str,
                        num_nodes: int, fallback: float) -> dict[int, float]:
    """Per-depth measured hardware shot times, matching the notebook."""
    if "p" not in exact_df.columns:
        return {}
    depths = sorted(pd.to_numeric(exact_df["p"], errors="coerce").dropna().round().astype(int).unique())
    by_p: dict[int, float] = {}
    for depth in depths:
        try:
            by_p[int(depth)] = estimate_hardware_time_per_shot(
                hardware_root, graph_type=graph_type, num_nodes=num_nodes,
                job_p=int(depth), statistic="median", normalize_by_pubs=True,
            )
        except (FileNotFoundError, ValueError):
            by_p[int(depth)] = fallback
    return by_p


def build_variant(exact_df: pd.DataFrame, out_root: Path, codebooks: dict,
                  *, circuit_prep_time: float, num_bins: int,
                  bootstrap_range, train_test_split: float) -> dict:
    """Cost, bin and run the prescription pipeline for one variant."""
    priced = exact_df
    if circuit_prep_time > 0:
        # use_recorded_shots=False keeps the published N*M shot accounting, so
        # the only difference from the baseline is the preparation charge.
        priced = recost_exact_points_with_circuit_prep(
            exact_df,
            circuit_prep_time=circuit_prep_time,
            time_per_shot=None,
            use_recorded_shots=False,
            charge_sampling_job=True,
        )
        priced["T_proxy"] = priced["T_exact_proxy"]

    frontier_df = build_resource_frontier_from_exact_points(
        priced, codebooks=codebooks, num_bins=num_bins, budget_col="T_proxy", scale="log",
    )
    out_root.mkdir(parents=True, exist_ok=True)
    results = run_stochastic_benchmark_pss(
        frontier_df,
        output_dir=out_root / "window_sticker",
        codebooks=codebooks,
        response_col="BestApproximationRatio",
        resource_col="T",
        response_key="Response",
        bootstrap_range=bootstrap_range,
        train_test_split=train_test_split,
        clear_checkpoints=True,
    )
    # load_multi_strategy_summaries looks for this first and only falls back to
    # rebuilding it from a frontier pickle we do not write here.
    budget_summary = build_strategy_budget_summary(frontier_df, split="train")
    if not budget_summary.empty:
        budget_summary.to_csv(out_root / "strategy_budget_summary_train.csv", index=False)
    frontier_df.to_pickle(out_root / "pss_exp_raw_frontier.pkl")
    (out_root / "ws_codebooks.json").write_text(json.dumps(codebooks, indent=2), encoding="utf-8")
    return {"frontier_rows": len(frontier_df), "results": results, "priced": priced}


def variant_label(circuit_prep_time: float) -> str:
    if circuit_prep_time <= 0:
        return "prep0"
    return "prep" + f"{circuit_prep_time:g}".replace(".", "p") + "s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-base", default=str(HERE / "results" / "pss_window_sticker"))
    parser.add_argument("--tags", default=None,
                        help="Comma-separated campaign roots (default: the Fig. 12 set).")
    parser.add_argument("--circuit-prep-time", type=float, default=13.87,
                        help="Seconds charged per submitted circuit job.")
    parser.add_argument("--q-cap", type=int, default=0,
                        help="Drop points above this Q. Use to compare families on a "
                             "common sampling grid; 0 disables.")
    parser.add_argument("--baseline", default=True, action=argparse.BooleanOptionalAction,
                        help="Also regenerate the zero-charge baseline through the same path.")
    parser.add_argument("--graph-type", default="heavy_hex")
    parser.add_argument("--num-nodes", type=int, default=144)
    parser.add_argument("--hardware-root", default=None)
    parser.add_argument("--time-per-shot", type=float, default=1.0 / 2470.0,
                        help="Fallback shot time when no hardware measurement applies.")
    parser.add_argument(
        "--shot-time-by-depth", default=None,
        help='JSON depth->seconds map, e.g. \'{"5": 4.05e-4, "6": 4.05e-4}\'. Use this to '
             "emit resources directly on the paper's calibrated shot basis. The preparation "
             "charge is real wall-clock seconds and is never rescaled, so a figure mixing the "
             "two must fix the shot basis here rather than rescaling the total afterwards.",
    )
    parser.add_argument(
        "--no-hardware-shot-times", dest="use_hardware_shot_times", action="store_false",
        default=True, help="Skip the measured per-depth hardware lookup.",
    )
    parser.add_argument("--num-bins", type=int, default=1000)
    parser.add_argument("--train-test-split", type=float, default=0.5)
    parser.add_argument("--bootstrap-start", type=int, default=10)
    parser.add_argument("--bootstrap-stop", type=int, default=51)
    parser.add_argument("--bootstrap-step", type=int, default=10)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args(argv)

    results_base = Path(args.results_base).resolve()
    cache_dir = results_base / "_slim_cache"
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else list(DEFAULT_TAGS)
    bootstrap_range = range(args.bootstrap_start, args.bootstrap_stop, args.bootstrap_step)

    hardware_root = Path(args.hardware_root) if args.hardware_root else (
        Path(__import__("os").environ.get(
            "QAOA_PARAMETER_SETTING_ROOT", REPO_ROOT.parent / "QAOA-Parameter-Setting"
        )) / "data" / "hardware"
    )

    variants = []
    if args.baseline:
        variants.append(0.0)
    if args.circuit_prep_time > 0:
        variants.append(float(args.circuit_prep_time))

    print(f"results base : {results_base}")
    print(f"hardware root: {hardware_root}")
    print(f"campaigns    : {len(tags)}")
    print(f"variants     : {[variant_label(v) for v in variants]}")
    print(f"Q cap        : {args.q_cap or 'none'}")
    print()

    failures = []
    started = time.time()
    for tag in tags:
        root = results_base / tag
        if not root.exists():
            print(f"  SKIP {tag} (no such root)")
            failures.append((tag, "missing root"))
            continue
        try:
            exact_df = load_exact_points(root, cache_dir, refresh=args.refresh_cache)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {tag}: {type(exc).__name__}: {exc}")
            failures.append((tag, f"load: {exc}"))
            continue

        if args.q_cap:
            exact_df = exact_df[pd.to_numeric(exact_df["Q"], errors="coerce") <= args.q_cap]

        codebook_path = root / "ws_codebooks.json"
        codebooks = json.loads(codebook_path.read_text()) if codebook_path.exists() else {}

        if args.shot_time_by_depth:
            by_p = {int(k): float(v) for k, v in json.loads(args.shot_time_by_depth).items()}
        elif args.use_hardware_shot_times:
            by_p = hardware_shot_times(exact_df, hardware_root, args.graph_type,
                                       args.num_nodes, args.time_per_shot)
        else:
            by_p = {}
        priced_base = build_pss_proxy_costs(
            exact_df, time_per_shot_by_p=by_p, default_time_per_shot=args.time_per_shot,
        )

        for prep in variants:
            label = variant_label(prep)
            if args.q_cap:
                label += f"_Q{args.q_cap}"
            out_root = results_base / f"{tag}__{label}"
            t0 = time.time()
            try:
                info = build_variant(
                    priced_base, out_root, codebooks,
                    circuit_prep_time=prep, num_bins=args.num_bins,
                    bootstrap_range=bootstrap_range, train_test_split=args.train_test_split,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {tag} [{label}]: {type(exc).__name__}: {exc}")
                failures.append((f"{tag}[{label}]", str(exc)))
                continue
            span = info["priced"]["T_proxy"]
            print(f"  {tag:44s} {label:14s} {time.time()-t0:6.1f}s  "
                  f"rows={info['frontier_rows']:6d}  T=[{span.min():.4g}, {span.max():.4g}] s")
        del exact_df, priced_base
        gc.collect()

    print()
    print(f"total {time.time()-started:.1f}s")
    if failures:
        print(f"{len(failures)} failure(s):")
        for tag, why in failures:
            print(f"  {tag}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

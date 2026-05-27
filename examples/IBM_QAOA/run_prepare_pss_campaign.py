from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.simulation_validation import (  # noqa: E402
    DEFAULT_INSTANCE_CACHE_ROOT,
    DEFAULT_MAIN_REPO,
    DEFAULT_PIPELINE_REPO,
    build_pss_frontier_outputs,
    build_t_grid,
    build_train_test_instance_sets,
    ensure_qiskit_imports,
    filter_pss_exact_points,
    generate_pss_exact_points,
    instance_specs_to_dataframe,
    prepare_pss_exp_raw_dataset,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _normalize_optional_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if value == "" or value.lower() == "none":
        return None
    return value


def _json_dumps_pretty(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _load_exact_cache_frames(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in paths:
        for filename in ("fa_raw_points.pkl", "pt_raw_points.pkl"):
            candidate = root / filename
            if candidate.exists():
                frames.append(pd.read_pickle(candidate))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _shard_specs(
    train_specs: list,
    test_specs: list,
    *,
    shard_index: int | None,
    shard_count: int | None,
) -> tuple[list, list]:
    if shard_index is None and shard_count is None:
        return train_specs, test_specs
    if shard_index is None or shard_count is None:
        raise ValueError("Provide both --shard-index and --shard-count, or neither.")
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count.")

    tagged_specs = [("train", spec) for spec in train_specs] + [
        ("test", spec) for spec in test_specs
    ]
    shard_specs = [
        (split, spec)
        for pos, (split, spec) in enumerate(tagged_specs)
        if pos % shard_count == shard_index
    ]
    return (
        [spec for split, spec in shard_specs if split == "train"],
        [spec for split, spec in shard_specs if split == "test"],
    )


def _shard_output_root(shards_root: Path, shard_index: int) -> Path:
    return shards_root / f"shard-{int(shard_index):02d}"


def _shard_reuse_roots(shards_root: Path) -> list[Path]:
    if not shards_root.exists():
        return []
    return sorted(path for path in shards_root.glob("shard-*") if path.is_dir())


def _write_frontier_summaries(
    output_root: Path,
    frontier_df: pd.DataFrame,
    ws_codebooks: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    summary_df = frontier_df.groupby(["split", "strategy"], as_index=False).agg(
        n_rows=("instance", "size"),
        min_T=("T", "min"),
        max_T=("T", "max"),
        best_response=("BestApproximationRatio", "max"),
    )
    preview_columns = [
        "split",
        "instance",
        "strategy",
        "p",
        "N",
        "M",
        "Q",
        "T",
        "selected_exact_T_proxy",
        "selected_exact_T",
        "budget_distance",
        "BestApproximationRatio",
        "best_found_value",
        "training_cost",
        "sampling_cost",
        "classical_setup_cost",
        "training_cost_proxy",
        "sampling_cost_proxy",
        "T_exact_proxy",
    ]
    preview_columns = [col for col in preview_columns if col in frontier_df.columns]
    preview_df = frontier_df.loc[:, preview_columns].head(30).copy()

    summary_path = output_root / "frontier_group_summary.csv"
    preview_path = output_root / "frontier_preview.csv"
    codebook_path = output_root / "ws_codebooks.json"

    summary_df.to_csv(summary_path, index=False)
    preview_df.to_csv(preview_path, index=False)
    codebook_path.write_text(_json_dumps_pretty(ws_codebooks), encoding="utf-8")

    return summary_df, preview_df, summary_path, preview_path, codebook_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare IBM QAOA FA/PT resource-frontier data for stochastic-benchmark. "
            "This is the script equivalent of the setup/precompute cells in "
            "Simulation_Method_Validation_and_WS.ipynb."
        )
    )
    parser.add_argument("--graph-type", default="heavy_hex")
    parser.add_argument("--num-nodes", type=int, default=144)
    parser.add_argument("--p-values", default="5", help="Comma-separated QAOA depths.")

    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--swap-layers", type=int, default=None)
    parser.add_argument("--er-probability", type=int, default=None)

    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=10)
    parser.add_argument("--start-train-index", type=int, default=100)
    parser.add_argument("--overwrite-train-instances", action="store_true")

    parser.add_argument("--fa-method-name", default="FA_PP_opt")
    parser.add_argument("--pt-method-name", default=None)
    parser.add_argument("--fa-n-values", default="10,20,50,75,100")
    parser.add_argument("--fa-m-values", default="10,50,100,200,500")
    parser.add_argument("--q-values", default="100,200,500,1000,5000")

    parser.add_argument("--time-per-shot", type=float, default=2.5e-6)
    parser.add_argument("--fa-cobyla-overhead-c", type=float, default=1.0)
    parser.add_argument("--pt-transfer-cost", type=float, default=0.0)

    parser.add_argument(
        "--t-grid-points",
        type=int,
        default=1000,
        help="Number of dense T-grid points when no explicit T grid is provided.",
    )
    parser.add_argument(
        "--t-grid-scale",
        choices=("linear", "log"),
        default="log",
        help="Spacing for the dense T grid.",
    )
    parser.add_argument("--t-grid-start", type=float, default=None)
    parser.add_argument("--t-grid-stop", type=float, default=None)
    parser.add_argument("--t-grid-step", type=float, default=None)

    parser.add_argument("--mps-chi", type=int, default=20)
    parser.add_argument("--max-parallel-threads", type=int, default=4)

    parser.add_argument(
        "--main-repo",
        default=os.environ.get("QAOA_PARAMETER_SETTING_ROOT", str(DEFAULT_MAIN_REPO)),
        help="Path to the QAOA-Parameter-Setting checkout.",
    )
    parser.add_argument(
        "--pipeline-repo",
        default=os.environ.get("QAOA_TRAINING_PIPELINE_ROOT", str(DEFAULT_PIPELINE_REPO)),
        help="Path to the qaoa_training_pipeline checkout.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Directory for campaign frontier outputs. "
            "Defaults to examples/IBM_QAOA/results/pss_window_sticker/<graph>_<num_nodes>."
        ),
    )
    parser.add_argument(
        "--instance-cache-root",
        default=os.environ.get("IBM_QAOA_INSTANCE_CACHE_ROOT", str(DEFAULT_INSTANCE_CACHE_ROOT)),
        help=(
            "Shared cache root for generated training instances and min/max cut files. "
            "A graph-set cache key is added below this root."
        ),
    )
    parser.add_argument(
        "--reuse-output-root",
        action="append",
        default=[],
        help=(
            "Optional existing campaign output directory to reuse exact-point pickles from. "
            "May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--no-reuse-current-output",
        action="store_true",
        help="Do not reuse fa_raw_points.pkl/pt_raw_points.pkl already present in the current output directory.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Resume from fa_raw_points.pkl/pt_raw_points.pkl already present in the current output directory. "
            "This is the explicit restart mode for interrupted runs."
        ),
    )
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Only generate/reuse exact FA/PT raw points. Used by Nautilus shard jobs.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only merge cached exact points and write final frontier outputs. Used after Nautilus shards finish.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based shard index for --exact-only runs.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="Total shard count for --exact-only runs.",
    )
    parser.add_argument(
        "--shards-root",
        default=None,
        help="Directory containing shard output directories. Defaults to <output-root>/shards.",
    )
    return parser


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root).expanduser().resolve()
    return (
        HERE
        / "results"
        / "pss_window_sticker"
        / f"{args.graph_type}_{args.num_nodes}"
    ).resolve()


def resolve_t_grid(args: argparse.Namespace) -> list[float] | None:
    explicit = (args.t_grid_start, args.t_grid_stop, args.t_grid_step)
    if all(value is None for value in explicit):
        return None
    if any(value is None for value in explicit):
        raise ValueError(
            "Provide all of --t-grid-start, --t-grid-stop, and --t-grid-step, "
            "or omit all three to use the dense automatic T grid."
        )
    return build_t_grid(args.t_grid_start, args.t_grid_stop, args.t_grid_step)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    ensure_qiskit_imports()
    args.fa_method_name = _normalize_optional_name(args.fa_method_name)
    args.pt_method_name = _normalize_optional_name(args.pt_method_name)

    p_values = _parse_int_list(args.p_values)
    fa_n_values = _parse_int_list(args.fa_n_values)
    fa_m_values = _parse_int_list(args.fa_m_values)
    q_values = _parse_int_list(args.q_values)

    main_repo = Path(args.main_repo).expanduser().resolve()
    pipeline_repo = Path(args.pipeline_repo).expanduser().resolve()
    output_root = resolve_output_root(args)
    instance_cache_root = Path(args.instance_cache_root).expanduser().resolve()
    shards_root = (
        Path(args.shards_root).expanduser().resolve()
        if args.shards_root is not None
        else output_root / "shards"
    )
    t_grid = resolve_t_grid(args)
    if args.restart and args.no_reuse_current_output:
        parser.error("--restart cannot be combined with --no-reuse-current-output.")
    if args.exact_only and args.finalize_only:
        parser.error("--exact-only cannot be combined with --finalize-only.")
    if (args.shard_index is not None or args.shard_count is not None) and not args.exact_only:
        parser.error("--shard-index and --shard-count are valid only with --exact-only.")
    if args.finalize_only and args.shards_root is None and not args.reuse_output_root:
        parser.error("--finalize-only requires --shards-root or at least one --reuse-output-root.")

    run_output_root = output_root
    if args.exact_only and args.shard_index is not None:
        run_output_root = _shard_output_root(shards_root, args.shard_index)

    reuse_roots = [Path(raw).expanduser().resolve() for raw in args.reuse_output_root]
    if args.exact_only and run_output_root != output_root:
        reuse_roots.append(output_root)
    if args.finalize_only:
        reuse_roots.extend(_shard_reuse_roots(shards_root))
    if args.restart or not args.no_reuse_current_output:
        reuse_roots.append(run_output_root if args.exact_only else output_root)
    reuse_roots = list(dict.fromkeys(reuse_roots))

    sample_config = {
        "chi": int(args.mps_chi),
        "max_parallel_threads": int(args.max_parallel_threads),
    }

    methods_dir = main_repo / "methods"
    method_names = [name for name in [args.fa_method_name, args.pt_method_name] if name is not None]
    method_configs = {name: methods_dir / f"{name}.json" for name in method_names}

    run_output_root.mkdir(parents=True, exist_ok=True)
    if not args.exact_only:
        output_root.mkdir(parents=True, exist_ok=True)

    print("Campaign configuration:")
    print(
        _json_dumps_pretty(
            {
                "graph_type": args.graph_type,
                "num_nodes": args.num_nodes,
                "p_values": p_values,
                "train_count": args.train_count,
                "test_count": args.test_count,
                "start_train_index": args.start_train_index,
                "fa_method_name": args.fa_method_name,
                "pt_method_name": args.pt_method_name,
                "fa_n_values": fa_n_values,
                "fa_m_values": fa_m_values,
                "q_values": q_values,
                "time_per_shot": args.time_per_shot,
                "fa_cobyla_overhead_c": args.fa_cobyla_overhead_c,
                "pt_transfer_cost": args.pt_transfer_cost,
                "t_grid_mode": "explicit" if t_grid is not None else "dense_auto",
                "t_grid_points": args.t_grid_points,
                "t_grid_scale": args.t_grid_scale,
                "sample_config": sample_config,
                "main_repo": str(main_repo),
                "pipeline_repo": str(pipeline_repo),
                "output_root": str(output_root),
                "run_output_root": str(run_output_root),
                "shards_root": str(shards_root),
                "exact_only": bool(args.exact_only),
                "finalize_only": bool(args.finalize_only),
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "instance_cache_root": str(instance_cache_root),
                "reuse_output_roots": [str(path) for path in reuse_roots],
                "restart": bool(args.restart),
            }
        )
    )

    train_specs, test_specs = build_train_test_instance_sets(
        args.graph_type,
        args.num_nodes,
        instance_cache_root=instance_cache_root,
        train_count=args.train_count,
        test_count=args.test_count,
        start_train_index=args.start_train_index,
        main_repo=main_repo,
        pipeline_repo=pipeline_repo,
        degree=args.degree,
        swap_layers=args.swap_layers,
        er_probability=args.er_probability,
        overwrite_train=args.overwrite_train_instances,
    )
    full_train_specs = train_specs
    full_test_specs = test_specs
    train_specs, test_specs = _shard_specs(
        full_train_specs,
        full_test_specs,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )

    train_specs_df = instance_specs_to_dataframe(train_specs)
    test_specs_df = instance_specs_to_dataframe(test_specs)
    all_specs_df = pd.concat([train_specs_df, test_specs_df], ignore_index=True)
    all_specs_path = run_output_root / "all_instance_specs.csv"
    all_specs_df.to_csv(all_specs_path, index=False)

    print(f"Training instances: {len(train_specs)}")
    print(f"Testing instances: {len(test_specs)}")
    print(f"Wrote {all_specs_path}")

    cached_exact_df = _load_exact_cache_frames(reuse_roots)
    if not cached_exact_df.empty:
        print(f"Loaded {len(cached_exact_df)} cached exact rows from {len(reuse_roots)} reuse root(s).")

    if args.finalize_only:
        if cached_exact_df.empty:
            raise RuntimeError("No cached exact rows were found for --finalize-only.")
        cached_exact_df = filter_pss_exact_points(
            cached_exact_df,
            train_specs=full_train_specs,
            test_specs=full_test_specs,
            p_values=p_values,
            fa_method_name=args.fa_method_name,
            pt_method_name=args.pt_method_name,
            fa_n_values=fa_n_values,
            fa_m_values=fa_m_values,
            q_values=q_values,
        )
        if cached_exact_df.empty:
            raise RuntimeError("No cached exact rows matched the requested campaign for --finalize-only.")
        frontier_df, ws_codebooks = build_pss_frontier_outputs(
            cached_exact_df,
            train_specs=full_train_specs,
            test_specs=full_test_specs,
            t_grid=t_grid,
            t_grid_points=args.t_grid_points,
            t_grid_scale=args.t_grid_scale,
            output_root=output_root,
            fa_method_name=args.fa_method_name,
            pt_method_name=args.pt_method_name,
        )
        summary_df, preview_df, summary_path, preview_path, codebook_path = _write_frontier_summaries(
            output_root,
            frontier_df,
            ws_codebooks,
        )
        print(f"Exact rows: {len(cached_exact_df)}")
        print(f"Frontier rows: {len(frontier_df)}")
        print(f"Dense T grid points: {frontier_df['T'].nunique()}")
        print(f"Wrote {summary_path}")
        print(f"Wrote {preview_path}")
        print(f"Wrote {codebook_path}")
        print()
        print("Frontier preview:")
        print(preview_df.to_string(index=False))
        print()
        print("Frontier summary:")
        print(summary_df.to_string(index=False))
        return 0

    if args.exact_only:
        exact_df = generate_pss_exact_points(
            train_specs=train_specs,
            test_specs=test_specs,
            p_values=p_values,
            method_configs=method_configs,
            main_repo=main_repo,
            pipeline_repo=pipeline_repo,
            fa_method_name=args.fa_method_name,
            pt_method_name=args.pt_method_name,
            fa_n_values=fa_n_values,
            fa_m_values=fa_m_values,
            q_values=q_values,
            sample_config=sample_config,
            time_per_shot=args.time_per_shot,
            cobyla_overhead_c=args.fa_cobyla_overhead_c,
            pt_transfer_cost=args.pt_transfer_cost,
            output_root=run_output_root,
            existing_exact_df=cached_exact_df if not cached_exact_df.empty else None,
        )
        print(f"Wrote exact raw-point checkpoints under {run_output_root}")
        print(f"Exact rows available to this shard/run: {len(exact_df)}")
        return 0

    exact_df, frontier_df, ws_codebooks = prepare_pss_exp_raw_dataset(
        train_specs=train_specs,
        test_specs=test_specs,
        p_values=p_values,
        t_grid=t_grid,
        t_grid_points=args.t_grid_points,
        t_grid_scale=args.t_grid_scale,
        method_configs=method_configs,
        main_repo=main_repo,
        pipeline_repo=pipeline_repo,
        fa_method_name=args.fa_method_name,
        pt_method_name=args.pt_method_name,
        fa_n_values=fa_n_values,
        fa_m_values=fa_m_values,
        q_values=q_values,
        sample_config=sample_config,
        time_per_shot=args.time_per_shot,
        cobyla_overhead_c=args.fa_cobyla_overhead_c,
        pt_transfer_cost=args.pt_transfer_cost,
        output_root=output_root,
        existing_exact_df=cached_exact_df if not cached_exact_df.empty else None,
    )

    summary_df, preview_df, summary_path, preview_path, codebook_path = _write_frontier_summaries(
        output_root,
        frontier_df,
        ws_codebooks,
    )

    print(f"Exact rows: {len(exact_df)}")
    print(f"Frontier rows: {len(frontier_df)}")
    print(f"Dense T grid points: {frontier_df['T'].nunique()}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {preview_path}")
    print(f"Wrote {codebook_path}")
    print()
    print("Frontier preview:")
    print(preview_df.to_string(index=False))
    print()
    print("Frontier summary:")
    print(summary_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

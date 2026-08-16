from __future__ import annotations

import copy
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd

from .Processing import expected_energy_from_counts, load_problem_instance
from .approx_ratio_calc import (
    best_prefix_metrics,
    extract_minmax_args,
    get_minmax,
    load_maxcut_instance_context,
    maxcut_approximation_ratio,
    maxcut_energy_from_bitstring,
)


DEFAULT_MAIN_REPO = Path(__file__).resolve().parents[4] / "QAOA-Parameter-Setting"
DEFAULT_PIPELINE_REPO = Path(__file__).resolve().parents[4] / "qaoa_training_pipeline"
DEFAULT_INSTANCE_CACHE_ROOT = Path(__file__).resolve().parents[1] / "data" / "generated_instances"
STRATEGY_RAW_POINTS_FILENAME = "strategy_raw_points.pkl"
TRANSFER_RAW_POINTS_FILENAME = "transfer_raw_points.pkl"
LEGACY_FA_RAW_POINTS_FILENAME = "fa_raw_points.pkl"
LEGACY_PT_RAW_POINTS_FILENAME = "pt_raw_points.pkl"

SAMPLED_BACKEND_METHODS = {
    "SV": "statevector",
    "MPSAer": "matrix_product_state",
    # Backward-compatible alias for old notebook state.
    "MPS": "matrix_product_state",
}

EVALUATOR_NAMES = {
    "SV": "StatevectorEvaluator",
    "MPSAer": "MPSAerEvaluator",
    "MPS": "MPSAerEvaluator",
    "PP": "PPEvaluator",
}

TRANSFER_DATABASE_FALLBACKS = {
    "optimized_angles_database_HH_cluster.json": "optimized_angles_database.json",
}

ZERO_TRAINING_METHODS = {"PT_PP_AAA", "FA_PP_no_opt", "linear_ramp_no_opt"}


@dataclass(frozen=True)
class InstanceSpec:
    graph_type: str
    num_nodes: int
    instance: int
    degree: int | None = None
    swap_layers: int | None = None
    er_probability: int | None = None
    graph_path: str | None = None
    minmax_path: str | None = None
    split: str | None = None
    source: str | None = None

    @property
    def instance_id(self) -> str:
        return str(self.instance).zfill(3)


def main_repo_path(main_repo: str | Path | None = None) -> Path:
    return Path(main_repo or os.environ.get("QAOA_PARAMETER_SETTING_ROOT", DEFAULT_MAIN_REPO)).expanduser()


def pipeline_repo_path(pipeline_repo: str | Path | None = None) -> Path:
    return Path(
        pipeline_repo or os.environ.get("QAOA_TRAINING_PIPELINE_ROOT", DEFAULT_PIPELINE_REPO)
    ).expanduser()


def instance_cache_root_path(instance_cache_root: str | Path | None = None) -> Path:
    return Path(
        instance_cache_root
        or os.environ.get("IBM_QAOA_INSTANCE_CACHE_ROOT", DEFAULT_INSTANCE_CACHE_ROOT)
    ).expanduser()


def estimate_hardware_time_per_shot(
    hardware_root: str | Path,
    *,
    graph_type: str = "heavy_hex",
    num_nodes: int | None = None,
    job_p: int | None = None,
    statistic: str = "median",
    normalize_by_pubs: bool = True,
) -> float:
    """Estimate effective hardware seconds per circuit/PUB shot from saved IBM results."""
    def _result_depths(result_path: Path, payload: list[dict]) -> set[int]:
        # Keep this local so notebook sessions holding a reloaded function do not
        # depend on a separately imported helper from an older module namespace.
        depths: set[int] = set()

        stem_parts = result_path.stem.split("_")
        for token in stem_parts[1:3]:
            if re.fullmatch(r"\d+", token):
                depths.add(int(token))

        for record in payload:
            if not isinstance(record, dict):
                continue
            for key in ("job_p", "p", "depth", "qaoa_depth", "reps"):
                value = record.get(key)
                try:
                    if value is not None:
                        depths.add(int(value))
                except (TypeError, ValueError):
                    pass

            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                continue
            for key in ("job_p", "p", "depth", "qaoa_depth", "reps"):
                value = metadata.get(key)
                try:
                    if value is not None:
                        depths.add(int(value))
                except (TypeError, ValueError):
                    pass
            method = str(metadata.get("method", ""))
            match = re.search(r"_(\d+)$", method)
            if match:
                depths.add(int(match.group(1)))
            result_file = str(metadata.get("result_file", ""))
            match = re.search(r"_(\d+)\.json$", result_file)
            if match:
                depths.add(int(match.group(1)))

        return depths

    root = Path(hardware_root).expanduser()
    search_root = root / graph_type if (root / graph_type).exists() else root
    if not search_root.exists():
        raise FileNotFoundError(f"Hardware results root does not exist: {search_root}")

    node_pattern = re.compile(rf"N{int(num_nodes)}(?!\d)") if num_nodes is not None else None
    job_p_value = int(job_p) if job_p is not None else None
    values: list[float] = []
    for result_path in sorted(search_root.rglob("*.json")):
        if node_pattern is not None and node_pattern.search(result_path.name) is None:
            continue
        try:
            payload = load_json(result_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
            continue
        if job_p_value is not None and job_p_value not in _result_depths(result_path, payload):
            continue

        summary = payload[0]
        if "total_time" not in summary or "num_shots" not in summary:
            continue
        try:
            total_time = float(summary["total_time"])
            num_shots = float(summary["num_shots"])
            num_pubs = float(summary.get("num_pubs", 1.0) or 1.0)
        except (TypeError, ValueError):
            continue

        denominator = num_shots * num_pubs if normalize_by_pubs else num_shots
        if total_time > 0 and denominator > 0:
            values.append(total_time / denominator)

    if not values:
        raise ValueError(
            "No hardware summary records with total_time and num_shots were found "
            f"under {search_root} for graph_type={graph_type!r}, num_nodes={num_nodes!r}, "
            f"job_p={job_p!r}."
        )

    statistic_key = str(statistic).lower()
    if statistic_key == "median":
        return float(statistics.median(values))
    if statistic_key == "mean":
        return float(statistics.mean(values))
    if statistic_key == "min":
        return float(min(values))
    if statistic_key == "max":
        return float(max(values))
    raise ValueError(f"Unsupported hardware time-per-shot statistic: {statistic!r}")


def generated_instance_cache_key(
    graph_type: str,
    num_nodes: int,
    *,
    degree: int | None = None,
    swap_layers: int | None = None,
    er_probability: int | None = None,
    start_train_index: int = 100,
    train_count: int = 20,
) -> str:
    def _part(name: str, value: object) -> str:
        value_text = "none" if value is None else str(value)
        value_text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", value_text)
        return f"{name}={value_text}"

    parts = [
        graph_type,
        _part("n", int(num_nodes)),
        _part("degree", degree),
        _part("swap_layers", swap_layers),
        _part("er_probability", er_probability),
        _part("start", int(start_train_index)),
        _part("count", int(train_count)),
    ]
    return "__".join(parts)


def generated_instance_set_root(
    instance_cache_root: str | Path | None,
    graph_type: str,
    num_nodes: int,
    *,
    degree: int | None = None,
    swap_layers: int | None = None,
    er_probability: int | None = None,
    start_train_index: int = 100,
    train_count: int = 20,
) -> Path:
    return instance_cache_root_path(instance_cache_root) / generated_instance_cache_key(
        graph_type,
        num_nodes,
        degree=degree,
        swap_layers=swap_layers,
        er_probability=er_probability,
        start_train_index=start_train_index,
        train_count=train_count,
    )


def ensure_pipeline_imports(pipeline_repo: str | Path | None = None) -> dict[str, Any]:
    repo = pipeline_repo_path(pipeline_repo)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    try:
        from qaoa_training_pipeline.evaluation import EVALUATORS
        from qaoa_training_pipeline.training import TRAINERS
        from qaoa_training_pipeline.utils.data_utils import input_to_operator, load_input
    except ImportError as exc:
        raise ImportError(
            "qaoa_training_pipeline and its dependencies are required for this notebook. "
            "Set QAOA_TRAINING_PIPELINE_ROOT and install qiskit/qiskit-aer in the active env."
        ) from exc

    return {
        "TRAINERS": TRAINERS,
        "EVALUATORS": EVALUATORS,
        "input_to_operator": input_to_operator,
        "load_input": load_input,
    }


def ensure_qiskit_imports() -> dict[str, Any]:
    try:
        from qiskit.circuit.library import qaoa_ansatz
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise ImportError(
            "Sampling requires qiskit and qiskit-aer in the active environment."
        ) from exc

    return {
        "qaoa_ansatz": qaoa_ansatz,
        "SparsePauliOp": SparsePauliOp,
        "AerSimulator": AerSimulator,
    }


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_method_config(config_path: str | Path) -> dict[str, Any]:
    return load_json(config_path)


def resolve_config_file_reference(
    file_name: str,
    config_path: str | Path | None = None,
    main_repo: str | Path | None = None,
    allow_transfer_fallback: bool = False,
) -> str:
    raw_path = Path(file_name).expanduser()
    if raw_path.is_absolute() and raw_path.exists():
        return str(raw_path)

    repo = main_repo_path(main_repo)
    candidates: list[Path] = []

    if config_path is not None:
        config_dir = Path(config_path).expanduser().resolve().parent
        candidates.append((config_dir / raw_path).resolve())

    candidates.append((repo / raw_path).resolve())
    candidates.append((repo / "data" / "training" / "parameter_transfer" / raw_path.name).resolve())

    if allow_transfer_fallback and raw_path.name in TRANSFER_DATABASE_FALLBACKS:
        candidates.append(
            (
                repo
                / "data"
                / "training"
                / "parameter_transfer"
                / TRANSFER_DATABASE_FALLBACKS[raw_path.name]
            ).resolve()
        )

    if allow_transfer_fallback and "optimized_angles_database" in raw_path.name:
        candidates.append(
            (
                repo / "data" / "training" / "parameter_transfer" / "optimized_angles_database.json"
            ).resolve()
        )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return str(candidate)

    return str(raw_path)


def normalize_method_config(
    config: dict[str, Any],
    config_path: str | Path | None = None,
    main_repo: str | Path | None = None,
    allow_transfer_fallback: bool = False,
) -> dict[str, Any]:
    normalized = copy.deepcopy(config)

    for stage in normalized.get("trainer_chain", []):
        trainer_init = stage.get("trainer_init", {})
        data_loader_init = trainer_init.get("data_loader_init")
        if isinstance(data_loader_init, dict) and "file_name" in data_loader_init:
            requested_name = str(data_loader_init["file_name"])
            resolved = resolve_config_file_reference(
                requested_name,
                config_path=config_path,
                main_repo=main_repo,
                allow_transfer_fallback=allow_transfer_fallback,
            )
            data_loader_init["file_name"] = resolved

            if "optimized_angles_database" in requested_name and not Path(resolved).exists():
                raise FileNotFoundError(
                    "Required parameter-transfer database is missing. "
                    f"Requested '{requested_name}' while normalizing this method config."
                )

    return normalized


def infer_fixed_angle_degree(
    *,
    spec: InstanceSpec | None = None,
    graph_path: str | Path | None = None,
) -> int | None:
    if spec is not None and spec.degree is not None:
        return int(spec.degree)

    if graph_path is None:
        return None

    stem = Path(graph_path).stem.lower()
    heavy_hex_match = re.match(r"^\d+_\d+_(\d+)_heavyhex_", stem)
    if heavy_hex_match:
        return int(heavy_hex_match.group(1))

    return None


def native_train_kwargs_overrides(
    config: dict[str, Any],
    *,
    reps: int,
    spec: InstanceSpec | None = None,
    graph_path: str | Path | None = None,
) -> dict[int, dict[str, Any]]:
    trainer_chain = config.get("trainer_chain", [])
    if not trainer_chain:
        return {}

    overrides: dict[int, dict[str, Any]] = {}
    recursive_indices = [
        idx for idx, stage in enumerate(trainer_chain) if stage.get("trainer") == "RecursionTrainer"
    ]
    if recursive_indices:
        overrides[recursive_indices[-1]] = {"reps": int(reps)}
    else:
        primary_trainer = trainer_chain[0].get("trainer")
        if primary_trainer == "TransferTrainer":
            overrides[0] = {"qaoa_depth": int(reps)}
        else:
            overrides[0] = {"reps": int(reps)}

    if trainer_chain[0].get("trainer") == "FixedAngleConjecture":
        degree_hint = infer_fixed_angle_degree(spec=spec, graph_path=graph_path)
        if degree_hint is not None:
            overrides.setdefault(0, {})
            overrides[0]["degree"] = int(degree_hint)

    return overrides


def latest_stage_data(result_bundle: dict[str | int, Any]) -> dict[str, Any]:
    stage_keys = sorted(key for key in result_bundle if isinstance(key, int))
    if not stage_keys:
        return result_bundle
    return result_bundle[stage_keys[-1]]


def locate_instance_file(spec: InstanceSpec, main_repo: str | Path | None = None) -> Path:
    if spec.graph_path is not None:
        return Path(spec.graph_path)

    repo = main_repo_path(main_repo)
    matches = load_problem_instance(
        str(repo / "instances"),
        spec.graph_type,
        str(spec.num_nodes),
        spec.instance_id,
        degree=str(spec.degree) if spec.degree is not None else None,
        ER_probability=str(spec.er_probability) if spec.er_probability is not None else None,
        swap_layers=str(spec.swap_layers) if spec.swap_layers is not None else None,
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one instance file for {spec}, found {len(matches)}")
    return matches[0]


def load_instance_context(spec: InstanceSpec, main_repo: str | Path | None = None) -> dict[str, Any]:
    repo = main_repo_path(main_repo)
    graph_path = locate_instance_file(spec, repo)
    if spec.minmax_path is not None:
        minmax_file = Path(spec.minmax_path)
    else:
        minmax_file = get_minmax(
            str(repo / "data" / "minmax_cuts"),
            spec.graph_type,
            spec.instance_id,
            str(spec.num_nodes),
            ER_probability=str(spec.er_probability) if spec.er_probability is not None else None,
            swap_layers=str(spec.swap_layers) if spec.swap_layers is not None else None,
            degree=str(spec.degree) if spec.degree is not None else None,
        )
    min_cut, max_cut, sum_weights = extract_minmax_args(minmax_file)
    instance_context = load_maxcut_instance_context(graph_path)
    return {
        "graph_path": graph_path,
        "minmax_path": Path(minmax_file),
        "instance_context": instance_context,
        "min_cut": min_cut,
        "max_cut": max_cut,
        "sum_weights": sum_weights,
        "short_name": graph_path.stem,
    }


def instance_specs_to_dataframe(specs: list[InstanceSpec]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "graph_type": spec.graph_type,
                "num_nodes": spec.num_nodes,
                "instance": spec.instance_id,
                "degree": spec.degree,
                "swap_layers": spec.swap_layers,
                "er_probability": spec.er_probability,
                "graph_path": spec.graph_path,
                "minmax_path": spec.minmax_path,
                "split": spec.split,
                "source": spec.source,
            }
            for spec in specs
        ]
    )


def _edge_weight(data: dict[str, Any]) -> float:
    return float(data.get("weight", 1.0))


def _graph_to_instance_payload(graph: nx.Graph, description: str) -> dict[str, Any]:
    edge_list = [
        {"nodes": [int(u), int(v)], "weight": _edge_weight(data)}
        for u, v, data in graph.edges(data=True)
    ]
    return {"edge list": edge_list, "Description": description}


def _write_graph_instance(graph: nx.Graph, description: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_graph_to_instance_payload(graph, description), handle, indent=4)
    return output_path


def _compute_and_write_minmax(
    graph_path: str | Path,
    minmax_path: str | Path,
    *,
    pipeline_repo: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    minmax_path = Path(minmax_path)
    if minmax_path.exists() and not overwrite:
        return minmax_path

    repo = pipeline_repo_path(pipeline_repo)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from qaoa_training_pipeline.utils.graph_utils import (  # pylint: disable=import-outside-toplevel
        graph_to_operator,
        load_graph,
        operator_to_graph,
        solve_max_cut,
    )

    graph = load_graph(Path(graph_path))
    cost_op = graph_to_operator(graph, pre_factor=-0.5)
    time_start = time.time_ns()
    max_cut, min_cut, _ = solve_max_cut(cost_op)
    time_end = time.time_ns()

    weighted_graph = operator_to_graph(cost_op, pre_factor=-2)
    sum_weights = float(sum(data.get("weight", 1.0) for _, _, data in weighted_graph.edges(data=True)))

    minmax_path.parent.mkdir(parents=True, exist_ok=True)
    with minmax_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "instance": Path(graph_path).name,
                "max_cut": float(max_cut),
                "min_cut": float(min_cut),
                "sum_of_weights": sum_weights,
                "time_solve_max_cut_ns": int(time_end - time_start),
            },
            handle,
        )

    return minmax_path


def infer_heavy_hex_lattice_shape(
    num_nodes: int,
    *,
    main_repo: str | Path | None = None,
) -> tuple[int, int]:
    repo = main_repo_path(main_repo)
    matches = sorted((repo / "instances" / "heavy_hex").glob(f"*_*_*_heavyhex_{int(num_nodes)}nodes*.json"))
    if not matches:
        raise FileNotFoundError(
            f"Could not infer heavy-hex rows/cols for {num_nodes} nodes from {repo / 'instances' / 'heavy_hex'}."
        )

    match = re.match(r"^\d+_(\d+)_(\d+)_heavyhex_", matches[0].stem)
    if match is None:
        raise ValueError(f"Could not parse heavy-hex rows/cols from {matches[0].name}.")
    return int(match.group(1)), int(match.group(2))


def _make_heavy_hex_graph(rows: int, cols: int, seed: int) -> nx.Graph:
    base_graph = nx.hexagonal_lattice_graph(cols, rows)
    graph = nx.Graph()
    node_map = {node: idx for idx, node in enumerate(base_graph.nodes())}
    rng = np.random.default_rng(seed=seed)

    for edge_idx, edge in enumerate(base_graph.edges()):
        u = node_map[edge[0]]
        v = node_map[edge[1]]
        ancilla = len(base_graph.nodes()) + edge_idx
        graph.add_edge(u, ancilla, weight=float(rng.normal()))
        graph.add_edge(ancilla, v, weight=float(rng.normal()))

    return graph


def build_test_instance_set_from_repo(
    graph_type: str,
    num_nodes: int,
    *,
    count: int = 10,
    main_repo: str | Path | None = None,
    degree: int | None = None,
    swap_layers: int | None = None,
    er_probability: int | None = None,
) -> list[InstanceSpec]:
    repo = main_repo_path(main_repo)
    specs: list[InstanceSpec] = []

    for instance in range(int(count)):
        spec = InstanceSpec(
            graph_type=graph_type,
            num_nodes=int(num_nodes),
            instance=int(instance),
            degree=degree,
            swap_layers=swap_layers,
            er_probability=er_probability,
        )
        graph_path = locate_instance_file(spec, repo)
        minmax_path = get_minmax(
            str(repo / "data" / "minmax_cuts"),
            graph_type,
            spec.instance_id,
            str(num_nodes),
            ER_probability=str(er_probability) if er_probability is not None else None,
            swap_layers=str(swap_layers) if swap_layers is not None else None,
            degree=str(degree) if degree is not None else None,
        )
        specs.append(
            InstanceSpec(
                graph_type=graph_type,
                num_nodes=int(num_nodes),
                instance=int(instance),
                degree=degree,
                swap_layers=swap_layers,
                er_probability=er_probability,
                graph_path=str(graph_path),
                minmax_path=str(minmax_path),
                split="test",
                source="repo_test",
            )
        )

    return specs


def generate_training_instances_like_qps(
    graph_type: str,
    num_nodes: int,
    *,
    output_root: str | Path,
    count: int = 20,
    start_index: int = 100,
    main_repo: str | Path | None = None,
    pipeline_repo: str | Path | None = None,
    degree: int | None = None,
    swap_layers: int | None = None,
    er_probability: int | None = None,
    overwrite: bool = False,
) -> list[InstanceSpec]:
    output_root = Path(output_root)
    instances_dir = output_root / "instances" / graph_type
    minmax_dir = output_root / "data" / "minmax_cuts" / graph_type

    if graph_type == "heavy_hex":
        rows, cols = infer_heavy_hex_lattice_shape(num_nodes, main_repo=main_repo)

    generated_specs: list[InstanceSpec] = []
    for offset in range(int(count)):
        instance_number = int(start_index) + offset
        idx_str = str(instance_number).zfill(3)

        if graph_type == "random_regular":
            if degree is None:
                raise ValueError("degree must be provided when graph_type='random_regular'.")
            graph = nx.random_regular_graph(n=int(num_nodes), d=int(degree), seed=instance_number)
            description = (
                f"Random {int(degree)}-regular graph without weights. "
                f"Generated by nx.random_regular_graph(n={int(num_nodes)}, d={int(degree)}, seed={instance_number})."
            )
            file_name = f"{idx_str}_{int(num_nodes)}nodes_random{int(degree)}regular.json"
        elif graph_type == "erdos_renyi":
            if er_probability is None:
                raise ValueError("er_probability must be provided when graph_type='erdos_renyi'.")
            prob = float(er_probability) / 100.0 if float(er_probability) > 1 else float(er_probability)
            seed = instance_number
            while True:
                candidate = nx.erdos_renyi_graph(n=int(num_nodes), p=prob, seed=seed)
                if nx.is_connected(candidate):
                    graph = candidate
                    break
                seed += 1
            description = (
                f"Erdos-Renyi graph without weights and {int(round(prob * 100))}% edge probability. "
                f"Generated by nx.erdos_renyi_graph(n={int(num_nodes)}, p={prob}, seed={seed}) and filtered for connectivity."
            )
            file_name = f"{idx_str}_{int(num_nodes)}nodes_erdosrenyi{int(round(prob * 100))}percent.json"
        elif graph_type == "line_to_full":
            if swap_layers is None:
                raise ValueError("swap_layers must be provided when graph_type='line_to_full'.")
            from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (  # pylint: disable=import-outside-toplevel
                SwapStrategy,
            )

            rng = np.random.default_rng(seed=instance_number)
            strategy = SwapStrategy.from_line(range(int(num_nodes)))
            adjacency = np.array(np.ones((int(num_nodes), int(num_nodes))) * strategy.distance_matrix <= int(swap_layers), dtype=int)
            adjacency = adjacency - np.diag([1] * int(num_nodes))
            signs = rng.choice([1, -1], (int(num_nodes), int(num_nodes)))
            signs = np.triu(signs) + np.triu(signs, 1).T
            adjacency = adjacency * signs
            graph = nx.from_numpy_array(adjacency)
            description = (
                f"Graph generated from a line of nodes and {int(swap_layers)} layers of swap gates. "
                f"Each edge has weight +/-1 generated by np.random.default_rng(seed={instance_number})."
            )
            file_name = f"{idx_str}_{int(num_nodes)}nodes_{int(swap_layers)}swap_layers.json"
        elif graph_type == "heavy_hex":
            graph = _make_heavy_hex_graph(rows=rows, cols=cols, seed=instance_number)
            description = (
                f"Heavy-hex graph generated from nx.hexagonal_lattice_graph({cols}, {rows}) "
                f"and promoted to heavy-hex with normal edge weights, seed={instance_number}."
            )
            file_name = f"{idx_str}_{rows}_{cols}_heavyhex_{graph.order()}nodes_weighted.json"
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")

        graph_path = instances_dir / file_name
        minmax_path = minmax_dir / f"{graph_path.stem}_maxmin_cut.json"

        if overwrite or not graph_path.exists():
            _write_graph_instance(graph, description, graph_path)
        _compute_and_write_minmax(
            graph_path,
            minmax_path,
            pipeline_repo=pipeline_repo,
            overwrite=overwrite,
        )

        generated_specs.append(
            InstanceSpec(
                graph_type=graph_type,
                num_nodes=int(num_nodes),
                instance=int(instance_number),
                degree=degree,
                swap_layers=swap_layers,
                er_probability=er_probability,
                graph_path=str(graph_path),
                minmax_path=str(minmax_path),
                split="train",
                source="generated_train",
            )
        )

    return generated_specs


def build_train_test_instance_sets(
    graph_type: str,
    num_nodes: int,
    *,
    generated_root: str | Path | None = None,
    instance_cache_root: str | Path | None = None,
    train_count: int = 20,
    test_count: int = 10,
    start_train_index: int = 100,
    main_repo: str | Path | None = None,
    pipeline_repo: str | Path | None = None,
    degree: int | None = None,
    swap_layers: int | None = None,
    er_probability: int | None = None,
    overwrite_train: bool = False,
) -> tuple[list[InstanceSpec], list[InstanceSpec]]:
    if generated_root is not None and instance_cache_root is not None:
        raise ValueError("Provide only one of generated_root or instance_cache_root.")
    cache_root = generated_root if generated_root is not None else instance_cache_root
    generated_set_root = generated_instance_set_root(
        cache_root,
        graph_type,
        num_nodes,
        degree=degree,
        swap_layers=swap_layers,
        er_probability=er_probability,
        start_train_index=start_train_index,
        train_count=train_count,
    )
    train_specs = generate_training_instances_like_qps(
        graph_type,
        num_nodes,
        output_root=generated_set_root,
        count=train_count,
        start_index=start_train_index,
        main_repo=main_repo,
        pipeline_repo=pipeline_repo,
        degree=degree,
        swap_layers=swap_layers,
        er_probability=er_probability,
        overwrite=overwrite_train,
    )
    test_specs = build_test_instance_set_from_repo(
        graph_type,
        num_nodes,
        count=test_count,
        main_repo=main_repo,
        degree=degree,
        swap_layers=swap_layers,
        er_probability=er_probability,
    )
    return train_specs, test_specs


def build_t_grid(
    start: float,
    stop: float,
    step: float,
) -> list[float]:
    start = float(start)
    stop = float(stop)
    step = float(step)
    if step <= 0:
        raise ValueError("step must be positive when building a T grid.")
    return [float(val) for val in np.arange(start, stop + 0.5 * step, step)]


def build_cost_operator_from_graph(
    graph_path: str | Path,
    pipeline_repo: str | Path | None = None,
    pre_factor: float = -0.5,
):
    imports = ensure_pipeline_imports(pipeline_repo)
    return imports["input_to_operator"](imports["load_input"](str(graph_path)), pre_factor=pre_factor)


def build_cost_operator_from_serialized(serialized_cost_operator: Any):
    qiskit_imports = ensure_qiskit_imports()
    SparsePauliOp = qiskit_imports["SparsePauliOp"]

    if isinstance(serialized_cost_operator, SparsePauliOp):
        return serialized_cost_operator
    if isinstance(serialized_cost_operator, list):
        return SparsePauliOp.from_list(
            [(str(pauli), complex(coeff)) for pauli, coeff in serialized_cost_operator]
        )
    raise TypeError("Unsupported serialized cost operator format.")


def generate_linear_ramp_angles(
    cost_op,
    reps: int,
    slopes: tuple[float, float] = (0.5, 0.5),
    pipeline_repo: str | Path | None = None,
) -> dict[str, Any]:
    imports = ensure_pipeline_imports(pipeline_repo)
    trainer_cls = imports["TRAINERS"]["TQATrainer"]
    trainer = trainer_cls(evaluator=None, initial_dt=list(slopes))
    return trainer.train(cost_op=cost_op, reps=reps).data


def _saved_stage_metadata(stage: dict[str, Any]) -> dict[str, Any]:
    augmented = copy.deepcopy(stage)
    if "num_objective_evaluations" not in augmented:
        energy_history = augmented.get("energy_history") or []
        parameter_history = augmented.get("parameter_history") or []
        if energy_history:
            augmented["num_objective_evaluations"] = int(len(energy_history))
        elif parameter_history:
            augmented["num_objective_evaluations"] = int(len(parameter_history))
        elif augmented.get("energy") not in (None, "NA"):
            augmented["num_objective_evaluations"] = 1
        else:
            augmented["num_objective_evaluations"] = 0

    augmented.setdefault("shots_per_objective", None)
    augmented.setdefault("training_shots_used", 0)
    return augmented


def load_saved_training_result(
    spec: InstanceSpec,
    method_name: str,
    reps: int,
    *,
    main_repo: str | Path | None = None,
) -> tuple[dict[str | int, Any], Path]:
    repo = main_repo_path(main_repo)
    training_dir = repo / "data" / "training" / spec.graph_type
    if not training_dir.exists():
        raise FileNotFoundError(f"No saved training directory exists for graph family '{spec.graph_type}'.")

    pattern = f"*{spec.instance_id}N{spec.num_nodes}*_{method_name}*_{int(reps)}.json"
    matches = sorted(training_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No saved training result matched pattern '{pattern}' in {training_dir}."
        )

    result_path = matches[-1]
    payload = load_json(result_path)
    result_bundle: dict[str | int, Any] = {}
    for key, value in payload.items():
        if isinstance(key, str) and key.isdigit():
            result_bundle[int(key)] = _saved_stage_metadata(value)
        else:
            result_bundle[key] = value
    return result_bundle, result_path


def register_local_pipeline_evaluators(pipeline_repo: str | Path | None = None) -> dict[str, Any]:
    imports = ensure_pipeline_imports(pipeline_repo)
    imports["EVALUATORS"]["MPSAerSampleEvaluator"] = MPSAerSampleEvaluator
    return imports


def _prepare_config_object(
    config_path_or_dict: str | Path | dict[str, Any],
    *,
    main_repo: str | Path | None = None,
) -> tuple[dict[str, Any], str | Path | None]:
    if isinstance(config_path_or_dict, dict):
        return copy.deepcopy(config_path_or_dict), None
    return load_method_config(config_path_or_dict), config_path_or_dict


def _infer_objective_evaluation_count(trainer, stage_result: dict[str, Any]) -> int:
    if hasattr(trainer, "energy_history") and getattr(trainer, "energy_history"):
        return int(len(trainer.energy_history))

    evaluator = getattr(trainer, "_evaluator", None)
    if evaluator is not None and hasattr(evaluator, "num_evaluations"):
        return int(evaluator.num_evaluations)

    if evaluator is not None and stage_result.get("energy") not in (None, "NA"):
        return 1

    return 0


def _augment_stage_result(trainer, stage_result: dict[str, Any]) -> dict[str, Any]:
    augmented = copy.deepcopy(stage_result)
    evaluator = getattr(trainer, "_evaluator", None)

    num_objective_evaluations = _infer_objective_evaluation_count(trainer, augmented)
    augmented["num_objective_evaluations"] = num_objective_evaluations

    shots_per_objective = None
    if evaluator is not None and hasattr(evaluator, "shots"):
        shots_per_objective = int(evaluator.shots)
    augmented["shots_per_objective"] = shots_per_objective
    augmented["training_shots_used"] = (
        int(num_objective_evaluations * shots_per_objective)
        if shots_per_objective is not None
        else 0
    )

    if evaluator is not None and hasattr(evaluator, "get_results_from_last_iteration"):
        last_iteration = evaluator.get_results_from_last_iteration() or {}
        for key, value in last_iteration.items():
            augmented.setdefault(key, value)

    return augmented


def run_method_from_config(
    cost_op,
    config_path: str | Path | dict[str, Any],
    pipeline_repo: str | Path | None = None,
    main_repo: str | Path | None = None,
    train_kwargs_overrides: dict[int, dict[str, Any]] | None = None,
    disable_transfer_evaluator: bool = False,
    config_mutator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    allow_transfer_fallback: bool = False,
) -> dict[str | int, Any]:
    imports = register_local_pipeline_evaluators(pipeline_repo)
    raw_config, config_path_ref = _prepare_config_object(config_path, main_repo=main_repo)
    config = normalize_method_config(
        raw_config,
        config_path=config_path_ref,
        main_repo=main_repo,
        allow_transfer_fallback=allow_transfer_fallback,
    )
    if config_mutator is not None:
        config = config_mutator(copy.deepcopy(config))

    latest_result: dict[str, Any] = {}
    results: dict[str | int, Any] = {}

    for idx, stage in enumerate(config["trainer_chain"]):
        trainer_name = stage["trainer"]
        trainer_init = copy.deepcopy(stage.get("trainer_init", {}))

        if disable_transfer_evaluator and trainer_name == "TransferTrainer":
            trainer_init["evaluator"] = None
            trainer_init["evaluator_init"] = {}

        trainer_cls = imports["TRAINERS"][trainer_name]
        trainer = trainer_cls.from_config(trainer_init)

        train_kwargs = copy.deepcopy(stage.get("train_kwargs", {}))
        if latest_result and "result" in train_kwargs:
            if not hasattr(latest_result, "__getitem__"):
                raise TypeError(
                    f"Stage {idx} ({trainer_name}) chains 'result' keys from the previous "
                    f"stage's output, but that output is a {type(latest_result).__name__}, "
                    "which does not support subscripting. Expected a dict-like ParamResult."
                )
            for result_key, arg_name in train_kwargs["result"].items():
                train_kwargs[arg_name] = latest_result[result_key]
            train_kwargs.pop("result")

        if train_kwargs_overrides and idx in train_kwargs_overrides:
            train_kwargs.update(train_kwargs_overrides[idx])

        if trainer_name == "TransferTrainer" and "reps" in train_kwargs and "qaoa_depth" not in train_kwargs:
            train_kwargs["qaoa_depth"] = train_kwargs.pop("reps")

        latest = trainer.train(cost_op, **train_kwargs)
        latest_result = latest
        results[idx] = _augment_stage_result(trainer, latest.data)

    results["config"] = config
    return results


def generate_angle_payload(
    cost_op,
    angle_method: str,
    reps: int,
    *,
    method_configs: dict[str, Path] | None = None,
    main_repo: str | Path | None = None,
    pipeline_repo: str | Path | None = None,
    linear_ramp_slopes: tuple[float, float] = (0.5, 0.5),
    disable_transfer_evaluator: bool = True,
    spec: InstanceSpec | None = None,
    graph_path: str | Path | None = None,
) -> dict[str, Any]:
    if angle_method == "linear_ramp_no_opt":
        stage = generate_linear_ramp_angles(
            cost_op,
            reps=reps,
            slopes=linear_ramp_slopes,
            pipeline_repo=pipeline_repo,
        )
        return {
            "angle_method": angle_method,
            "angles": stage["optimized_qaoa_angles"],
            "raw_result": stage,
            "result_bundle": {0: stage},
        }

    if "_PP" in angle_method and spec is not None:
        result_bundle, saved_result_path = load_saved_training_result(
            spec,
            angle_method,
            reps,
            main_repo=main_repo,
        )
        stage = latest_stage_data(result_bundle)
        return {
            "angle_method": angle_method,
            "angles": stage["optimized_qaoa_angles"],
            "raw_result": stage,
            "result_bundle": result_bundle,
            "saved_result_path": str(saved_result_path),
            "saved_cost_operator": result_bundle.get("cost_operator"),
        }

    if method_configs is None or angle_method not in method_configs:
        raise ValueError(f"Unknown angle method {angle_method}.")

    method_config = load_method_config(method_configs[angle_method])
    result_bundle = run_method_from_config(
        cost_op,
        method_config,
        pipeline_repo=pipeline_repo,
        main_repo=main_repo,
        train_kwargs_overrides=native_train_kwargs_overrides(
            method_config,
            reps=reps,
            spec=spec,
            graph_path=graph_path,
        ),
        disable_transfer_evaluator=disable_transfer_evaluator,
    )
    stage = latest_stage_data(result_bundle)
    return {
        "angle_method": angle_method,
        "angles": stage["optimized_qaoa_angles"],
        "raw_result": stage,
        "result_bundle": result_bundle,
    }


def build_bound_qaoa_circuit(cost_op, params: list[float], mixer=None, initial_state=None, ansatz_circuit=None):
    qiskit_imports = ensure_qiskit_imports()
    qaoa_ansatz = qiskit_imports["qaoa_ansatz"]
    SparsePauliOp = qiskit_imports["SparsePauliOp"]

    if isinstance(ansatz_circuit, SparsePauliOp):
        ansatz_op = ansatz_circuit
    elif ansatz_circuit is None:
        ansatz_op = cost_op
    else:
        raise NotImplementedError("Custom ansatz circuits are not supported in this notebook helper.")

    circuit = qaoa_ansatz(
        ansatz_op,
        reps=len(params) // 2,
        mixer_operator=mixer,
        initial_state=initial_state,
    )
    if len(circuit.parameters) != len(params):
        raise ValueError("The QAOA circuit parameter count does not match the provided angles.")
    return circuit.assign_parameters(dict(zip(circuit.parameters, params)), inplace=False)


class MPSAerSampleEvaluator:
    """Generate samples with Aer MPS and expose counts like an evaluator."""

    def __init__(self, chi: int | None = None, max_parallel_threads: int | None = None, shots: int | None = None):
        qiskit_imports = ensure_qiskit_imports()
        try:
            from qiskit.primitives import BackendSamplerV2
        except ImportError as exc:
            raise ImportError(
                "MPSAer sample generation requires qiskit.primitives.BackendSamplerV2."
            ) from exc

        self._cost_op = None
        self._counts: dict[str, int] = {}
        self._term_indices: list[tuple[int, ...]] = []
        self._term_coeffs: list[float] = []
        self.num_evaluations = 0

        self.chi = chi or 20
        self.max_parallel_threads = max_parallel_threads or 10
        self.shots = shots or 1000

        self._backend = qiskit_imports["AerSimulator"](
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=self.chi,
            max_parallel_threads=self.max_parallel_threads,
        )
        if hasattr(self._backend.options, "use_fractional_gates"):
            self._backend.options.use_fractional_gates = False
        self._sampler = BackendSamplerV2(backend=self._backend)

    @property
    def cost_op(self):
        return self._cost_op

    @cost_op.setter
    def cost_op(self, cost_op):
        self._cost_op = cost_op
        self._term_indices = []
        self._term_coeffs = []
        for pauli in cost_op:
            indices = tuple(idx for idx, value in enumerate(pauli.paulis[0].z) if value)
            self._term_indices.append(indices)
            self._term_coeffs.append(float(np.real(pauli.coeffs[0])))

    def energy(self, bitstring: str) -> float:
        sample = [value == "1" for value in bitstring[::-1]]
        energy = 0.0
        for indices, coeff in zip(self._term_indices, self._term_coeffs):
            if len(indices) == 1:
                energy += -coeff if sample[indices[0]] else coeff
            elif len(indices) == 2:
                energy += coeff if sample[indices[0]] == sample[indices[1]] else -coeff
        return float(energy)

    def total_energy(self, counts: dict[str, int]) -> float:
        total_shots = sum(counts.values())
        if total_shots == 0:
            return float("nan")
        total_energy = 0.0
        for bitstring, count in counts.items():
            total_energy += self.energy(bitstring) * count / total_shots
        return float(total_energy)

    def evaluate(self, cost_op, params, mixer=None, initial_state=None, ansatz_circuit=None) -> float:
        qiskit_imports = ensure_qiskit_imports()
        qaoa_ansatz = qiskit_imports["qaoa_ansatz"]
        SparsePauliOp = qiskit_imports["SparsePauliOp"]

        if self._cost_op is None:
            self.cost_op = cost_op

        if isinstance(ansatz_circuit, SparsePauliOp):
            ansatz_op = ansatz_circuit
        elif ansatz_circuit is None:
            ansatz_op = cost_op
        else:
            raise NotImplementedError("Custom ansatz circuits are not supported in this notebook helper.")

        circuit = qaoa_ansatz(
            ansatz_op,
            reps=len(params) // 2,
            mixer_operator=mixer,
            initial_state=initial_state,
        ).decompose()
        circuit.measure_all()
        result = self._sampler.run([(circuit, params, self.shots)]).result()
        self.num_evaluations += 1

        pub_result = result[0]
        for register_name in ("meas", "c"):
            register = getattr(pub_result.data, register_name, None)
            if register is not None and hasattr(register, "get_counts"):
                counts = register.get_counts()
                self._counts = {str(bitstring): int(count) for bitstring, count in counts.items()}
                break
        else:
            raise RuntimeError("Could not extract counts from BackendSamplerV2 result.")

        return self.total_energy(self._counts)

    def get_results_from_last_iteration(self) -> dict[str, dict[str, int]]:
        return {"counts": self._counts}

    def to_config(self) -> dict[str, Any]:
        return {
            "chi": self.chi,
            "max_parallel_threads": self.max_parallel_threads,
            "shots": self.shots,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MPSAerSampleEvaluator":
        return cls(**(config or {}))


def sample_bound_circuit_counts(
    circuit,
    backend_choice: str,
    shots: int = 4096,
    simulator_options: dict[str, Any] | None = None,
    simulator: Any | None = None,
) -> dict[str, int]:
    backend_choice = "MPSAer" if backend_choice == "MPS" else backend_choice
    if backend_choice not in SAMPLED_BACKEND_METHODS:
        raise ValueError(f"{backend_choice} is not a sampled backend.")

    if simulator is None:
        simulator = build_bound_circuit_simulator(backend_choice, simulator_options)

    measured = circuit.copy()
    measured.measure_all()
    result = simulator.run(measured, shots=shots).result()
    counts = result.get_counts()
    if isinstance(counts, list):
        counts = counts[0]
    return {str(bitstring): int(count) for bitstring, count in counts.items()}


def build_bound_circuit_simulator(backend_choice: str, simulator_options: dict[str, Any] | None = None):
    """Build an AerSimulator for repeated reuse across many sample_bound_circuit_* calls.

    Constructing the C++ Aer backend has non-trivial overhead, so callers that sample
    many circuits with the same backend_choice/simulator_options (e.g. one (N, M) grid
    sweep) should build one simulator here and pass it via the `simulator=` kwarg
    instead of letting each call rebuild its own.
    """
    qiskit_imports = ensure_qiskit_imports()
    AerSimulator = qiskit_imports["AerSimulator"]
    backend_choice = "MPSAer" if backend_choice == "MPS" else backend_choice
    if backend_choice not in SAMPLED_BACKEND_METHODS:
        raise ValueError(f"{backend_choice} is not a sampled backend.")
    options = dict(simulator_options or {})
    options.setdefault("method", SAMPLED_BACKEND_METHODS[backend_choice])
    return AerSimulator(**options)


def sample_bound_circuit_memory(
    circuit,
    backend_choice: str,
    shots: int,
    simulator_options: dict[str, Any] | None = None,
    simulator: Any | None = None,
) -> list[str]:
    backend_choice = "MPSAer" if backend_choice == "MPS" else backend_choice
    if backend_choice not in SAMPLED_BACKEND_METHODS:
        raise ValueError(f"{backend_choice} is not a sampled backend.")

    if simulator is None:
        simulator = build_bound_circuit_simulator(backend_choice, simulator_options)

    measured = circuit.copy()
    measured.measure_all()
    result = simulator.run(measured, shots=int(shots), memory=True).result()
    memory = result.get_memory()
    if isinstance(memory, list):
        return [str(bitstring) for bitstring in memory]
    return [str(memory)]


def counts_to_metric_rows(
    counts: dict[str, int],
    instance_context: dict[str, Any],
    min_cut: float,
    max_cut: float,
    sum_weights: float,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows = []
    extra = extra or {}
    total = int(sum(counts.values()))
    for bitstring, count in counts.items():
        energy = maxcut_energy_from_bitstring(bitstring, instance_context)
        cut_value = energy + 0.5 * sum_weights
        approx_ratio = maxcut_approximation_ratio(min_cut, max_cut, sum_weights, energy)
        rows.append(
            {
                "bitstring": bitstring,
                "count": int(count),
                "probability": float(count) / float(total) if total else np.nan,
                "expected_energy": float(energy),
                "cut_value": float(cut_value),
                "approximation_ratio": float(approx_ratio),
                **extra,
            }
        )
    return pd.DataFrame(rows)


def summarize_counts_metrics(
    counts: dict[str, int],
    instance_context: dict[str, Any],
    min_cut: float,
    max_cut: float,
    sum_weights: float,
) -> dict[str, Any]:
    metric_rows = counts_to_metric_rows(counts, instance_context, min_cut, max_cut, sum_weights)
    expected_energy = expected_energy_from_counts(
        counts,
        maxcut_energy_from_bitstring,
        objective_context=instance_context,
    )
    return {
        "expected_energy_from_counts": expected_energy,
        "approx_ratio_mean": float(np.average(metric_rows["approximation_ratio"], weights=metric_rows["count"])),
        "approx_ratio_best": float(metric_rows["approximation_ratio"].max()),
        "best_cut_value": float(metric_rows["cut_value"].max()),
        "metric_rows": metric_rows,
    }


def sample_fixed_angles(
    cost_op,
    params: list[float],
    context: dict[str, Any],
    backend_choice: str,
    shots: int = 4096,
    evaluator_init: dict[str, Any] | None = None,
    simulator_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backend_choice = "MPSAer" if backend_choice == "MPS" else backend_choice
    start = time.perf_counter()
    expected_energy_eval = None
    backend_metadata: dict[str, Any] = {}
    if backend_choice == "MPSAer":
        evaluator = MPSAerSampleEvaluator(shots=shots, **(evaluator_init or {}))
        expected_energy_eval = evaluator.evaluate(cost_op=cost_op, params=params)
        counts = evaluator.get_results_from_last_iteration()["counts"]
        backend_metadata["evaluator_init"] = evaluator.to_config()
    else:
        counts = sample_bound_circuit_counts(
            build_bound_qaoa_circuit(cost_op, params),
            backend_choice=backend_choice,
            shots=shots,
            simulator_options=simulator_options,
        )
        if simulator_options:
            backend_metadata["simulator_options"] = dict(simulator_options)
    runtime = time.perf_counter() - start
    summary = summarize_counts_metrics(
        counts,
        context["instance_context"],
        context["min_cut"],
        context["max_cut"],
        context["sum_weights"],
    )
    result = {
        "simulation_method": backend_choice,
        "supports_sampling": True,
        "counts": counts,
        "num_shots": shots,
        "runtime_seconds": runtime,
        **{key: value for key, value in summary.items() if key != "metric_rows"},
        "metric_rows": summary["metric_rows"],
        **backend_metadata,
    }
    if expected_energy_eval is not None:
        result["expected_energy_eval"] = expected_energy_eval
    return result


def strategy_family(method_name: str) -> str:
    if method_name == "PT_PP_AAA":
        return "PT"
    if method_name == "linear_ramp_no_opt":
        return "LR_no_opt"
    if method_name.startswith("FA_MPSAer"):
        return "FA_MPSAer_native"
    if method_name.startswith("LR_MPSAer"):
        return "LR_MPSAer_native"
    if method_name.startswith("FA_"):
        return "FA"
    if method_name.startswith("LR_"):
        return "LR"
    return method_name


def strategy_runtime_label(method_name: str, reps: int) -> str:
    return f"{method_name}_{int(reps)}"


def stage_records_from_bundle(result_bundle: dict[str | int, Any]) -> list[dict[str, Any]]:
    int_keys = sorted(key for key in result_bundle if isinstance(key, int))
    return [result_bundle[key] for key in int_keys]


def total_training_shots_from_bundle(result_bundle: dict[str | int, Any]) -> int:
    return int(sum(int(stage.get("training_shots_used", 0) or 0) for stage in stage_records_from_bundle(result_bundle)))


def total_objective_evaluations_from_bundle(result_bundle: dict[str | int, Any]) -> int:
    return int(
        sum(int(stage.get("num_objective_evaluations", 0) or 0) for stage in stage_records_from_bundle(result_bundle))
    )


def total_train_duration_from_bundle(result_bundle: dict[str | int, Any]) -> float:
    return float(sum(float(stage.get("train_duration", 0.0) or 0.0) for stage in stage_records_from_bundle(result_bundle)))


def build_sampled_training_config(
    method_name: str,
    config: dict[str, Any],
    *,
    reps: int,
    cobyla_maxiter: int | None,
    shots_per_evaluation: int,
    sample_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sampled_config = copy.deepcopy(config)
    evaluator_init = dict(sample_config or {})
    evaluator_init["shots"] = int(shots_per_evaluation)

    for stage in sampled_config.get("trainer_chain", []):
        trainer_name = stage.get("trainer")
        trainer_init = stage.setdefault("trainer_init", {})
        train_kwargs = stage.setdefault("train_kwargs", {})

        if trainer_name == "TransferTrainer":
            trainer_init.pop("evaluator", None)
            trainer_init.pop("evaluator_init", None)
            train_kwargs["qaoa_depth"] = int(reps)
            train_kwargs.pop("reps", None)
            continue

        if trainer_name in {"FixedAngleConjecture", "TQATrainer"}:
            train_kwargs["reps"] = int(reps)

        if "evaluator" in trainer_init:
            trainer_init["evaluator"] = "MPSAerSampleEvaluator"
            trainer_init["evaluator_init"] = dict(evaluator_init)

        if cobyla_maxiter is not None and trainer_name in {"ScipyTrainer", "TQATrainer"}:
            minimize_args = copy.deepcopy(trainer_init.get("minimize_args", {}))
            options = copy.deepcopy(minimize_args.get("options", {}))
            options["maxiter"] = int(cobyla_maxiter)
            minimize_args["options"] = options
            minimize_args.setdefault("method", "COBYLA")
            trainer_init["minimize_args"] = minimize_args

        if trainer_name == "RecursionTrainer":
            # RecursionTrainer builds one ScipyTrainer instance up front and reuses
            # it for every recursion step (see RecursionTrainer.train's while loop
            # in qaoa_training_pipeline), so overriding this nested trainer_init
            # once here applies the same (N, M) grid point to every sub-depth from
            # p=2 up to the target depth, not just the p=1 warm-start stage above.
            nested_init = trainer_init.setdefault("trainer_init", {})
            if "evaluator" in nested_init:
                nested_init["evaluator"] = "MPSAerSampleEvaluator"
                nested_init["evaluator_init"] = dict(evaluator_init)
            if cobyla_maxiter is not None:
                nested_minimize_args = copy.deepcopy(nested_init.get("minimize_args", {}))
                nested_options = copy.deepcopy(nested_minimize_args.get("options", {}))
                nested_options["maxiter"] = int(cobyla_maxiter)
                nested_minimize_args["options"] = nested_options
                nested_minimize_args.setdefault("method", "COBYLA")
                nested_init["minimize_args"] = nested_minimize_args

    return sampled_config


def run_strategy_with_final_sampling(
    spec: InstanceSpec,
    strategy_name: str,
    reps: int,
    *,
    final_shots: int,
    pipeline_repo: str | Path | None = None,
    main_repo: str | Path | None = None,
    method_configs: dict[str, Path] | None = None,
    linear_ramp_slopes: tuple[float, float] = (0.5, 0.5),
    training_mode: str = "native",
    cobyla_maxiter: int | None = None,
    training_shots_per_eval: int | None = None,
    sample_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = load_instance_context(spec, main_repo)
    cost_op = None

    if strategy_name == "linear_ramp_no_opt":
        cost_op = build_cost_operator_from_graph(context["graph_path"], pipeline_repo)
        angle_payload = generate_angle_payload(
            cost_op,
            angle_method=strategy_name,
            reps=reps,
            method_configs=method_configs,
            main_repo=main_repo,
            pipeline_repo=pipeline_repo,
            linear_ramp_slopes=linear_ramp_slopes,
            disable_transfer_evaluator=True,
            spec=spec,
            graph_path=context["graph_path"],
        )
        result_bundle = angle_payload["result_bundle"]
    elif training_mode == "native":
        if "_PP" not in strategy_name:
            cost_op = build_cost_operator_from_graph(context["graph_path"], pipeline_repo)
        angle_payload = generate_angle_payload(
            cost_op,
            angle_method=strategy_name,
            reps=reps,
            method_configs=method_configs,
            main_repo=main_repo,
            pipeline_repo=pipeline_repo,
            linear_ramp_slopes=linear_ramp_slopes,
            disable_transfer_evaluator=strategy_name in ZERO_TRAINING_METHODS,
            spec=spec,
            graph_path=context["graph_path"],
        )
        if cost_op is None:
            saved_cost_operator = angle_payload.get("saved_cost_operator")
            if saved_cost_operator is None:
                raise ValueError(
                    f"Saved training result for {strategy_name} did not include a serialized cost operator."
                )
            cost_op = build_cost_operator_from_serialized(saved_cost_operator)
        result_bundle = angle_payload["result_bundle"]
    elif training_mode == "sampled_mps":
        cost_op = build_cost_operator_from_graph(context["graph_path"], pipeline_repo)
        if method_configs is None or strategy_name not in method_configs:
            raise ValueError(f"Unknown strategy {strategy_name} for sampled training.")
        sampled_config = build_sampled_training_config(
            strategy_name,
            load_method_config(method_configs[strategy_name]),
            reps=reps,
            cobyla_maxiter=cobyla_maxiter,
            shots_per_evaluation=int(training_shots_per_eval or final_shots),
            sample_config=sample_config,
        )
        result_bundle = run_method_from_config(
            cost_op,
            sampled_config,
            pipeline_repo=pipeline_repo,
            main_repo=main_repo,
            train_kwargs_overrides=native_train_kwargs_overrides(
                sampled_config,
                reps=reps,
                spec=spec,
                graph_path=context["graph_path"],
            ),
            disable_transfer_evaluator=strategy_name in ZERO_TRAINING_METHODS,
        )
        angle_payload = {
            "angle_method": strategy_name,
            "angles": latest_stage_data(result_bundle)["optimized_qaoa_angles"],
            "raw_result": latest_stage_data(result_bundle),
            "result_bundle": result_bundle,
        }
    else:
        raise ValueError(f"Unsupported training_mode {training_mode}.")

    sample_result = sample_fixed_angles(
        cost_op,
        angle_payload["angles"],
        context,
        backend_choice="MPSAer",
        shots=final_shots,
        evaluator_init=sample_config,
    )

    training_shots_used = total_training_shots_from_bundle(result_bundle)
    num_objective_evaluations = total_objective_evaluations_from_bundle(result_bundle)
    runtime_train = total_train_duration_from_bundle(result_bundle)

    if strategy_name in ZERO_TRAINING_METHODS:
        training_shots_used = 0
        num_objective_evaluations = 0

    return {
        "strategy": strategy_name,
        "strategy_family": strategy_family(strategy_name),
        "strategy_runtime_label": strategy_runtime_label(strategy_name, reps),
        "angle_method": angle_payload["angle_method"],
        "saved_result_path": angle_payload.get("saved_result_path"),
        "training_mode": training_mode,
        "simulation_method": "MPSAer",
        "family": spec.graph_type,
        "graph_type": spec.graph_type,
        "num_nodes": spec.num_nodes,
        "instance_id": spec.instance_id,
        "short_name": context["short_name"],
        "p": int(reps),
        "angles": [float(value) for value in angle_payload["angles"]],
        "counts": sample_result["counts"],
        "num_shots_final": int(final_shots),
        "runtime_train_wallclock": runtime_train,
        "runtime_sample_wallclock": float(sample_result["runtime_seconds"]),
        "runtime_seconds": float(runtime_train + sample_result["runtime_seconds"]),
        "num_objective_evaluations": int(num_objective_evaluations),
        "total_training_shots": int(training_shots_used),
        "supports_sampling": True,
        "expected_energy_eval": sample_result.get("expected_energy_eval"),
        "expected_energy_from_counts": sample_result["expected_energy_from_counts"],
        "Approximation_Ratio": sample_result["approx_ratio_mean"],
        "BestApproximationRatio": sample_result["approx_ratio_best"],
        "approx_ratio_mean": sample_result["approx_ratio_mean"],
        "approx_ratio_best": sample_result["approx_ratio_best"],
        "best_found_value": sample_result["best_cut_value"],
        "best_cut_value": sample_result["best_cut_value"],
        "metric_rows": sample_result["metric_rows"],
        "evaluator_init": sample_result.get("evaluator_init"),
        "result_bundle": result_bundle,
    }


def _mpsaer_simulator_options(sample_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(sample_config or {})
    options: dict[str, Any] = {}
    if "chi" in config:
        options["matrix_product_state_max_bond_dimension"] = int(config["chi"])
    if "max_parallel_threads" in config:
        options["max_parallel_threads"] = int(config["max_parallel_threads"])
    return options


def _first_optimizer_stage_index(config: dict[str, Any]) -> int | None:
    for idx, stage in enumerate(config.get("trainer_chain", [])):
        if stage.get("trainer") in {"ScipyTrainer", "TQATrainer"}:
            return idx
    return None


def _prepare_prefix_rows(
    *,
    sample_stream: list[str],
    q_values: list[int],
    context: dict[str, Any],
    base_row: dict[str, Any],
) -> list[dict[str, Any]]:
    prefix_metrics = best_prefix_metrics(
        sample_stream,
        checkpoints=[int(q) for q in q_values],
        instance_context=context["instance_context"],
        min_cut=float(context["min_cut"]),
        max_cut=float(context["max_cut"]),
        sum_weights=float(context["sum_weights"]),
    )
    rows: list[dict[str, Any]] = []
    for prefix in prefix_metrics:
        rows.append(
            {
                **base_row,
                "Q": int(prefix["Q"]),
                "counts": prefix["counts"],
                "best_bitstring": prefix["best_bitstring"],
                "best_found_value": float(prefix["best_cut_value"]),
                "BestApproximationRatio": float(prefix["best_approx_ratio"]),
            }
        )
    return rows


def _instance_average_degree(graph_path: str | Path, num_nodes: int) -> float:
    graph_data = load_json(graph_path)
    if "edge list" not in graph_data:
        raise KeyError(f"Expected 'edge list' in graph file: {graph_path}")
    return 2.0 * float(len(graph_data["edge list"])) / float(num_nodes)


def _load_depth_matched_pt_angles(
    database_path: str | Path,
    *,
    reps: int,
    graph_path: str | Path,
    num_nodes: int,
) -> list[float]:
    database = load_json(database_path)
    avg_degree = _instance_average_degree(graph_path, num_nodes)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for key, value in database.items():
        features = tuple(float(part.strip()) for part in str(key).split(","))
        if not features or int(round(features[0])) != int(reps):
            continue
        degree_distance = abs(float(features[1]) - avg_degree) if len(features) > 1 else 0.0
        candidates.append((degree_distance, value))

    if not candidates:
        raise ValueError(
            f"No PT transfer angles with QAOA depth p={reps} were found in {database_path}."
        )

    _, selected = min(candidates, key=lambda item: item[0])
    angles = np.asarray(selected["qaoa_angles"], dtype=float)
    if angles.ndim == 2:
        angles = np.mean(angles, axis=0)
    angles = np.ravel(angles)
    if len(angles) // 2 != int(reps):
        raise ValueError(
            "Depth-matched PT database lookup returned angles for the wrong QAOA depth."
        )
    return angles.tolist()


def _resolve_time_per_shot(time_per_shot: float | dict[int, float], reps: int) -> float:
    """Return the per-shot time for a given QAOA depth.

    Accepts a depth-independent float or a {depth: seconds} mapping.
    When the mapping is provided but the exact depth is absent, falls back
    to the mean of all available depth values so no run silently uses 1.0.
    """
    if isinstance(time_per_shot, dict):
        if int(reps) in time_per_shot:
            return float(time_per_shot[int(reps)])
        if time_per_shot:
            return float(sum(time_per_shot.values()) / len(time_per_shot))
        return 1.0
    return float(time_per_shot)


def run_pt_pss_exact_points(
    spec: InstanceSpec,
    *,
    reps: int,
    q_values: list[int],
    method_name: str = "PT_PP_AAA",
    method_configs: dict[str, Path] | None = None,
    main_repo: str | Path | None = None,
    pipeline_repo: str | Path | None = None,
    sample_config: dict[str, Any] | None = None,
    transfer_cost: float = 0.0,
    time_per_shot: float | dict[int, float] = 1.0,
) -> pd.DataFrame:
    if not q_values:
        return pd.DataFrame()
    _tps = _resolve_time_per_shot(time_per_shot, reps)
    if method_configs is None or method_name not in method_configs:
        raise ValueError(f"Missing method config for {method_name}.")
    if method_name == "PT_PP_AAA":
        required_transfer_db = (
            main_repo_path(main_repo) / "data" / "training" / "parameter_transfer" / "optimized_angles_database_HH_cluster.json"
        )
        if not required_transfer_db.exists():
            raise FileNotFoundError(
                "PT_PP_AAA for heavy-hex instances requires "
                f"'{required_transfer_db.name}', but that file is not present in the current "
                "QAOA-Parameter-Setting checkout. The generic optimized_angles_database.json "
                "cannot be used as a drop-in fallback here because it has incompatible transfer "
                "feature keys."
            )

    context = load_instance_context(spec, main_repo)
    cost_op = build_cost_operator_from_graph(context["graph_path"], pipeline_repo)
    method_config = load_method_config(method_configs[method_name])
    transfer_start = time.perf_counter()
    if method_name == "PT_PP_AAA":
        qaoa_angles = _load_depth_matched_pt_angles(
            required_transfer_db,
            reps=reps,
            graph_path=context["graph_path"],
            num_nodes=spec.num_nodes,
        )
        runtime_train_wallclock = time.perf_counter() - transfer_start
    else:
        def _strip_evaluators(config: dict) -> dict:
            # FixedAngleConjecture computes angles from an analytic formula and
            # does not need a circuit evaluator.  Stripping it here prevents PP
            # (or any other evaluator) from running a full circuit simulation
            # during angle generation for training-independent methods.
            for stage in config.get("trainer_chain", []):
                if stage.get("trainer") == "FixedAngleConjecture":
                    stage.get("trainer_init", {}).pop("evaluator", None)
                    stage.get("trainer_init", {}).pop("evaluator_init", None)
            return config

        result_bundle = run_method_from_config(
            cost_op,
            method_config,
            main_repo=main_repo,
            pipeline_repo=pipeline_repo,
            train_kwargs_overrides=native_train_kwargs_overrides(
                method_config,
                reps=reps,
                spec=spec,
                graph_path=context["graph_path"],
            ),
            disable_transfer_evaluator=True,
            allow_transfer_fallback=False,
            config_mutator=_strip_evaluators,
        )
        stage = latest_stage_data(result_bundle)
        qaoa_angles = stage["optimized_qaoa_angles"]
        runtime_train_wallclock = float(total_train_duration_from_bundle(result_bundle))

    circuit = build_bound_qaoa_circuit(cost_op, qaoa_angles)
    q_max = max(int(q) for q in q_values)
    sample_start = time.perf_counter()
    sample_stream = sample_bound_circuit_memory(
        circuit,
        backend_choice="MPSAer",
        shots=q_max,
        simulator_options=_mpsaer_simulator_options(sample_config),
    )
    sample_runtime = time.perf_counter() - sample_start
    sample_time_per_shot = float(sample_runtime) / float(q_max)

    rows = _prepare_prefix_rows(
        sample_stream=sample_stream,
        q_values=q_values,
        context=context,
        base_row={
            "graph_type": spec.graph_type,
            "num_nodes": int(spec.num_nodes),
            "instance": spec.instance_id,
            "instance_id": spec.instance_id,
            "split": spec.split or "test",
            "source": spec.source or "repo_test",
            "p": int(reps),
            "strategy": method_name,
            "strategy_family": strategy_family(method_name),
            "strategy_runtime_label": strategy_runtime_label(method_name, reps),
            "simulation_method": "MPSAer",
            "N": 0,
            "M": 0,
            "classical_setup_cost": runtime_train_wallclock + float(transfer_cost),
            "training_cost": runtime_train_wallclock + float(transfer_cost),
            "sampling_cost": 0.0,
            "T_exact": 0.0,
            "classical_setup_cost_proxy": float(transfer_cost),
            "training_cost_proxy": float(transfer_cost),
            "sampling_cost_proxy": 0.0,
            "T_exact_proxy": 0.0,
            "runtime_train_wallclock": runtime_train_wallclock,
            "runtime_sample_wallclock": 0.0,
        },
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["runtime_sample_wallclock"] = df["Q"].astype(float) * sample_time_per_shot
    df["sampling_cost"] = df["runtime_sample_wallclock"].astype(float)
    df["sampling_cost_proxy"] = df["Q"].astype(float) * _tps
    df["T_exact"] = df["training_cost"].astype(float) + df["sampling_cost"].astype(float)
    df["T_exact_proxy"] = df["training_cost_proxy"].astype(float) + df["sampling_cost_proxy"].astype(float)
    df["Approximation_Ratio"] = df["BestApproximationRatio"]
    df["num_objective_evaluations"] = 0
    df["total_training_shots"] = 0
    return df


def run_fa_pss_exact_points(
    spec: InstanceSpec,
    *,
    reps: int,
    n_values: list[int],
    m_values: list[int],
    q_values: list[int],
    method_name: str = "FA_PP_opt",
    method_configs: dict[str, Path] | None = None,
    main_repo: str | Path | None = None,
    pipeline_repo: str | Path | None = None,
    sample_config: dict[str, Any] | None = None,
    time_per_shot: float | dict[int, float] = 1.0,
    cobyla_overhead_c: float = 1.0,
) -> pd.DataFrame:
    if method_configs is None or method_name not in method_configs:
        raise ValueError(f"Missing method config for {method_name}.")
    if not n_values or not m_values or not q_values:
        return pd.DataFrame()

    _tps = _resolve_time_per_shot(time_per_shot, reps)
    context = load_instance_context(spec, main_repo)
    cost_op = build_cost_operator_from_graph(context["graph_path"], pipeline_repo)
    base_config = load_method_config(method_configs[method_name])
    optimizer_stage_idx = _first_optimizer_stage_index(base_config)
    q_max = max(int(q) for q in q_values)
    rows: list[dict[str, Any]] = []

    # Built once and reused across the whole (N, M) grid below: backend_choice and
    # simulator_options are constant across the grid, only the circuit changes, so
    # rebuilding the Aer C++ backend on every grid point is pure overhead.
    grid_simulator = build_bound_circuit_simulator("MPSAer", _mpsaer_simulator_options(sample_config))

    for m_value in sorted({int(m) for m in m_values}):
        previous_angles: list[float] | None = None
        previous_n = 0
        cumulative_evals = 0
        cumulative_shots = 0
        cumulative_duration = 0.0

        for n_value in sorted({int(n) for n in n_values}):
            delta_n = int(n_value) - int(previous_n)
            if delta_n <= 0:
                continue

            sampled_config = build_sampled_training_config(
                method_name,
                base_config,
                reps=reps,
                cobyla_maxiter=delta_n,
                shots_per_evaluation=int(m_value),
                sample_config=sample_config,
            )
            overrides = native_train_kwargs_overrides(
                sampled_config,
                reps=reps,
                spec=spec,
                graph_path=context["graph_path"],
            )
            if previous_angles is not None and optimizer_stage_idx is not None:
                overrides.setdefault(int(optimizer_stage_idx), {})
                overrides[int(optimizer_stage_idx)]["params0"] = list(previous_angles)

            result_bundle = run_method_from_config(
                cost_op,
                sampled_config,
                pipeline_repo=pipeline_repo,
                main_repo=main_repo,
                train_kwargs_overrides=overrides,
            )
            stage = latest_stage_data(result_bundle)
            previous_angles = list(stage["optimized_qaoa_angles"])
            previous_n = int(n_value)
            cumulative_evals += total_objective_evaluations_from_bundle(result_bundle)
            cumulative_shots += total_training_shots_from_bundle(result_bundle)
            cumulative_duration += total_train_duration_from_bundle(result_bundle)

            circuit = build_bound_qaoa_circuit(cost_op, previous_angles)
            sample_start = time.perf_counter()
            sample_stream = sample_bound_circuit_memory(
                circuit,
                backend_choice="MPSAer",
                shots=q_max,
                simulator=grid_simulator,
            )
            sample_runtime = time.perf_counter() - sample_start
            sample_time_per_shot = float(sample_runtime) / float(q_max)
            rows.extend(
                _prepare_prefix_rows(
                    sample_stream=sample_stream,
                    q_values=q_values,
                    context=context,
                    base_row={
                        "graph_type": spec.graph_type,
                        "num_nodes": int(spec.num_nodes),
                        "instance": spec.instance_id,
                        "instance_id": spec.instance_id,
                        "split": spec.split or "train",
                        "source": spec.source or "generated_train",
                        "p": int(reps),
                        "strategy": method_name,
                        "strategy_family": strategy_family(method_name),
                        "strategy_runtime_label": strategy_runtime_label(method_name, reps),
                        "simulation_method": "MPSAer",
                        "N": int(n_value),
                        "M": int(m_value),
                        "classical_setup_cost": 0.0,
                        "training_cost": float(cumulative_duration),
                        "sampling_cost": 0.0,
                        "T_exact": 0.0,
                        "classical_setup_cost_proxy": 0.0,
                        "training_cost_proxy": float(cobyla_overhead_c) * float(n_value) * float(m_value) * _tps,
                        "sampling_cost_proxy": 0.0,
                        "T_exact_proxy": 0.0,
                        "num_objective_evaluations": int(cumulative_evals),
                        "total_training_shots": int(cumulative_shots),
                        "runtime_train_wallclock": float(cumulative_duration),
                        "runtime_sample_wallclock": 0.0,
                        "runtime_sample_per_shot": sample_time_per_shot,
                    },
                )
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "runtime_sample_per_shot" in df.columns:
        df["runtime_sample_wallclock"] = (
            df["Q"].astype(float) * df["runtime_sample_per_shot"].astype(float)
        )
        df = df.drop(columns=["runtime_sample_per_shot"])
    else:
        df["runtime_sample_wallclock"] = 0.0
    df["sampling_cost"] = df["runtime_sample_wallclock"].astype(float)
    df["sampling_cost_proxy"] = df["Q"].astype(float) * _tps
    df["T_exact"] = df["training_cost"].astype(float) + df["sampling_cost"].astype(float)
    df["T_exact_proxy"] = df["training_cost_proxy"].astype(float) + df["sampling_cost_proxy"].astype(float)
    df["Approximation_Ratio"] = df["BestApproximationRatio"]
    return df


def build_dense_budget_grid(
    exact_df: pd.DataFrame,
    *,
    num_points: int = 1000,
    resource_col: str = "T_exact_proxy",
    scale: str = "log",
) -> list[float]:
    if exact_df.empty:
        return []

    values = (
        exact_df[resource_col]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values()
    )
    if values.empty:
        return []

    lo = float(values.iloc[0])
    hi = float(values.iloc[-1])
    if hi <= lo:
        return [lo]

    count = max(int(num_points), 2)
    if scale == "log":
        positive = values[values > 0]
        if positive.empty:
            return [float(val) for val in np.linspace(lo, hi, count)]
        lo = float(positive.iloc[0])
        grid = np.geomspace(lo, hi, count)
    else:
        grid = np.linspace(lo, hi, count)

    return [float(val) for val in np.unique(np.round(grid, 12))]


def build_budget_bin_edges(
    exact_df: pd.DataFrame,
    *,
    num_bins: int = 1000,
    resource_col: str = "T_exact_proxy",
    scale: str = "log",
) -> np.ndarray:
    if exact_df.empty:
        return np.array([], dtype=float)

    values = (
        exact_df[resource_col]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values()
    )
    if values.empty:
        return np.array([], dtype=float)

    lo = float(values.iloc[0])
    hi = float(values.iloc[-1])
    if hi <= lo:
        return np.array([lo, hi], dtype=float)

    count = max(int(num_bins), 1) + 1
    if scale == "log":
        positive = values[values > 0]
        if positive.empty:
            return np.linspace(lo, hi, count)
        lo = float(positive.iloc[0])
        return np.geomspace(lo, hi, count)

    return np.linspace(lo, hi, count)


def _centers_from_edges(edges: np.ndarray, *, scale: str = "log") -> np.ndarray:
    if edges.size < 2:
        return np.array([], dtype=float)

    if scale == "log":
        return np.sqrt(edges[:-1] * edges[1:])

    return 0.5 * (edges[:-1] + edges[1:])


def _edges_from_centers(centers: np.ndarray, *, scale: str = "log") -> np.ndarray:
    if centers.size == 0:
        return np.array([], dtype=float)
    if centers.size == 1:
        center = float(centers[0])
        if scale == "log" and center > 0:
            return np.array([center / np.sqrt(2.0), center * np.sqrt(2.0)], dtype=float)
        width = max(abs(center) * 0.5, 1e-12)
        return np.array([center - width, center + width], dtype=float)

    edges = np.empty(centers.size + 1, dtype=float)
    if scale == "log" and np.all(centers > 0):
        log_centers = np.log(centers)
        log_edges = np.empty(centers.size + 1, dtype=float)
        log_edges[1:-1] = 0.5 * (log_centers[:-1] + log_centers[1:])
        log_edges[0] = log_centers[0] - (log_edges[1] - log_centers[0])
        log_edges[-1] = log_centers[-1] + (log_centers[-1] - log_edges[-2])
        return np.exp(log_edges)

    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def build_binned_budget_dataset(
    exact_df: pd.DataFrame,
    *,
    t_grid: list[float] | None = None,
    num_bins: int = 1000,
    budget_col: str = "T_exact_proxy",
    scale: str = "log",
) -> pd.DataFrame:
    if exact_df.empty:
        return pd.DataFrame()

    binned = exact_df.copy()
    if t_grid is not None:
        centers = np.asarray(sorted({float(val) for val in t_grid}), dtype=float)
        edges = _edges_from_centers(centers, scale=scale)
    else:
        edges = build_budget_bin_edges(
            exact_df,
            num_bins=num_bins,
            resource_col=budget_col,
            scale=scale,
        )
        centers = _centers_from_edges(edges, scale=scale)

    if edges.size < 2 or centers.size == 0:
        return pd.DataFrame()

    values = pd.to_numeric(binned[budget_col], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return pd.DataFrame()

    valid_values = values[finite_mask]
    bin_idx = np.searchsorted(edges, valid_values, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(centers) - 1)

    assigned = binned.loc[finite_mask].copy()
    assigned["budget_bin"] = bin_idx.astype(int)
    assigned["budget_left"] = edges[bin_idx].astype(float)
    assigned["budget_right"] = edges[bin_idx + 1].astype(float)
    assigned["T"] = centers[bin_idx].astype(float)
    assigned["resource"] = assigned["T"].astype(float)
    assigned["selected_exact_T"] = assigned.get("T_exact", np.nan)
    assigned["selected_exact_T_proxy"] = assigned.get("T_exact_proxy", np.nan)
    assigned["budget_distance"] = 0.0

    return assigned.reset_index(drop=True)


def build_cumulative_budget_frontier(
    exact_df: pd.DataFrame,
    *,
    t_grid: list[float] | None = None,
    num_points: int = 1000,
    budget_col: str = "T_exact_proxy",
    response_col: str = "BestApproximationRatio",
    scale: str = "log",
    group_cols: tuple[str, ...] = (
        "graph_type",
        "num_nodes",
        "instance",
        "split",
        "strategy",
        "simulation_method",
        "p",
    ),
) -> pd.DataFrame:
    """Build a common-budget cumulative PSS frontier.

    For every instance/method group and every budget in ``t_grid``, select the
    best exact PSS row whose resource cost is no larger than that budget.  This
    creates the common-support table expected by the Window Sticker workflow:
    stochastic-benchmark sees one best-feasible row per instance and budget
    rather than sparse exact rows assigned to bins.
    """
    if exact_df.empty:
        return pd.DataFrame()
    if budget_col not in exact_df.columns:
        raise KeyError(f"Expected budget column '{budget_col}' in exact_df.")
    if response_col not in exact_df.columns:
        raise KeyError(f"Expected response column '{response_col}' in exact_df.")

    if t_grid is None:
        t_grid = build_dense_budget_grid(
            exact_df,
            num_points=num_points,
            resource_col=budget_col,
            scale=scale,
        )
    grid = np.asarray(sorted({float(value) for value in t_grid}), dtype=float)
    grid = grid[np.isfinite(grid)]
    if grid.size == 0:
        return pd.DataFrame()

    present_group_cols = [col for col in group_cols if col in exact_df.columns]
    rows: list[pd.DataFrame] = []

    work = exact_df.copy()
    work[budget_col] = pd.to_numeric(work[budget_col], errors="coerce")
    work[response_col] = pd.to_numeric(work[response_col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[budget_col, response_col])
    work = work[work[budget_col] > 0]
    if work.empty:
        return pd.DataFrame()

    for _, group in work.groupby(present_group_cols, dropna=False):
        group = group.sort_values(
            [budget_col, response_col],
            ascending=[True, False],
        ).reset_index(drop=True)
        budgets = group[budget_col].to_numpy(dtype=float)
        responses = group[response_col].to_numpy(dtype=float)
        if budgets.size == 0:
            continue

        best_indices = np.empty(len(group), dtype=int)
        best_idx = 0
        best_response = -np.inf
        for idx, response in enumerate(responses):
            if response > best_response:
                best_response = float(response)
                best_idx = idx
            best_indices[idx] = best_idx

        grid_positions = np.searchsorted(budgets, grid, side="right") - 1
        feasible_mask = grid_positions >= 0
        if not feasible_mask.any():
            continue

        feasible_grid = grid[feasible_mask]
        selected_positions = best_indices[grid_positions[feasible_mask]]
        selected = group.iloc[selected_positions].copy().reset_index(drop=True)
        selected["budget_bin"] = np.flatnonzero(feasible_mask).astype(int)
        selected["budget_left"] = feasible_grid
        selected["budget_right"] = feasible_grid
        selected["T"] = feasible_grid
        selected["resource"] = feasible_grid
        selected["selected_exact_T"] = selected.get("T_exact", np.nan)
        selected["selected_exact_T_proxy"] = selected.get("T_exact_proxy", np.nan)
        selected["budget_distance"] = feasible_grid - selected[budget_col].astype(float).to_numpy()
        rows.append(selected)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _resource_distance(
    values: pd.Series,
    target: float,
    *,
    scale: str = "log",
) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    distances = np.full(arr.shape, np.inf, dtype=float)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return distances

    if scale == "log" and target > 0:
        positive_mask = finite_mask & (arr > 0)
        if positive_mask.any():
            distances[positive_mask] = np.abs(np.log(arr[positive_mask]) - np.log(float(target)))
            return distances

    distances[finite_mask] = np.abs(arr[finite_mask] - float(target))
    return distances


def build_budget_frontier(
    exact_df: pd.DataFrame,
    *,
    t_grid: list[float],
    response_col: str = "BestApproximationRatio",
    budget_col: str = "T_exact_proxy",
    distance_scale: str = "log",
) -> pd.DataFrame:
    return build_cumulative_budget_frontier(
        exact_df,
        t_grid=t_grid,
        budget_col=budget_col,
        response_col=response_col,
        scale=distance_scale,
    )


def attach_ws_codebook_columns(
    frontier_df: pd.DataFrame,
    codebooks: dict[str, dict[int, Any]],
) -> pd.DataFrame:
    """Attach stochastic-benchmark code columns expected by the WS workflow."""
    if frontier_df.empty:
        return frontier_df

    out = frontier_df.copy()
    strategy_reverse = {str(value): int(code) for code, value in codebooks.get("strategy", {}).items()}
    simulation_reverse = {str(value): int(code) for code, value in codebooks.get("simulation", {}).items()}
    out["strategy_code"] = out["strategy"].astype(str).map(strategy_reverse).astype(float)
    out["simulation_code"] = out["simulation_method"].astype(str).map(simulation_reverse).astype(float)
    out["train"] = out["split"].astype(str).map({"train": 1, "test": 0}).astype(int)
    return out


def summarize_frontier_instance_coverage(
    frontier_df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("split", "strategy", "simulation_method", "p"),
) -> pd.DataFrame:
    """Summarize how many instances support each method/resource frontier."""
    if frontier_df.empty or "T" not in frontier_df.columns or "instance" not in frontier_df.columns:
        return pd.DataFrame()

    present_group_cols = [col for col in group_cols if col in frontier_df.columns]
    per_budget = (
        frontier_df.groupby(present_group_cols + ["T"], dropna=False)
        .agg(n_instances=("instance", "nunique"), n_rows=("instance", "size"))
        .reset_index()
    )
    if per_budget.empty:
        return pd.DataFrame()

    expected = (
        per_budget.groupby(present_group_cols, dropna=False)["n_instances"]
        .max()
        .rename("expected_instances")
        .reset_index()
    )
    summary = per_budget.merge(expected, on=present_group_cols, how="left")
    summary["has_full_instance_support"] = summary["n_instances"].eq(summary["expected_instances"])
    return (
        summary.groupby(present_group_cols, dropna=False)
        .agg(
            expected_instances=("expected_instances", "max"),
            min_instances=("n_instances", "min"),
            max_instances=("n_instances", "max"),
            n_budget_points=("T", "nunique"),
            n_partial_budget_points=("has_full_instance_support", lambda values: int((~values).sum())),
        )
        .reset_index()
    )


def build_resource_frontier_from_exact_points(
    exact_df: pd.DataFrame,
    *,
    codebooks: dict[str, dict[int, Any]],
    budget_col: str,
    num_bins: int = 1000,
    scale: str = "log",
) -> pd.DataFrame:
    """Build a cumulative common-budget frontier using a resource column."""
    frontier_df = build_cumulative_budget_frontier(
        exact_df,
        num_points=num_bins,
        budget_col=budget_col,
        scale=scale,
    )
    return attach_ws_codebook_columns(frontier_df, codebooks)


def _positive_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=columns)


def _fit_exact_resource_model(
    train_df: pd.DataFrame,
    model: str,
    *,
    target_col: str,
) -> dict[str, Any]:
    n = train_df["N"].to_numpy(dtype=float)
    m = train_df["M"].to_numpy(dtype=float)
    q = train_df["Q"].to_numpy(dtype=float)
    y = train_df[target_col].to_numpy(dtype=float)

    if model == "proxy_like_nm_q":
        x = np.column_stack([n * m, q])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        expression = f"T_exact = {beta[0]:.6g} N M + {beta[1]:.6g} Q"
    elif model == "linear_n_m_q":
        x = np.column_stack([np.ones(len(train_df)), n, m, q])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        expression = f"T_exact = {beta[0]:.6g} + {beta[1]:.6g} N + {beta[2]:.6g} M + {beta[3]:.6g} Q"
    elif model == "linear_n_m_q_nm":
        x = np.column_stack([np.ones(len(train_df)), n, m, q, n * m])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        expression = (
            f"T_exact = {beta[0]:.6g} + {beta[1]:.6g} N + {beta[2]:.6g} M "
            f"+ {beta[3]:.6g} Q + {beta[4]:.6g} N M"
        )
    elif model == "interaction_n_m_q":
        x = np.column_stack([np.ones(len(train_df)), n, m, q, n * m, n * q, m * q])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        expression = (
            f"T_exact = {beta[0]:.6g} + {beta[1]:.6g} N + {beta[2]:.6g} M + {beta[3]:.6g} Q "
            f"+ {beta[4]:.6g} N M + {beta[5]:.6g} N Q + {beta[6]:.6g} M Q"
        )
    elif model == "power_law_n_m_q":
        x = np.column_stack([np.ones(len(train_df)), np.log(n), np.log(m), np.log(q)])
        beta = np.linalg.lstsq(x, np.log(y), rcond=None)[0]
        expression = f"T_exact = {np.exp(beta[0]):.6g} N^{beta[1]:.6g} M^{beta[2]:.6g} Q^{beta[3]:.6g}"
    else:
        raise ValueError(f"Unknown exact-resource model: {model}")

    return {"model": model, "beta": beta, "expression": expression}


def _predict_exact_resource_model(df: pd.DataFrame, fit: dict[str, Any]) -> np.ndarray:
    model = str(fit["model"])
    beta = np.asarray(fit["beta"], dtype=float)
    n = df["N"].to_numpy(dtype=float)
    m = df["M"].to_numpy(dtype=float)
    q = df["Q"].to_numpy(dtype=float)

    if model == "proxy_like_nm_q":
        pred = beta[0] * n * m + beta[1] * q
    elif model == "linear_n_m_q":
        pred = beta[0] + beta[1] * n + beta[2] * m + beta[3] * q
    elif model == "linear_n_m_q_nm":
        pred = beta[0] + beta[1] * n + beta[2] * m + beta[3] * q + beta[4] * n * m
    elif model == "interaction_n_m_q":
        pred = beta[0] + beta[1] * n + beta[2] * m + beta[3] * q + beta[4] * n * m + beta[5] * n * q + beta[6] * m * q
    elif model == "power_law_n_m_q":
        pred = np.exp(beta[0] + beta[1] * np.log(n) + beta[2] * np.log(m) + beta[3] * np.log(q))
    else:
        raise ValueError(f"Unknown exact-resource model: {model}")

    return np.maximum(pred, 1e-12)


def _exact_resource_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    if not mask.any():
        return {
            "log10_rmse": np.nan,
            "median_factor_error": np.nan,
            "p90_factor_error": np.nan,
            "log_r2": np.nan,
        }

    y = y_true[mask]
    pred = y_pred[mask]
    log_y = np.log(y)
    log_pred = np.log(pred)
    factor = np.maximum(pred / y, y / pred)
    denom = np.sum((log_y - np.mean(log_y)) ** 2)
    log_r2 = np.nan if denom <= 0 else 1.0 - float(np.sum((log_y - log_pred) ** 2) / denom)
    return {
        "log10_rmse": float(np.sqrt(np.mean((np.log10(pred) - np.log10(y)) ** 2))),
        "median_factor_error": float(np.median(factor)),
        "p90_factor_error": float(np.quantile(factor, 0.9)),
        "log_r2": log_r2,
    }


def fit_exact_resource_models(
    exact_df: pd.DataFrame,
    *,
    target_col: str = "T_exact",
    model_names: tuple[str, ...] = (
        "proxy_like_nm_q",
        "linear_n_m_q",
        "linear_n_m_q_nm",
        "interaction_n_m_q",
        "power_law_n_m_q",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit compact models that predict measured resource from N, M, and Q."""
    required = ["N", "M", "Q", target_col]
    if "split" not in exact_df.columns:
        raise KeyError("fit_exact_resource_models requires a 'split' column.")

    data = _positive_numeric_frame(exact_df, required)
    data = data[(data["N"] > 0) & (data["M"] > 0) & (data["Q"] > 0) & (data[target_col] > 0)].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    group_cols = [col for col in ["split", "N", "M", "Q"] if col in data.columns]
    data = (
        data.groupby(group_cols, as_index=False)
        .agg(**{target_col: (target_col, "median")})
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    train_df = data[data["split"].astype(str).eq("train")].copy()
    test_df = data[data["split"].astype(str).eq("test")].copy()
    if train_df.empty:
        train_df = data.copy()
    if test_df.empty:
        test_df = data.copy()

    summary_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model in model_names:
        fit = _fit_exact_resource_model(train_df, model, target_col=target_col)
        model_predictions = data.copy()
        model_predictions["model"] = model
        model_predictions["expression"] = fit["expression"]
        model_predictions["T_exact_pred"] = _predict_exact_resource_model(model_predictions, fit)
        model_predictions["factor_error"] = np.maximum(
            model_predictions["T_exact_pred"] / model_predictions[target_col],
            model_predictions[target_col] / model_predictions["T_exact_pred"],
        )
        prediction_frames.append(model_predictions)

        train_pred = _predict_exact_resource_model(train_df, fit)
        test_pred = _predict_exact_resource_model(test_df, fit)
        train_metrics = _exact_resource_model_metrics(train_df[target_col].to_numpy(dtype=float), train_pred)
        test_metrics = _exact_resource_model_metrics(test_df[target_col].to_numpy(dtype=float), test_pred)
        summary_rows.append(
            {
                "model": model,
                "expression": fit["expression"],
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["test_log10_rmse", "test_median_factor_error"],
        ascending=[True, True],
    ).reset_index(drop=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return summary_df, prediction_df


def _exact_group_filter(
    df: pd.DataFrame,
    *,
    spec: InstanceSpec,
    reps: int,
    strategy: str,
) -> pd.Series:
    return (
        df["graph_type"].astype(str).eq(str(spec.graph_type))
        & df["num_nodes"].astype(int).eq(int(spec.num_nodes))
        & df["instance"].astype(str).eq(str(spec.instance_id))
        & df["split"].astype(str).eq(str(spec.split or "train"))
        & df["p"].astype(int).eq(int(reps))
        & df["strategy"].astype(str).eq(str(strategy))
    )


def _exact_group_is_complete(
    df: pd.DataFrame,
    *,
    strategy: str,
    fa_method_name: str | None,
    pt_method_name: str | None,
    fa_n_values: list[int],
    fa_m_values: list[int],
    q_values: list[int],
) -> bool:
    if df.empty:
        return False

    q_expected = {int(q) for q in q_values}
    if strategy == fa_method_name:
        expected = {
            (int(n), int(m), int(q))
            for n in fa_n_values
            for m in fa_m_values
            for q in q_values
        }
        actual = {
            (int(n), int(m), int(q))
            for n, m, q in df.loc[:, ["N", "M", "Q"]].drop_duplicates().itertuples(index=False, name=None)
        }
        return expected.issubset(actual)

    if strategy == pt_method_name:
        actual_q = {int(q) for q in df["Q"].dropna().astype(int).tolist()}
        return q_expected.issubset(actual_q)

    return False


def _deduplicate_exact_points(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    dedupe_cols = [
        "graph_type",
        "num_nodes",
        "instance",
        "split",
        "strategy",
        "p",
        "N",
        "M",
        "Q",
    ]
    dedupe_cols = [col for col in dedupe_cols if col in df.columns]
    if not dedupe_cols:
        return df.reset_index(drop=True)

    return df.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)


def filter_pss_exact_points(
    exact_df: pd.DataFrame,
    *,
    train_specs: list[InstanceSpec],
    test_specs: list[InstanceSpec],
    p_values: list[int],
    fa_method_name: str | None,
    pt_method_name: str | None,
    fa_n_values: list[int],
    fa_m_values: list[int],
    q_values: list[int],
) -> pd.DataFrame:
    """Keep only exact rows that belong to the requested PSS campaign grid."""
    if exact_df.empty:
        return exact_df

    required_cols = {"graph_type", "num_nodes", "instance", "split", "p", "strategy", "Q"}
    missing_cols = required_cols.difference(exact_df.columns)
    if missing_cols:
        raise KeyError(f"Cannot filter PSS exact points; missing columns: {sorted(missing_cols)}")

    all_specs = list(train_specs) + list(test_specs)
    q_allowed = {int(q) for q in q_values}
    fa_nm_allowed = {(int(n), int(m)) for n in fa_n_values for m in fa_m_values}
    p_allowed = {int(p) for p in p_values}

    n_values = pd.to_numeric(exact_df.get("N", pd.Series(index=exact_df.index)), errors="coerce")
    m_values = pd.to_numeric(exact_df.get("M", pd.Series(index=exact_df.index)), errors="coerce")
    q_series = pd.to_numeric(exact_df["Q"], errors="coerce")
    p_series = pd.to_numeric(exact_df["p"], errors="coerce")

    mask = pd.Series(False, index=exact_df.index)
    for spec in all_specs:
        spec_mask = (
            exact_df["graph_type"].astype(str).eq(str(spec.graph_type))
            & exact_df["num_nodes"].astype(int).eq(int(spec.num_nodes))
            & exact_df["instance"].astype(str).eq(str(spec.instance_id))
            & exact_df["split"].astype(str).eq(str(spec.split or "train"))
            & p_series.isin(p_allowed)
        )
        if fa_method_name is not None:
            fa_mask = spec_mask & exact_df["strategy"].astype(str).eq(str(fa_method_name))
            fa_mask &= q_series.isin(q_allowed)
            fa_mask &= pd.Series(
                [
                    (int(n), int(m)) in fa_nm_allowed
                    if pd.notna(n) and pd.notna(m)
                    else False
                    for n, m in zip(n_values, m_values)
                ],
                index=exact_df.index,
            )
            mask |= fa_mask
        if pt_method_name is not None:
            pt_mask = spec_mask & exact_df["strategy"].astype(str).eq(str(pt_method_name))
            pt_mask &= q_series.isin(q_allowed)
            mask |= pt_mask

    return exact_df.loc[mask].reset_index(drop=True)


def _write_exact_cache_checkpoints(
    output_root: str | Path | None,
    exact_df: pd.DataFrame,
    *,
    fa_method_name: str | None,
    pt_method_name: str | None,
) -> None:
    if output_root is None or exact_df.empty:
        return

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    def _atomic_to_pickle(df: pd.DataFrame, path: Path) -> None:
        tmp_path = path.with_name(f".{path.name}.tmp")
        df.to_pickle(tmp_path)
        tmp_path.replace(path)

    if fa_method_name is not None:
        fa_df = exact_df[exact_df["strategy"].eq(fa_method_name)].copy()
        if not fa_df.empty:
            _atomic_to_pickle(fa_df, root / STRATEGY_RAW_POINTS_FILENAME)

    if pt_method_name is not None:
        pt_df = exact_df[exact_df["strategy"].eq(pt_method_name)].copy()
        if not pt_df.empty:
            _atomic_to_pickle(pt_df, root / TRANSFER_RAW_POINTS_FILENAME)


def generate_pss_exact_points(
    *,
    train_specs: list[InstanceSpec],
    test_specs: list[InstanceSpec],
    p_values: list[int],
    method_configs: dict[str, Path],
    main_repo: str | Path | None = None,
    pipeline_repo: str | Path | None = None,
    fa_method_name: str | None = "FA_PP_opt",
    pt_method_name: str | None = "PT_PP_AAA",
    fa_n_values: list[int] | None = None,
    fa_m_values: list[int] | None = None,
    q_values: list[int] | None = None,
    sample_config: dict[str, Any] | None = None,
    time_per_shot: float | dict[int, float] = 1.0,
    cobyla_overhead_c: float = 1.0,
    pt_transfer_cost: float = 0.0,
    output_root: str | Path | None = None,
    existing_exact_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate or reuse exact FA/PT PSS rows.

    This intentionally contains only the expensive exact-point loop and cache
    checkpointing. Frontier construction is separated so Nautilus can run exact
    point shards independently and merge them later.
    """
    fa_n_values = fa_n_values or [10, 20, 50, 75, 100]
    fa_m_values = fa_m_values or [10, 50, 100, 200, 500]
    q_values = q_values or [100, 200, 500, 1000, 5000]

    if fa_method_name is None and pt_method_name is None:
        raise ValueError("At least one of fa_method_name or pt_method_name must be provided.")

    all_specs = list(train_specs) + list(test_specs)
    cached_exact_df = _deduplicate_exact_points(existing_exact_df.copy()) if existing_exact_df is not None else pd.DataFrame()
    _pending_new_rows: list[pd.DataFrame] = []
    _CHECKPOINT_BATCH_SIZE = 10

    def _flush_exact_checkpoint() -> None:
        nonlocal cached_exact_df
        if not _pending_new_rows:
            return

        parts = [cached_exact_df] if not cached_exact_df.empty else []
        parts.extend(_pending_new_rows)
        cached_exact_df = _deduplicate_exact_points(pd.concat(parts, ignore_index=True))
        _pending_new_rows.clear()

        _write_exact_cache_checkpoints(
            output_root,
            cached_exact_df,
            fa_method_name=fa_method_name,
            pt_method_name=pt_method_name,
        )

    def _append_exact_checkpoint(df_new: pd.DataFrame) -> None:
        if df_new.empty:
            return
        _pending_new_rows.append(df_new)
        if len(_pending_new_rows) >= _CHECKPOINT_BATCH_SIZE:
            _flush_exact_checkpoint()

    for spec in all_specs:
        for reps in p_values:
            if pt_method_name is not None:
                cached_group = pd.DataFrame()
                if not cached_exact_df.empty:
                    cached_group = cached_exact_df.loc[
                        _exact_group_filter(
                            cached_exact_df,
                            spec=spec,
                            reps=reps,
                            strategy=pt_method_name,
                        )
                    ].copy()
                if not _exact_group_is_complete(
                    cached_group,
                    strategy=pt_method_name,
                    fa_method_name=fa_method_name,
                    pt_method_name=pt_method_name,
                    fa_n_values=fa_n_values,
                    fa_m_values=fa_m_values,
                    q_values=q_values,
                ):
                    _append_exact_checkpoint(
                        run_pt_pss_exact_points(
                            spec,
                            reps=reps,
                            q_values=q_values,
                            method_name=pt_method_name,
                            method_configs=method_configs,
                            main_repo=main_repo,
                            pipeline_repo=pipeline_repo,
                            sample_config=sample_config,
                            transfer_cost=pt_transfer_cost,
                            time_per_shot=time_per_shot,
                        )
                    )
            if fa_method_name is not None:
                cached_group = pd.DataFrame()
                if not cached_exact_df.empty:
                    cached_group = cached_exact_df.loc[
                        _exact_group_filter(
                            cached_exact_df,
                            spec=spec,
                            reps=reps,
                            strategy=fa_method_name,
                        )
                    ].copy()
                if not _exact_group_is_complete(
                    cached_group,
                    strategy=fa_method_name,
                    fa_method_name=fa_method_name,
                    pt_method_name=pt_method_name,
                    fa_n_values=fa_n_values,
                    fa_m_values=fa_m_values,
                    q_values=q_values,
                ):
                    _append_exact_checkpoint(
                        run_fa_pss_exact_points(
                            spec,
                            reps=reps,
                            n_values=fa_n_values,
                            m_values=fa_m_values,
                            q_values=q_values,
                            method_name=fa_method_name,
                            method_configs=method_configs,
                            main_repo=main_repo,
                            pipeline_repo=pipeline_repo,
                            sample_config=sample_config,
                            time_per_shot=time_per_shot,
                            cobyla_overhead_c=cobyla_overhead_c,
                        )
                    )

    _flush_exact_checkpoint()
    exact_df = _deduplicate_exact_points(cached_exact_df)
    exact_df = filter_pss_exact_points(
        exact_df,
        train_specs=train_specs,
        test_specs=test_specs,
        p_values=p_values,
        fa_method_name=fa_method_name,
        pt_method_name=pt_method_name,
        fa_n_values=fa_n_values,
        fa_m_values=fa_m_values,
        q_values=q_values,
    )
    if exact_df.empty:
        raise RuntimeError("No FA/PT PSS exact points were generated.")

    if output_root is not None:
        _write_exact_cache_checkpoints(
            output_root,
            exact_df,
            fa_method_name=fa_method_name,
            pt_method_name=pt_method_name,
        )

    return exact_df


def build_pss_frontier_outputs(
    exact_df: pd.DataFrame,
    *,
    train_specs: list[InstanceSpec],
    test_specs: list[InstanceSpec],
    t_grid: list[float] | None,
    t_grid_points: int | None = None,
    t_grid_scale: str = "linear",
    output_root: str | Path | None = None,
    fa_method_name: str | None = "FA_PP_opt",
    pt_method_name: str | None = "PT_PP_AAA",
) -> tuple[pd.DataFrame, dict[str, dict[int, Any]]]:
    """Build and optionally write budget-frontier outputs from exact rows."""
    exact_df = _deduplicate_exact_points(exact_df.copy())
    if exact_df.empty:
        raise RuntimeError("No FA/PT PSS exact points were provided.")

    if t_grid is None:
        t_grid = build_dense_budget_grid(
            exact_df,
            num_points=int(t_grid_points or 1000),
            resource_col="T_exact_proxy",
            scale=t_grid_scale,
        )
    # Drop per-shot count dictionaries before the frontier expansion, not after:
    # build_budget_frontier repeats each selected exact row across a dense T grid
    # (up to ~1000 points), so copying "counts" along for that expansion and then
    # dropping it afterward pays the copy/memory cost for nothing.
    frontier_df = build_budget_frontier(
        exact_df.drop(columns=["counts"], errors="ignore"),
        t_grid=t_grid,
        budget_col="T_exact_proxy",
        distance_scale=t_grid_scale,
    )
    frontier_df["train"] = frontier_df["split"].astype(str).map({"train": 1, "test": 0}).astype(int)
    strategy_codes, strategy_book = _encode_categories(frontier_df["strategy"])
    simulation_codes, simulation_book = _encode_categories(frontier_df["simulation_method"])
    frontier_df["strategy_code"] = strategy_codes.astype(float)
    frontier_df["simulation_code"] = simulation_codes.astype(float)

    codebooks = {"strategy": strategy_book, "simulation": simulation_book}

    if output_root is not None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        instance_specs_to_dataframe(train_specs).to_csv(output_root / "train_instance_specs.csv", index=False)
        instance_specs_to_dataframe(test_specs).to_csv(output_root / "test_instance_specs.csv", index=False)
        _write_exact_cache_checkpoints(
            output_root,
            exact_df,
            fa_method_name=fa_method_name,
            pt_method_name=pt_method_name,
        )
        frontier_df.to_pickle(output_root / "pss_exp_raw_frontier.pkl")
        frontier_df.to_csv(output_root / "pss_exp_raw_frontier.csv", index=False)
        coverage_df = summarize_frontier_instance_coverage(frontier_df)
        if not coverage_df.empty:
            coverage_df.to_csv(output_root / "frontier_instance_coverage.csv", index=False)

    return frontier_df, codebooks


def prepare_pss_exp_raw_dataset(
    *,
    train_specs: list[InstanceSpec],
    test_specs: list[InstanceSpec],
    p_values: list[int],
    t_grid: list[float] | None,
    method_configs: dict[str, Path],
    t_grid_points: int | None = None,
    t_grid_scale: str = "linear",
    main_repo: str | Path | None = None,
    pipeline_repo: str | Path | None = None,
    fa_method_name: str | None = "FA_PP_opt",
    pt_method_name: str | None = "PT_PP_AAA",
    fa_n_values: list[int] | None = None,
    fa_m_values: list[int] | None = None,
    q_values: list[int] | None = None,
    sample_config: dict[str, Any] | None = None,
    time_per_shot: float | dict[int, float] = 1.0,
    cobyla_overhead_c: float = 1.0,
    pt_transfer_cost: float = 0.0,
    output_root: str | Path | None = None,
    existing_exact_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[int, Any]]]:
    exact_df = generate_pss_exact_points(
        train_specs=train_specs,
        test_specs=test_specs,
        p_values=p_values,
        method_configs=method_configs,
        main_repo=main_repo,
        pipeline_repo=pipeline_repo,
        fa_method_name=fa_method_name,
        pt_method_name=pt_method_name,
        fa_n_values=fa_n_values,
        fa_m_values=fa_m_values,
        q_values=q_values,
        sample_config=sample_config,
        time_per_shot=time_per_shot,
        cobyla_overhead_c=cobyla_overhead_c,
        pt_transfer_cost=pt_transfer_cost,
        output_root=output_root,
        existing_exact_df=existing_exact_df,
    )
    frontier_df, codebooks = build_pss_frontier_outputs(
        exact_df,
        train_specs=train_specs,
        test_specs=test_specs,
        t_grid=t_grid,
        t_grid_points=t_grid_points,
        t_grid_scale=t_grid_scale,
        output_root=output_root,
        fa_method_name=fa_method_name,
        pt_method_name=pt_method_name,
    )
    return exact_df, frontier_df, codebooks


def build_strategy_budget_summary(
    frontier_df: pd.DataFrame,
    *,
    split: str = "train",
    response_col: str = "BestApproximationRatio",
) -> pd.DataFrame:
    if frontier_df.empty:
        return pd.DataFrame()

    split_df = frontier_df[frontier_df["split"].astype(str).eq(split)].copy()
    if split_df.empty:
        return pd.DataFrame()

    summary = (
        split_df.groupby(["strategy", "simulation_method", "p", "T"], as_index=False)
        .agg(
            response_mean=(response_col, "mean"),
            response_std=(response_col, "std"),
            response_min=(response_col, "min"),
            response_max=(response_col, "max"),
            N_mean=("N", "mean"),
            M_mean=("M", "mean"),
            Q_mean=("Q", "mean"),
            training_cost_mean=("training_cost", "mean"),
            sampling_cost_mean=("sampling_cost", "mean"),
            training_cost_proxy_mean=("training_cost_proxy", "mean"),
            sampling_cost_proxy_mean=("sampling_cost_proxy", "mean"),
            T_exact_mean=("selected_exact_T", "mean"),
            T_exact_proxy_mean=("selected_exact_T_proxy", "mean"),
            n_instances=("instance", "nunique"),
            n_candidates=("instance", "size"),
        )
        .sort_values(["T", "response_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    summary["response_std"] = summary["response_std"].fillna(0.0)
    return summary


def fit_recommended_recipe_curves(
    recipe_df: pd.DataFrame,
    *,
    t_grid: list[float] | None = None,
    fit_parameter_cols: tuple[str, ...] = ("N", "M", "Q"),
    resource_col: str = "resource",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if recipe_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    source = recipe_df.sort_values(resource_col).reset_index(drop=True).copy()
    if t_grid is None:
        target_t = source[resource_col].astype(float).to_numpy()
    else:
        target_t = np.array(sorted(float(val) for val in t_grid), dtype=float)
    source_t = source[resource_col].astype(float).to_numpy()
    support_mask = (target_t >= float(source_t.min())) & (target_t <= float(source_t.max()))
    target_t = target_t[support_mask]
    if target_t.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    fitted = pd.DataFrame({resource_col: target_t})
    source_idx = np.searchsorted(source_t, target_t, side="right") - 1
    source_idx = np.clip(source_idx, 0, len(source) - 1)
    model_rows: list[dict[str, Any]] = []

    for col in source.columns:
        if col == resource_col:
            continue
        if col in fit_parameter_cols:
            coeffs = _fit_log_polynomial(
                source_t,
                source[col].astype(float).to_numpy(),
                degree=2,
            )
            if coeffs is None:
                fitted[col] = np.nan
                continue
            fitted[col] = _predict_log_polynomial(
                target_t,
                coeffs,
                y_min=float(source[col].min()),
                y_max=float(source[col].max()),
            )
            model_rows.append(
                {
                    "parameter": col,
                    "expression": _expression_text(col, coeffs),
                    **{f"coeff_{idx}": float(val) for idx, val in enumerate(coeffs)},
                }
            )
        else:
            fitted[col] = source.iloc[source_idx][col].to_numpy()

    return fitted, pd.DataFrame(model_rows)


def _fit_log_polynomial(
    x_values,
    y_values,
    *,
    degree: int = 2,
) -> np.ndarray | None:
    x = pd.to_numeric(pd.Series(x_values), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y_values), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return None

    actual_degree = min(int(degree), max(len(np.unique(x)) - 1, 0))
    if actual_degree == 0:
        coeff_high_to_low = np.array([float(np.log(np.nanmedian(y)))])
    else:
        coeff_high_to_low = np.polyfit(np.log(x), np.log(y), deg=actual_degree)
    return coeff_high_to_low[::-1]


def _predict_log_polynomial(
    t_values,
    coeff_low_to_high: np.ndarray,
    *,
    y_min: float | None = None,
    y_max: float | None = None,
) -> np.ndarray:
    t = pd.to_numeric(pd.Series(t_values), errors="coerce").to_numpy(dtype=float)
    pred = np.full(len(t), np.nan, dtype=float)
    mask = np.isfinite(t) & (t > 0)
    if mask.any():
        log_t = np.log(t[mask])
        log_y = np.zeros(mask.sum(), dtype=float)
        for power, coeff in enumerate(coeff_low_to_high):
            log_y += float(coeff) * (log_t ** power)
        pred[mask] = np.exp(log_y)
    if y_min is not None or y_max is not None:
        pred = np.clip(
            pred,
            y_min if y_min is not None else -np.inf,
            y_max if y_max is not None else np.inf,
        )
    return pred


def _expression_text(param_name: str, coeff_low_to_high: np.ndarray, *, precision: int = 4) -> str:
    pieces = []
    for power, coeff in enumerate(coeff_low_to_high):
        coeff_text = f"{float(coeff):.{precision}g}"
        if power == 0:
            pieces.append(coeff_text)
        elif power == 1:
            pieces.append(f"{coeff_text} log(T)")
        else:
            pieces.append(f"{coeff_text} log(T)^{power}")
    return f"{param_name}(T) = exp(" + " + ".join(pieces).replace("+ -", "- ") + ")"


def snap_actionable_fit_to_feasible_grid(
    exact_df: pd.DataFrame,
    actionable_fit_df: pd.DataFrame,
    *,
    parameter_fit_cols: tuple[str, ...] = ("N_fit", "M_fit", "Q_fit"),
    budget_col: str = "T_exact_proxy",
    distance_scale: str = "log",
    resource_col: str = "T",
) -> pd.DataFrame:
    if exact_df.empty or actionable_fit_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped_exact = exact_df.groupby(["strategy", "simulation_method", "p"], dropna=False)

    for row in actionable_fit_df.to_dict(orient="records"):
        strategy = row.get("strategy")
        simulation_method = row.get("simulation_method")
        p_value = row.get("p")
        target_t = float(row[resource_col])
        key = (strategy, simulation_method, p_value)
        if key not in grouped_exact.groups:
            continue

        candidates = grouped_exact.get_group(key)
        budget_distances = _resource_distance(
            candidates[budget_col],
            target_t,
            scale=distance_scale,
        )
        if not np.isfinite(budget_distances).any():
            continue
        min_budget_distance = float(np.nanmin(budget_distances))
        feasible = candidates.loc[
            np.isfinite(budget_distances)
            & np.isclose(
                budget_distances,
                min_budget_distance,
                rtol=1e-9,
                atol=1e-12,
            )
        ].copy()
        if feasible.empty:
            continue

        distance = np.zeros(len(feasible), dtype=float)
        for col in parameter_fit_cols:
            if col not in row:
                continue
            exact_col = col.replace("_fit", "")
            if exact_col not in feasible.columns:
                exact_col = exact_col.replace("_mean", "")
            if exact_col not in feasible.columns:
                continue
            scale = max(float(feasible[exact_col].max()) - float(feasible[exact_col].min()), 1.0)
            distance += ((feasible[exact_col].astype(float) - float(row[col])) / scale) ** 2

        feasible = feasible.assign(_fit_distance=distance)
        _secondary_sort = [c for c in [budget_col, "Q"] if c in feasible.columns]
        best_row = feasible.sort_values(
            ["_fit_distance", "BestApproximationRatio"] + _secondary_sort,
            ascending=[True, False] + [True] * len(_secondary_sort),
        ).iloc[0].to_dict()
        best_row["T"] = target_t
        rows.append(best_row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def resource_metric_column(df: pd.DataFrame, resource_col: str) -> pd.Series:
    mean_col = f"Key=Mean{resource_col}"
    if mean_col in df.columns:
        return df[mean_col]
    if "Key=MeanTime" in df.columns:
        return df["Key=MeanTime"]
    if resource_col in df.columns:
        return df[resource_col]
    raise KeyError(f"Could not resolve a resource metric column for '{resource_col}'.")


def assign_deterministic_train_split(
    interp_results: pd.DataFrame,
    exp_raw_df: pd.DataFrame,
    *,
    instance_col: str = "instance",
) -> pd.DataFrame:
    split_map = (
        exp_raw_df[[instance_col, "train", "split"]]
        .drop_duplicates()
        .rename(columns={instance_col: "_instance_key"})
    )
    merged = interp_results.copy()
    merged["_instance_key"] = merged[instance_col].astype(str)
    merged = merged.drop(columns=["train", "split"], errors="ignore")
    merged = merged.merge(split_map, on="_instance_key", how="left")
    merged = merged.drop(columns=["_instance_key"])
    return merged


def snap_resources_to_grid(
    df: pd.DataFrame,
    grid: list[float] | np.ndarray,
    *,
    resource_col: str = "resource",
) -> pd.DataFrame:
    if df.empty or resource_col not in df.columns:
        return df

    snapped = df.copy()
    grid_arr = np.asarray(sorted({float(val) for val in grid}), dtype=float)
    if grid_arr.size == 0:
        return snapped

    values = snapped[resource_col].astype(float).to_numpy()
    insert_idx = np.searchsorted(grid_arr, values)
    insert_idx = np.clip(insert_idx, 0, len(grid_arr) - 1)
    left_idx = np.clip(insert_idx - 1, 0, len(grid_arr) - 1)
    right_idx = insert_idx
    left_vals = grid_arr[left_idx]
    right_vals = grid_arr[right_idx]
    choose_right = np.abs(right_vals - values) < np.abs(left_vals - values)
    snapped_vals = np.where(choose_right, right_vals, left_vals)
    snapped[resource_col] = snapped_vals.astype(float)
    return snapped


def align_projection_resources(
    sb: Any,
    *,
    shared_grid: list[float] | np.ndarray,
    resource_col: str = "resource",
) -> None:
    if getattr(sb, "training_stats", None) is not None:
        sb.training_stats = snap_resources_to_grid(
            sb.training_stats,
            shared_grid,
            resource_col=resource_col,
        )
    if getattr(sb, "testing_stats", None) is not None:
        sb.testing_stats = snap_resources_to_grid(
            sb.testing_stats,
            shared_grid,
            resource_col=resource_col,
        )


def run_stochastic_benchmark_pss(
    exp_raw_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    codebooks: dict[str, dict[int, Any]],
    parameter_names: list[str] | None = None,
    response_col: str = "BestApproximationRatio",
    resource_col: str = "T",
    response_key: str = "Response",
    bootstrap_range: range | None = None,
    train_test_split: float = 0.5,
    clear_checkpoints: bool = True,
    resource_match_bins: int | None = 150,
) -> dict[str, Any]:
    """Run the stochastic-benchmark PSS pipeline (bootstrap, interpolate, stats,
    virtual-best, training/projection) over campaign exp_raw rows.

    resource_match_bins : int | None
        Snaps ``sb.interp_results``'s resource column onto a ``resource_match_bins``
        -point log/linear grid before the recipe/evaluate matching in
        ``run_ProjectionExperiment`` runs. Defaults to 150 (on by default), which
        changes campaign results relative to not snapping: train and test
        instances are interpolated onto independent resource grids, and
        ``training.evaluate_single`` (core package) matches resource values by
        exact float equality, so without snapping the train/test overlap is
        small and effectively arbitrary from run to run. This is a workaround
        for that core-package matching behavior (tracked in issue #85), not a
        fix to it; every other consumer of ``evaluate_single`` still has the
        same brittleness. Pass ``None`` or ``0`` to disable snapping and match
        on the raw shared_grid instead.

        Note separately that bootstrap resampling below is unseeded (issue
        #86), so even with snapping enabled the number of surviving
        actionable-fit rows can still vary between otherwise-identical runs.
    """
    if exp_raw_df.empty:
        raise ValueError("exp_raw_df is empty.")

    exp_raw_df = normalize_sb_compatible_dtypes(exp_raw_df)
    output_dir = Path(output_dir)
    written_files = persist_campaign_exp_raw(exp_raw_df, output_dir)
    parameter_names = parameter_names or ["strategy_code", "simulation_code", "N", "M", "Q", "p"]

    sb, bs_iter = setup_stochastic_benchmark_campaign(
        output_dir,
        parameter_names=parameter_names,
        response_col=response_col,
        resource_col=resource_col,
        bootstrap_range=bootstrap_range,
        response_key=response_key,
        # `instance` is already a stochastic-benchmark grouping key. On newer
        # pandas versions the grouped frame passed into BootstrapSingle may not
        # retain that grouping column as a normal dataframe column, which causes
        # a KeyError if we also request it via keep_cols.
        keep_cols=["train", "split", "strategy", "simulation_method"],
    )

    if clear_checkpoints:
        checkpoints_dir = output_dir / "checkpoints"
        if checkpoints_dir.exists():
            for stale_path in checkpoints_dir.glob("*.pkl"):
                stale_path.unlink()
        for stale_name in (
            "VirtualBest_test.pkl",
            "VirtualBest_train.pkl",
            "Projection_from=TrainingResults.pkl",
            "BestRecommended_train.pkl",
        ):
            stale_path = checkpoints_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

    import interpolate  # pylint: disable=import-outside-toplevel
    import stats  # pylint: disable=import-outside-toplevel
    import training  # pylint: disable=import-outside-toplevel

    shared_grid = sorted(exp_raw_df[resource_col].astype(float).dropna().unique())
    i_params = interpolate.InterpolationParameters(
        lambda df: resource_metric_column(df, resource_col),
        parameters=parameter_names,
        resource_value_type="manual",
        resource_values=shared_grid,
    )
    st_params = stats.StatsParameters(
        metrics=["Response", "PerfRatio", "MeanTime", "SuccProb"],
        stats_measures=[stats.Mean()],
    )

    sb.run_Bootstrap(bs_iter, group_name_fcn=raw_results_group_name)
    if isinstance(sb.bs_results, list) and sb.bs_results and isinstance(sb.bs_results[0], (str, Path)):
        normalize_sb_pickle_files(sb.bs_results)
    sb.run_Interpolate(i_params)
    sb.run_Stats(st_params, train_test_split)
    sb.interp_results = assign_deterministic_train_split(sb.interp_results, exp_raw_df)
    sb.interp_results.to_pickle(sb.here.interpolate)

    # `training.evaluate_single` (core stochastic-benchmark package) matches a
    # train-derived recipe's resource value against test-instance resource
    # values via exact floating-point equality. Train and test instances land
    # on different points of the raw ~1000-point shared_grid independently
    # (they're different instances), so the exact-match overlap between the
    # two is small and effectively random from run to run (bootstrap
    # resampling below isn't seeded) -- the same campaign can yield wildly
    # different numbers of surviving actionable-fit rows on separate runs.
    # Snapping onto a coarser shared bin grid here, before the recipe/evaluate
    # matching happens, makes train- and test-side resource values collide far
    # more often -- without touching the shared training.py matching logic.
    if resource_match_bins and len(shared_grid) > resource_match_bins:
        _lo, _hi = float(shared_grid[0]), float(shared_grid[-1])
        _matching_grid = (
            np.geomspace(_lo, _hi, resource_match_bins)
            if _lo > 0
            else np.linspace(_lo, _hi, resource_match_bins)
        )
        sb.interp_results = snap_resources_to_grid(
            sb.interp_results, _matching_grid, resource_col="resource"
        )

    sb.training_stats = None
    sb.testing_stats = None
    sb.populate_training_stats()
    align_projection_resources(sb, shared_grid=shared_grid, resource_col="resource")
    if not hasattr(sb, "baseline"):
        sb.baseline = SimpleNamespace(recalibrate=lambda _df: None)
    sb.run_baseline()
    sb.run_ProjectionExperiment("TrainingResults", None, None)

    virtual_best_params_df, virtual_best_eval_df = sb.baseline.evaluate()
    projection_experiment = next(exp for exp in sb.experiments if exp.name == "Projection from TrainingResults")
    projection_params_df, projection_eval_df = projection_experiment.evaluate()
    vb_train_lookup_df = build_virtual_best_lookup_summary(
        projection_experiment.vb_train.copy(),
        sb.parameter_names,
        codebooks,
        response_key=sb.response_key,
    )
    recipe_train_raw_df = projection_experiment.recipe.copy()
    recipe_train_df = decode_ws_params(recipe_train_raw_df.copy(), codebooks)
    fitted_recipe_raw_df, fitted_recipe_model_df = fit_recommended_recipe_curves(
        recipe_train_raw_df,
        t_grid=sorted(recipe_train_raw_df["resource"].astype(float).dropna().unique()),
        fit_parameter_cols=("N", "M", "Q"),
        resource_col="resource",
    )
    fitted_recipe_df = decode_ws_params(fitted_recipe_raw_df.copy(), codebooks)

    # Snap the continuous polynomial fit back to the nearest feasible (N, M, Q) grid point
    # so the final prescription is always an exactly-computable parameter set.
    _snap_budget_col = next(
        (c for c in ("T_exact_proxy", "selected_exact_T_proxy") if c in exp_raw_df.columns),
        None,
    )
    snapped_recipe_raw_df: pd.DataFrame = pd.DataFrame()
    if _snap_budget_col is not None and not fitted_recipe_df.empty:
        # Use the decoded recipe (string strategy/simulation_method) so the groupby
        # keys match the string columns in exp_raw_df.
        _snapped = snap_actionable_fit_to_feasible_grid(
            exp_raw_df,
            fitted_recipe_df,
            parameter_fit_cols=("N", "M", "Q"),
            budget_col=_snap_budget_col,
            distance_scale="log",
            resource_col="resource",
        )
        if not _snapped.empty:
            _snapped = _snapped.copy()
            _snapped["resource"] = _snapped["T"]
            snapped_recipe_raw_df = _snapped
    snapped_recipe_df = decode_ws_params(snapped_recipe_raw_df.copy(), codebooks) if not snapped_recipe_raw_df.empty else pd.DataFrame()

    train_results = sb.interp_results[sb.interp_results["train"] == 1].copy()
    train_rec_params = training.evaluate(
        train_results,
        recipe_train_raw_df,
        training.scaled_distance,
        parameter_names=sb.parameter_names,
        resource_col="resource",
        group_on=sb.instance_cols,
    )
    actionable_train_eval_df = _build_response_summary_from_rec_params(
        train_rec_params,
        sb.parameter_names,
        codebooks,
        response_key=sb.response_key,
    )
    train_fitted_rec_params = training.evaluate(
        train_results,
        fitted_recipe_raw_df,
        training.scaled_distance,
        parameter_names=sb.parameter_names,
        resource_col="resource",
        group_on=sb.instance_cols,
    )
    fitted_train_eval_df = _build_response_summary_from_rec_params(
        train_fitted_rec_params,
        sb.parameter_names,
        codebooks,
        response_key=sb.response_key,
    )
    test_results = sb.interp_results[sb.interp_results["train"] == 0].copy()
    test_fitted_rec_params = training.evaluate(
        test_results,
        fitted_recipe_raw_df,
        training.scaled_distance,
        parameter_names=sb.parameter_names,
        resource_col="resource",
        group_on=sb.instance_cols,
    )
    fitted_test_eval_df = _build_response_summary_from_rec_params(
        test_fitted_rec_params,
        sb.parameter_names,
        codebooks,
        response_key=sb.response_key,
    )

    snapped_train_eval_df: pd.DataFrame = pd.DataFrame()
    snapped_test_eval_df: pd.DataFrame = pd.DataFrame()
    if not snapped_recipe_raw_df.empty:
        snapped_train_rec_params = training.evaluate(
            train_results,
            snapped_recipe_raw_df,
            training.scaled_distance,
            parameter_names=sb.parameter_names,
            resource_col="resource",
            group_on=sb.instance_cols,
        )
        snapped_train_eval_df = _build_response_summary_from_rec_params(
            snapped_train_rec_params,
            sb.parameter_names,
            codebooks,
            response_key=sb.response_key,
        )
        snapped_test_rec_params = training.evaluate(
            test_results,
            snapped_recipe_raw_df,
            training.scaled_distance,
            parameter_names=sb.parameter_names,
            resource_col="resource",
            group_on=sb.instance_cols,
        )
        snapped_test_eval_df = _build_response_summary_from_rec_params(
            snapped_test_rec_params,
            sb.parameter_names,
            codebooks,
            response_key=sb.response_key,
        )

    if "resource" not in virtual_best_params_df.columns:
        virtual_best_params_df = virtual_best_params_df.reset_index()
    if "resource" not in virtual_best_eval_df.columns:
        virtual_best_eval_df = virtual_best_eval_df.reset_index()
    if "resource" not in projection_params_df.columns:
        projection_params_df = projection_params_df.reset_index()
    if "resource" not in projection_eval_df.columns:
        projection_eval_df = projection_eval_df.reset_index()

    virtual_best_df = build_virtual_best_summary(sb, codebooks)
    if virtual_best_df.empty:
        virtual_best_df = decode_ws_params(
            virtual_best_params_df.merge(virtual_best_eval_df, on="resource", how="left"),
            codebooks,
        )
    projection_summary_df = build_projection_summary(sb, codebooks)
    if projection_summary_df.empty:
        projection_summary_df = decode_ws_params(
            projection_params_df.merge(projection_eval_df, on="resource", how="outer"),
            codebooks,
        )
    projection_summary_df = projection_summary_df.sort_values("resource").reset_index(drop=True)
    window_sticker_summary_df = projection_summary_df[
        [
            "resource",
            "strategy",
            "simulation_method",
            "N",
            "M",
            "Q",
            "p",
            "response",
            "response_lower",
            "response_upper",
        ]
    ].copy()

    virtual_best_df.to_csv(output_dir.parent / "virtual_best_summary.csv", index=False)
    projection_summary_df.to_csv(output_dir.parent / "projection_summary.csv", index=False)
    window_sticker_summary_df.to_csv(output_dir.parent / "window_sticker_summary.csv", index=False)
    vb_train_lookup_df.to_csv(output_dir.parent / "virtual_best_lookup_train.csv", index=False)
    recipe_train_df.to_csv(output_dir.parent / "actionable_recipe_train.csv", index=False)
    actionable_train_eval_df.to_csv(output_dir.parent / "actionable_projection_train.csv", index=False)
    fitted_recipe_df.to_csv(output_dir.parent / "fitted_actionable_recipe_train.csv", index=False)
    fitted_recipe_model_df.to_csv(output_dir.parent / "fitted_actionable_recipe_model.csv", index=False)
    fitted_train_eval_df.to_csv(output_dir.parent / "fitted_actionable_projection_train.csv", index=False)
    fitted_test_eval_df.to_csv(output_dir.parent / "fitted_actionable_projection_test.csv", index=False)
    if not snapped_recipe_df.empty:
        snapped_recipe_df.to_csv(output_dir.parent / "snapped_actionable_recipe_train.csv", index=False)
    if not snapped_train_eval_df.empty:
        snapped_train_eval_df.to_csv(output_dir.parent / "snapped_actionable_projection_train.csv", index=False)
    if not snapped_test_eval_df.empty:
        snapped_test_eval_df.to_csv(output_dir.parent / "snapped_actionable_projection_test.csv", index=False)
    pd.DataFrame({"path": [str(path) for path in written_files]}).to_csv(
        output_dir.parent / "campaign_exp_raw_manifest.csv",
        index=False,
    )

    return {
        "sb": sb,
        "written_files": written_files,
        "virtual_best_df": virtual_best_df,
        "projection_summary_df": projection_summary_df,
        "window_sticker_summary_df": window_sticker_summary_df,
        "vb_train_lookup_df": vb_train_lookup_df,
        "recipe_train_df": recipe_train_df,
        "actionable_train_eval_df": actionable_train_eval_df,
        "fitted_recipe_df": fitted_recipe_df,
        "fitted_recipe_model_df": fitted_recipe_model_df,
        "fitted_train_eval_df": fitted_train_eval_df,
        "fitted_test_eval_df": fitted_test_eval_df,
        "snapped_recipe_df": snapped_recipe_df,
        "snapped_train_eval_df": snapped_train_eval_df,
        "snapped_test_eval_df": snapped_test_eval_df,
    }


def _encode_categories(series: pd.Series) -> tuple[pd.Series, dict[int, Any]]:
    categories = sorted(series.dropna().astype(str).unique())
    codebook = {idx: value for idx, value in enumerate(categories)}
    reverse = {value: idx for idx, value in codebook.items()}
    return series.astype(str).map(reverse), codebook


def normalize_sb_compatible_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas extension dtypes into forms stochastic-benchmark expects."""
    normalized = df.copy()
    for col in normalized.columns:
        if isinstance(normalized[col].dtype, pd.StringDtype):
            normalized[col] = normalized[col].astype(object)
    return normalized


def normalize_sb_pickle_files(paths: list[str | Path]) -> None:
    """Rewrite pickled dataframes with SB-compatible dtypes.

    Pickle has no metadata-only read, so inspecting dtypes still requires loading
    each file, but files that are already compatible are left untouched instead of
    being unconditionally copied and rewritten to disk.
    """
    for path in paths:
        df = pd.read_pickle(path)
        if not any(isinstance(dtype, pd.StringDtype) for dtype in df.dtypes):
            continue
        normalized = normalize_sb_compatible_dtypes(df)
        normalized.to_pickle(path)


def persist_campaign_exp_raw(
    campaign_df: pd.DataFrame,
    output_dir: str | Path,
    instance_col: str = "instance",
    depth_col: str = "p",
) -> list[Path]:
    campaign_df = normalize_sb_compatible_dtypes(campaign_df)
    output_dir = Path(output_dir)
    raw_dir = output_dir / "exp_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for (instance, depth), group in campaign_df.groupby([instance_col, depth_col], dropna=False):
        filename = f"raw_results_inst={instance}_depth={depth}.pkl"
        path = raw_dir / filename
        group.to_pickle(path)
        written.append(path)
    return written


def raw_results_group_name(raw_filename: str | Path) -> str:
    raw_path = Path(raw_filename)
    stem = raw_path.stem
    return stem.replace("raw_results_", "", 1)


def setup_stochastic_benchmark_campaign(
    here: str | Path,
    parameter_names: list[str] | None = None,
    *,
    response_col: str = "BestApproximationRatio",
    resource_col: str = "T",
    bootstrap_range: range | None = None,
    response_key: str = "Response",
    keep_cols: list[str] | None = None,
):
    import bootstrap
    import success_metrics
    import stochastic_benchmark as SB

    parameter_names = parameter_names or ["strategy_code", "simulation_code", "N", "M", "Q", "p"]
    sb = SB.stochastic_benchmark(
        parameter_names=parameter_names,
        response_key=response_key,
        here=str(here),
        instance_cols=["instance"],
        response_dir=1,
        smooth=True,
    )

    shared_args = {
        "response_col": response_col,
        "resource_col": resource_col,
        "response_dir": 1,
        "confidence_level": 68,
        "random_value": 0.0,
    }

    metric_args = {
        "Response": {"opt_sense": 1},
        "SuccessProb": {"gap": 0.1, "response_dir": 1},
        "RTT": {
            "fail_value": np.nan,
            "RTT_factor": 1.0,
            "gap": 0.1,
            "s": 0.99,
        },
    }

    def update_rules(self, df):
        self.shared_args["best_value"] = float(df[self.shared_args["response_col"]].max())
        self.metric_args["RTT"]["RTT_factor"] = float(df[self.shared_args["resource_col"]].iloc[0])

    success_metric_refs = [
        success_metrics.Response,
        success_metrics.PerfRatio,
        success_metrics.SuccessProb,
        success_metrics.Resource,
    ]

    bs_params = bootstrap.BootstrapParameters(
        shared_args=shared_args,
        update_rule=update_rules,
        agg=None,
        metric_args=metric_args,
        success_metrics=success_metric_refs,
        keep_cols=list(keep_cols or []),
    )
    bs_iter = bootstrap.BSParams_range_iter()(bs_params, bootstrap_range or range(10, 51, 10))
    return sb, bs_iter


def decode_ws_params(df: pd.DataFrame, codebooks: dict[str, dict[int, Any]]) -> pd.DataFrame:
    decoded = df.copy()
    if "strategy_code" in decoded.columns and "strategy" in codebooks:
        reverse = {int(code): value for code, value in codebooks["strategy"].items()}
        decoded["strategy"] = decoded["strategy_code"].round().astype(int).map(reverse)
    if "simulation_code" in decoded.columns and "simulation" in codebooks:
        reverse = {int(code): value for code, value in codebooks["simulation"].items()}
        decoded["simulation_method"] = decoded["simulation_code"].round().astype(int).map(reverse)
    return decoded


def _build_response_summary_from_rec_params(
    rec_params: pd.DataFrame,
    parameter_names: list[str],
    codebooks: dict[str, dict[int, Any]],
    *,
    response_key: str,
) -> pd.DataFrame:
    import names

    if rec_params is None or rec_params.empty:
        return pd.DataFrame()

    params_df = rec_params.loc[:, ["resource"] + parameter_names].copy()
    params_df = params_df.groupby("resource").mean(numeric_only=True).reset_index()

    base = names.param2filename({"Key": response_key}, "")
    CIlower = names.param2filename({"Key": response_key, "ConfInt": "lower"}, "")
    CIupper = names.param2filename({"Key": response_key, "ConfInt": "upper"}, "")

    if base not in rec_params.columns:
        raise KeyError(f"Expected response column '{base}' in recommended-parameter dataframe.")

    eval_df = rec_params.copy()
    eval_df["response"] = eval_df[base]
    grouped_response = (
        eval_df.loc[:, ["resource", "response"]]
        .groupby("resource")["response"]
        .agg(response="mean", response_std="std", n_instances="count")
        .reset_index()
    )
    sem = grouped_response["response_std"] / np.sqrt(grouped_response["n_instances"].clip(lower=1))
    ci_half_width = 1.96 * sem.fillna(0.0)
    grouped_response["response_lower"] = grouped_response["response"] - ci_half_width
    grouped_response["response_upper"] = grouped_response["response"] + ci_half_width
    if CIlower in eval_df.columns and CIupper in eval_df.columns:
        native_ci = (
            eval_df.loc[:, ["resource", CIlower, CIupper]]
            .groupby("resource")
            .mean(numeric_only=True)
            .reset_index()
        )
        native_ci["_native_ci_nonzero"] = (native_ci[CIupper] - native_ci[CIlower]).abs() > 1e-12
        if native_ci["_native_ci_nonzero"].any():
            grouped_response = grouped_response.merge(native_ci, on="resource", how="left")
            has_nonzero_native_ci = grouped_response["_native_ci_nonzero"].fillna(False)
            grouped_response.loc[has_nonzero_native_ci, "response_lower"] = grouped_response.loc[
                has_nonzero_native_ci, CIlower
            ]
            grouped_response.loc[has_nonzero_native_ci, "response_upper"] = grouped_response.loc[
                has_nonzero_native_ci, CIupper
            ]
            grouped_response = grouped_response.drop(columns=[CIlower, CIupper, "_native_ci_nonzero"])

    summary = decode_ws_params(params_df.merge(grouped_response, on="resource", how="outer"), codebooks)
    return summary.sort_values("resource").reset_index(drop=True)


def build_virtual_best_lookup_summary(
    vb_train: pd.DataFrame,
    parameter_names: list[str],
    codebooks: dict[str, dict[int, Any]],
    *,
    response_key: str,
) -> pd.DataFrame:
    import names

    if vb_train is None or vb_train.empty:
        return pd.DataFrame()

    base = names.param2filename({"Key": response_key}, "")
    if base not in vb_train.columns:
        raise KeyError(f"Expected response column '{base}' in virtual-best dataframe.")

    params_df = (
        vb_train.loc[:, ["resource"] + parameter_names]
        .groupby("resource")
        .mean(numeric_only=True)
        .reset_index()
    )
    response_df = (
        vb_train.groupby("resource")
        .agg(
            response=(base, "mean"),
            n_instances=("instance", "nunique"),
            n_virtual_best_rows=("instance", "size"),
        )
        .reset_index()
    )
    summary = decode_ws_params(params_df.merge(response_df, on="resource", how="left"), codebooks)
    return summary.sort_values("resource").reset_index(drop=True)


def build_virtual_best_summary(
    sb,
    codebooks: dict[str, dict[int, Any]],
) -> pd.DataFrame:
    baseline = getattr(sb, "baseline", None)
    if baseline is None or not hasattr(baseline, "rec_params"):
        return pd.DataFrame()

    return _build_response_summary_from_rec_params(
        baseline.rec_params,
        sb.parameter_names,
        codebooks,
        response_key=sb.response_key,
    )


def build_projection_summary(
    sb,
    codebooks: dict[str, dict[int, Any]],
    *,
    experiment_name: str = "Projection from TrainingResults",
) -> pd.DataFrame:
    experiment = next((exp for exp in getattr(sb, "experiments", []) if exp.name == experiment_name), None)
    if experiment is None:
        return pd.DataFrame()
    if not hasattr(experiment, "rec_params"):
        return pd.DataFrame()

    return _build_response_summary_from_rec_params(
        experiment.rec_params,
        sb.parameter_names,
        codebooks,
        response_key=sb.response_key,
    )

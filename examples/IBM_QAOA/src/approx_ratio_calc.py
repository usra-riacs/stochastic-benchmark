from pathlib import Path
import json
import numpy as np
from collections import Counter

def get_minmax(minmax_path: str, graph_type: str, instance: str, num_nodes: str, 
                ER_probability: str = None, swap_layers: str = None, degree: str = None):
    """Locate the min/max-cut JSON file for a given instance. This is specific to the IBM QAOA data 
    for parameter setting strategies

    Parameters
    ----------
    minmax_path : str
        Base directory containing per-graph-type subfolders.
    graph_type : str
        Graph family identifier.
    instance : str
        Instance identifier used in the filename pattern.
    num_nodes : str
        Number of nodes encoded in the filename.
    ER_probability : str, optional
        Erdos-Renyi probability suffix used for `graph_type="erdos_renyi"`.
    swap_layers : str, optional
        Swap-layer count used for `graph_type="line_to_full"`.
    degree : str, optional
        Degree used for `graph_type="random_regular"`.

    Returns
    -------
    pathlib.Path
        Path to the unique matching min/max-cut JSON file.

    Raises
    ------
    ValueError
        If `graph_type` is not supported.
    FileNotFoundError
        If no matching file is found.
    RuntimeError
        If multiple matching files are found.
    """

    if graph_type == "heavy_hex":
        instance_path = f"{graph_type}/{instance}*heavyhex_{num_nodes}nodes*.json"
    elif graph_type == "erdos_renyi":
        instance_path = f"{graph_type}/{instance}_{num_nodes}nodes_erdosrenyi{ER_probability}percent*.json"
    elif graph_type == "line_to_full":
        instance_path = f"{graph_type}/{instance}_{num_nodes}nodes_{swap_layers}swap_layers*.json"
    elif graph_type == "random_regular":
        instance_path = f"{graph_type}/{instance}_{num_nodes}nodes_random{degree}regular*.json"
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    minmax_instance_paths = list(Path(minmax_path).glob(instance_path))

    if len(minmax_instance_paths) == 0:
        raise FileNotFoundError(f"No minmax file for instance {instance}")
    if len(minmax_instance_paths) > 1:
        raise RuntimeError(f"Multiple minmax files for instance {instance}")

    return minmax_instance_paths[0]

def extract_minmax_args(minmax_instance_paths: Path):

    """Extract min-cut, max-cut, and sum-of-weights values from a JSON file.

    Parameters
    ----------
    minmax_instance_paths : pathlib.Path
        Path to a min/max-cut JSON file.

    Returns
    -------
    tuple of (float, float, float) or None
        `(min_cut, max_cut, sum_of_weights)` if all keys exist; otherwise None.
    """

    with minmax_instance_paths.open("r") as f:
        content = json.load(f)

    if "min_cut" not in content or "max_cut" not in content or "sum_of_weights" not in content:
        return None
    min_cut = content["min_cut"]
    max_cut = content["max_cut"]
    sum_weights = content["sum_of_weights"]

    return min_cut, max_cut, sum_weights

def maxcut_approximation_ratio(
    min_cut: float, max_cut: float, sum_weights: float, energy: float
) -> float:
    """Compute the MaxCut approximation ratio for a given energy.

    Parameters
    ----------
    min_cut : float
        Minimum cut value for the instance.
    max_cut : float
        Maximum cut value for the instance.
    sum_weights : float
        Sum of edge weights for the instance.
    energy : float
        Energy returned by the MaximumCut objective.

    Returns
    -------
    float
        Approximation ratio computed from the implied cut value.
    """
    
    cut_val = energy + 0.5 * sum_weights

    return (cut_val - min_cut) / (max_cut - min_cut)


def load_maxcut_instance_context(graph_file: str | Path) -> dict[str, np.ndarray | float]:
    """Load the graph data needed to evaluate MaxCut bitstrings.

    Parameters
    ----------
    graph_file : str or pathlib.Path
        Path to the problem-instance JSON file.

    Returns
    -------
    dict
        Preprocessed edge endpoints, weights, and total edge-weight sum.
    """
    graph_path = Path(graph_file)

    with graph_path.open("r") as f:
        inst_json = json.load(f)

    edges = inst_json["edge list"]
    u = np.array([e["nodes"][0] for e in edges], dtype=int)
    v = np.array([e["nodes"][1] for e in edges], dtype=int)
    w = np.array([e["weight"] for e in edges], dtype=float)

    return {
        "u": u,
        "v": v,
        "w": w,
        "sum_weights": float(w.sum()),
    }


def maxcut_energy_from_bitstring(
    bitstring: str, instance_context: dict[str, np.ndarray | float]
) -> float:
    """Evaluate one measured bitstring in the MaxCut energy convention.

    Parameters
    ----------
    bitstring : str
        Measured bitstring from the hardware counts.
    instance_context : dict
        Graph data returned by :func:`load_maxcut_instance_context`.

    Returns
    -------
    float
        Energy corresponding to the measured bitstring.
    """
    bs = bitstring[::-1]
    bits = np.fromiter((c == "1" for c in bs), dtype=np.int8)

    u = instance_context["u"]
    v = instance_context["v"]
    w = instance_context["w"]
    sum_weights = instance_context["sum_weights"]

    cut_val = float(np.sum(w * (bits[u] != bits[v])))
    return cut_val - 0.5 * sum_weights


def counts_from_bitstring_samples(bitstrings: list[str]) -> dict[str, int]:
    """Aggregate a sample stream into a counts dictionary."""

    return {str(bitstring): int(count) for bitstring, count in Counter(bitstrings).items()}


def best_prefix_metrics(
    bitstrings: list[str],
    checkpoints: list[int],
    instance_context: dict[str, np.ndarray | float],
    min_cut: float,
    max_cut: float,
    sum_weights: float,
) -> list[dict[str, float | int | dict[str, int] | str]]:
    """Summarize best-so-far MaxCut metrics for cumulative sample checkpoints.

    Parameters
    ----------
    bitstrings : list[str]
        Ordered sample stream.
    checkpoints : list[int]
        Prefix lengths to evaluate. Values are clipped to the available stream length.
    instance_context : dict
        Graph data returned by :func:`load_maxcut_instance_context`.
    min_cut, max_cut, sum_weights : float
        Instance min/max-cut metadata.

    Returns
    -------
    list[dict]
        One dictionary per checkpoint containing counts and best-so-far metrics.
    """

    if not bitstrings:
        return []

    rows: list[dict[str, float | int | dict[str, int] | str]] = []
    running_counts: Counter[str] = Counter()
    best_energy = float("-inf")
    best_cut_value = float("-inf")
    best_ratio = float("-inf")
    best_bitstring = ""
    checkpoint_iter = iter(sorted({min(len(bitstrings), int(val)) for val in checkpoints if int(val) > 0}))
    next_checkpoint = next(checkpoint_iter, None)

    for shot_idx, bitstring in enumerate(bitstrings, start=1):
        running_counts[str(bitstring)] += 1
        energy = maxcut_energy_from_bitstring(bitstring, instance_context)
        cut_value = energy + 0.5 * sum_weights
        approx_ratio = maxcut_approximation_ratio(min_cut, max_cut, sum_weights, energy)

        if approx_ratio > best_ratio:
            best_ratio = float(approx_ratio)
            best_cut_value = float(cut_value)
            best_energy = float(energy)
            best_bitstring = str(bitstring)

        if next_checkpoint is None or shot_idx != next_checkpoint:
            continue

        rows.append(
            {
                "Q": int(shot_idx),
                "counts": {str(bitstring): int(count) for bitstring, count in running_counts.items()},
                "best_bitstring": best_bitstring,
                "best_energy": best_energy,
                "best_cut_value": best_cut_value,
                "best_approx_ratio": best_ratio,
            }
        )
        next_checkpoint = next(checkpoint_iter, None)
        if next_checkpoint is None:
            break

    return rows

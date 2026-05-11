import re
import math
import os
from pathlib import Path
from typing import Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.ticker import (
    FixedLocator,
    FormatStrFormatter,
    FuncFormatter,
    LogFormatterMathtext,
    LogLocator,
    MultipleLocator,
)


def is_empty_nested_list(x):
    """Check whether `x` is a non-empty list of empty lists.

    Parameters
    ----------
    x : object
        Value to check.

    Returns
    -------
    bool
        True if `x` is a list with at least one element and every element is an empty list.
    """
    return isinstance(x, list) and len(x) > 0 and all(isinstance(i, list) and len(i) == 0 for i in x)

def counts_to_samples_df(df_hardware: pd.DataFrame) -> pd.DataFrame:
    """
    Expand df_hardware rows into one row per *shot* (or per unique bitstring w/ weight).
    Returns a really long DataFrame with columns:
    - instance_name, training_method, job_p, training_p, bitstring, count, prob
    """
    rows = []
    for _, r in df_hardware.iterrows():
        counts = r["counts"]
        if not isinstance(counts, dict) or len(counts) == 0:
            continue

        total = sum(counts.values())
        for b, c in counts.items():
            rows.append({
                "instance_name": r["instance_name"],
                "training_method": r["training_method"],
                "job_p": r["job_p"],
                "train_p": r["training_p"],
                "bitstring": b,
                "count": c,
                "prob": c / total if total else np.nan,
            })

    return pd.DataFrame(rows)

def plot_ar_hist_by_training_method_with_points(
    df_samples,
    instance_name: str,
    job_p: int,
    bins=70,
    width_scale=0.42,
    symmetric=True,
    normalize="global"):

    def _ensure_training_method_column(dfin: pd.DataFrame) -> pd.DataFrame:
        """Normalize pandas groupby-apply output across pandas versions."""
        if "training_method" in dfin.columns:
            return dfin
        if getattr(dfin.index, "names", None) and "training_method" in dfin.index.names:
            return dfin.reset_index()
        raise KeyError("training_method")

    # ---------- Filter ----------
    df = df_samples[(df_samples["instance_name"] == instance_name) &
                    (df_samples["job_p"] == job_p)].copy()
    if df.empty:
        raise ValueError("No rows found for that (instance_name, job_p).")

    # ---------- Top 1% probability mass per method ----------
    top_groups = []
    ordered = df.sort_values(["training_method", "approximation_ratio"], ascending=[True, False])
    for _, group in ordered.groupby("training_method", sort=False):
        if group.empty:
            continue
        if float(group["prob"].iloc[0]) > 0.01:
            selected = group.head(1).copy()
        else:
            selected = group[group["prob"].cumsum() <= 0.01].copy()
            if selected.empty:
                # Keep one representative row so the "Top 1%" panel never disappears.
                selected = group.head(1).copy()
        top_groups.append(selected)
    df_top = pd.concat(top_groups, ignore_index=True) if top_groups else df.iloc[0:0].copy()

    def _plot_one(dfin, title_suffix):
        dfin = _ensure_training_method_column(dfin)
        methods = sorted(dfin["training_method"].unique())
        x_pos = np.arange(len(methods))
        x_map = {m: x_pos[i] for i, m in enumerate(methods)}

        best_df = (dfin.groupby("training_method", as_index=False)["approximation_ratio"]
                       .max().rename(columns={"approximation_ratio": "best_AR"}))

        y = dfin["approximation_ratio"].to_numpy()
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        bin_edges = np.linspace(y_min, y_max, bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_h = (bin_edges[1] - bin_edges[0]) * 0.9

        # ---------- Weighted hist per method ----------
        H = []
        for m in methods:
            d = dfin[dfin["training_method"] == m]
            h, _ = np.histogram(
                d["approximation_ratio"].to_numpy(),
                bins=bin_edges,
                weights=d["prob"].to_numpy(),
            )
            H.append(h)
        H = np.array(H)

        if normalize == "per_method":
            denom = np.maximum(H.max(axis=1, keepdims=True), 1e-12)
        else:
            denom = max(H.max(), 1e-12)

        W = (H / denom) * width_scale

        # ---------- Plot ----------
        fig, ax = plt.subplots(figsize=(1.25 * len(methods) + 3, 5))
        for i, m in enumerate(methods):
            widths = W[i]
            if symmetric:
                ax.barh(bin_centers, 2 * widths, left=x_pos[i] - widths,
                        height=bin_h, alpha=0.85)
            else:
                ax.barh(bin_centers, widths, left=x_pos[i],
                        height=bin_h, alpha=0.85)

        ax.scatter(best_df["training_method"].map(x_map),
                   best_df["best_AR"],
                   s=45, edgecolors="k", linewidths=0.6,
                   label="Best AR", zorder=5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylabel("Approximation Ratio")
        ax.set_xlabel("Training_method")
        ax.set_xlim(-0.7, len(methods) - 0.3)
        ax.set_ylim(0.45, 1.00)
        ax.set_title(f"{instance_name} | p={job_p} | {title_suffix}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.show()

    # ---------- Plots ----------
    _plot_one(df, "All samples")
    _plot_one(df_top, "Top 1%")

def plot_training_bricks(agg, step_cols):

    def lighten(color, amount=0.5):
        c = np.array(mcolors.to_rgb(color))
        return tuple(c + (1 - c) * amount)

    methods = sorted(agg["method_base"].unique())
    depths  = sorted(agg["job_p"].dropna().unique())

    EDGE_LW = 0.9

    fig, ax = plt.subplots(figsize=(14,5))

    cmap = plt.get_cmap("tab10")
    method_color = {m: cmap(i % 10) for i,m in enumerate(methods)}

    x = np.arange(len(depths))
    width = 0.8 / max(1,len(methods))

    for i,m in enumerate(methods):
        sub = agg[agg["method_base"]==m].set_index("job_p").reindex(depths)

        xpos = x - 0.4 + width/2 + i*width
        bottom = np.zeros(len(depths))

        base = method_color[m]

        outer_vals = sub["outer_init"].to_numpy()
        ax.bar(xpos, outer_vals, width, bottom=bottom,
            color=base, edgecolor="black", linewidth=EDGE_LW)
        bottom += outer_vals

        for s_idx,c in enumerate(step_cols, start=1):
            vals = sub[c].to_numpy()
            col = lighten(base, amount=min(0.85, 0.18+0.06*s_idx))
            ax.bar(xpos, vals, width, bottom=bottom,
                color=col, edgecolor="black", linewidth=EDGE_LW)
            bottom += vals

        ax.errorbar(
            xpos,
            sub["brick_total"].to_numpy(),
            yerr=sub["sem_total"].to_numpy(),
            fmt="none", ecolor="black", capsize=4, lw=1.5
        )

    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("Mean training duration (s)")
    ax.set_title("Mean training duration across instances with depth wise breakdown")

    handles=[plt.Rectangle((0,0),1,1,facecolor=method_color[m],edgecolor="black") for m in methods]
    ax.legend(handles,methods,bbox_to_anchor=(1.02,1),loc="upper left",frameon=False)

    plt.tight_layout()
    plt.show()


def sem(s: pd.Series) -> float:
    """Compute the standard error of the mean (SEM) of a Series.

    Parameters
    ----------
    s : pandas.Series
        Input values.

    Returns
    -------
    float
        Standard error of the mean. Returns 0.0 when fewer than two non-null
        observations are available.
    """
    n = int(s.count())
    return 0.0 if n <= 1 else float(s.std(ddof=1) / math.sqrt(n))


def title_from_instance_names(d: pd.DataFrame, p_val: float) -> str:
    """Build a plot title from instance names and a QAOA depth.

    Parameters
    ----------
    d : pandas.DataFrame
        DataFrame containing an optional ``instance_name`` column.
    p_val : float
        QAOA depth value to include in the title.

    Returns
    -------
    str
        Title string. If ``instance_name`` is not present or contains no valid
        values, the title will only include ``p``.
    """
    p_txt = int(p_val) if float(p_val).is_integer() else p_val
    names = d["instance_name"].dropna().astype(str).unique().tolist() if "instance_name" in d.columns else []
    if not names:
        return f"p = {p_txt}"
    cores = sorted({n[3:] for n in names if len(n) > 3})
    return f"{cores[0]} | p = {p_txt}" if len(cores) == 1 else f"p = {p_txt}"


def make_asof_per_file(inner: pd.DataFrame) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Create a per-file as-of merge function for accumulating inner durations.

    This factory exists to mirror the notebook-local ``asof_per_file`` helper,
    while keeping ``inner`` as an explicit dependency.

    Parameters
    ----------
    inner : pandas.DataFrame
        Precomputed DataFrame with cumulative inner durations.
        Must contain the columns ``file_name``, ``depth_step``, and
        ``inner_cum``.

    Returns
    -------
    Callable[[pandas.DataFrame], pandas.DataFrame]
        A function suitable for use with ``df.groupby('file_name').apply(...)``.
        The group DataFrame is expected to contain a numeric ``job_p`` column.

    Notes
    -----
    The returned function:
    - Drops rows where ``job_p`` is NaN (matching the notebook behavior).
    - Adds an ``inner_duration_sum`` column representing the cumulative sum of
      inner durations up to (and including) the largest ``depth_step`` not
      exceeding ``job_p``.
    """

    def asof_per_file(g: pd.DataFrame) -> pd.DataFrame:
        fn = g.name
        rhs = inner[inner["file_name"].eq(fn)].sort_values("depth_step")
        g2 = g.dropna(subset=["job_p"]).sort_values("job_p")
        g2 = g2.copy()
        g2["file_name"] = fn
        if rhs.empty:
            g2["inner_duration_sum"] = 0.0
            return g2
        out = pd.merge_asof(
            g2,
            rhs[["depth_step", "inner_cum"]],
            left_on="job_p",
            right_on="depth_step",
            direction="backward",
        )
        out["file_name"] = fn
        out["inner_duration_sum"] = out["inner_cum"].fillna(0.0)
        return out.drop(columns=["depth_step", "inner_cum"], errors="ignore")

    return asof_per_file


_HW_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf",
    "#999999", "#00c0ff", "#ff00aa", "#00ff80",
]

_HW_MARKERS = [
    "o", "s", "^", "D", "P", "X", "*", "v",
    "<", "h", "H", "d", "+", "x", "|", "_", ">",
]

_PREFIX_IDX = {"F": 0, "FA": 1, "I": 2, "TQA": 3, "PT": 4, "LR": 5}
_EVALUATOR_IDX = {"MPS": 0, "MPSAer": 0, "PP": 1}
_N_METHODS = len(_PREFIX_IDX)
_METHOD_NAMES = {
    "FA": "fixed angles",
    "F": "Fourier",
    "I": "INTERP",
    "LR": "linear ramp",
    "PT": "parameter transfer",
    "RTS": "recursive transition states",
    "TQA": "TQA",
}
_REOPT_METHODS = {"FA", "TQA"}
_PURE_FIXED_ANGLE_COLOR = "#6a3d9a"


def _is_pure_fixed_angle_label(color_label: str) -> bool:
    """Return True for fixed-angle methods without the reoptimization suffix."""
    parts = color_label.split("_")
    if not parts or parts[0] != "FA":
        return False
    has_reopt_suffix = "angleOpt" in parts or ("opt" in parts and "no" not in parts)
    return not has_reopt_suffix


def _hw_style_ind(color_label: str) -> int | None:
    """Return the hardware-style marker/color index for a method label."""
    parts = color_label.split("_")
    method_idx = _PREFIX_IDX.get(parts[0])
    evaluator_idx = _EVALUATOR_IDX.get(parts[1]) if len(parts) > 1 else None
    if method_idx is None or evaluator_idx is None:
        return None
    return evaluator_idx * _N_METHODS + method_idx - 1


def _compact_method_label(label: str) -> str:
    """Build the compact display label used in IBM QAOA plot legends."""
    parts = label.split("_")
    method_key = parts[0] if parts else label
    evaluator = parts[1] if len(parts) > 1 else ""
    method_str = _METHOD_NAMES.get(method_key, method_key)
    has_no_opt = "no" in parts and "opt" in parts
    has_angle_opt = any(token in {"angleOpt", "*"} for token in parts)
    has_param_opt = ("opt" in parts) and not has_no_opt and not has_angle_opt

    if has_angle_opt:
        method_str = rf"{method_str}$^{{*}}$"
    elif has_param_opt:
        method_str = rf"{method_str}$^{{\dagger}}$"

    return f"{method_str} with {evaluator}" if evaluator else method_str


def _optimization_level(label: str) -> int:
    """Return 0=no opt, 1=method-parameter opt, 2=angle opt."""
    parts = str(label).split("_")
    has_no_opt = "no" in parts and "opt" in parts
    has_angle_opt = any(token in {"angleOpt", "*"} for token in parts)
    has_param_opt = ("opt" in parts) and not has_no_opt and not has_angle_opt
    if has_angle_opt:
        return 2
    if has_param_opt:
        return 1
    return 0


def _optimization_size_maps(
    raw_small: float = 8.0,
    raw_medium: float = 10.0,
    raw_large: float = 12.0,
    centroid_small: float = 14.0,
    centroid_medium: float = 18.0,
    centroid_large: float = 22.0,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return marker-size maps keyed by optimization level."""
    raw_ms_map = {0: raw_small, 1: raw_medium, 2: raw_large}
    centroid_ms_map = {0: centroid_small, 1: centroid_medium, 2: centroid_large}
    return raw_ms_map, centroid_ms_map


def _optimization_legend_handles(
    marker: str = "o",
    markerfacecolor: str = "white",
    markeredgecolor: str = "k",
) -> list[Line2D]:
    """Build a compact legend key for optimization-state marker sizing."""
    _, centroid_ms_map = _optimization_size_maps()
    labels = {
        0: "no optimization",
        1: r"$^{\dagger}$ method-parameter optimization",
        2: r"$^{*}$ QAOA angle optimization",
    }
    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markeredgewidth=1.2,
            linestyle="",
            markersize=centroid_ms_map[level],
            label=labels[level],
        )
        for level in [0, 1, 2]
    ]


def _lighten_color(color, amount: float = 0.5):
    """Return a lighter version of a matplotlib-compatible color."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1 - c) * amount)


def _ensure_save_dir(save_dir: str | None) -> None:
    """Create the output directory when figure saving is enabled."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)


def _draw_training_bars(
    target_ax,
    agg: pd.DataFrame,
    step_cols: list[str],
    methods: list[str],
    depths,
    color_map: dict,
    edge_lw: float,
    bw: float | None = None,
    method_subset: list[str] | None = None,
) -> None:
    """Draw the stacked training-duration bars for the IBM QAOA brick plot."""
    x = np.arange(len(depths))
    draw_methods = method_subset if method_subset is not None else methods
    n = len(draw_methods)
    if bw is None:
        bw = 0.8 / max(1, n)

    for i, method in enumerate(draw_methods):
        sub = agg[agg["method_base"] == method].set_index("job_p").reindex(depths)
        xpos = x - n * bw / 2 + bw * (i + 0.5)
        base = color_map.get(method, plt.get_cmap("tab10")(methods.index(method) % 10))
        bottom = np.zeros(len(depths))

        outer_vals = sub["outer_init"].to_numpy()
        target_ax.bar(
            xpos,
            outer_vals,
            bw,
            bottom=bottom,
            color=base,
            edgecolor="black",
            linewidth=edge_lw,
        )
        bottom += outer_vals

        for step_idx, col_name in enumerate(step_cols, start=1):
            vals = sub[col_name].to_numpy()
            color = _lighten_color(base, amount=min(0.85, 0.18 + 0.06 * step_idx))
            target_ax.bar(
                xpos,
                vals,
                bw,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=edge_lw,
            )
            bottom += vals

        target_ax.errorbar(
            xpos,
            sub["brick_total"].to_numpy(),
            yerr=sub["sem_total"].to_numpy(),
            fmt="none",
            ecolor="black",
            capsize=4,
            lw=1.5,
        )


def prepare_ibm_qaoa_plot_data(
    df_sb_final: pd.DataFrame, df_hardware: pd.DataFrame, num_nodes: int
) -> dict[str, object]:
    """Prepare shared plot data and style mappings for IBM QAOA figures.

    Parameters
    ----------
    df_sb_final : pandas.DataFrame
        Aggregated stochastic-benchmark results used by the IBM QAOA notebook.
    df_hardware : pandas.DataFrame
        Hardware-run records used to backfill ``instance_name`` when needed.
    num_nodes : int
        Problem size used as a fallback graph identifier when no instance name
        is available.

    Returns
    -------
    dict of str to object
        Dictionary containing:

        - ``df_points`` : point-level plotting data
        - ``color_map`` : mapping from color labels to plot colors
        - ``shape_map`` : mapping from color labels to marker shapes
        - ``label_map`` : mapping from color labels to legend labels
        - ``graph_id`` : identifier used in saved figure filenames
    """
    df_plot = df_sb_final.copy()
    n_rows_input = len(df_plot)
    required_columns = ["total duration", "approximation_ratio", "job_p", "file_name"]
    missing_columns = [col for col in required_columns if col not in df_plot.columns]
    if missing_columns:
        raise KeyError(
            "df_sb_final is missing columns required for IBM QAOA plotting: "
            f"{missing_columns}. Available columns: {list(df_plot.columns)}"
        )

    df_plot["total_duration_s"] = pd.to_numeric(df_plot["total duration"], errors="coerce")
    df_plot["approximation_ratio"] = pd.to_numeric(df_plot["approximation_ratio"], errors="coerce")
    df_plot["job_p"] = pd.to_numeric(df_plot["job_p"], errors="coerce")

    numeric_debug = {
        "total_duration_non_null": int(df_sb_final["total duration"].notna().sum()),
        "total_duration_numeric": int(df_plot["total_duration_s"].notna().sum()),
        "total_duration_positive": int((df_plot["total_duration_s"] > 0).sum()),
        "approximation_ratio_non_null": int(df_sb_final["approximation_ratio"].notna().sum()),
        "approximation_ratio_numeric": int(df_plot["approximation_ratio"].notna().sum()),
        "job_p_non_null": int(df_sb_final["job_p"].notna().sum()),
        "job_p_numeric": int(df_plot["job_p"].notna().sum()),
        "file_name_non_null": int(df_sb_final["file_name"].notna().sum()),
    }
    numeric_debug["sample_required_rows"] = (
        df_sb_final.loc[:, required_columns].head(5).astype(str).to_dict("records")
    )

    if "instance_name" not in df_plot.columns and "file_name" in df_plot.columns:
        df_plot = df_plot.merge(
            df_hardware[["file_name", "instance_name"]].drop_duplicates(),
            on="file_name",
            how="left",
        )

    df_plot = df_plot.dropna(
        subset=["total_duration_s", "approximation_ratio", "job_p", "file_name"]
    )
    df_plot = df_plot[df_plot["total_duration_s"] > 0]
    n_rows_after_numeric_filters = len(df_plot)

    # Prefer the raw training method key when present so closely related
    # variants such as LR_PP_opt and LR_PP_angleOpt do not collapse into the
    # same display family before legend generation.
    df_plot["group_label"] = df_plot.get(
        "training_method", df_plot.get("trainer_label", "method")
    ).astype(str)
    if "evaluator_label" in df_plot.columns and df_plot["evaluator_label"].nunique(dropna=True) > 1:
        df_plot["group_label"] = (
            df_plot["group_label"] + " | " + df_plot["evaluator_label"].astype(str)
        )

    df_plot["color_label"] = df_plot["group_label"].str.replace(r"_\d+$", "", regex=True)

    df_points = (
        df_plot.groupby(["job_p", "group_label", "color_label", "file_name"], as_index=False)
        .agg(
            total_duration_s=("total_duration_s", "mean"),
            approximation_ratio=("approximation_ratio", "mean"),
            instance_name=("instance_name", "first"),
        )
    )

    df_points = df_points[
        ~df_points["color_label"].str.contains(r"_MPS(?!Aer)(?:_|$)", regex=True)
    ]
    n_rows_after_label_filters = len(df_points)

    color_labels_all = sorted(df_points["color_label"].unique())
    color_map = {}
    shape_map = {}
    fallback_idx = 11

    for label in color_labels_all:
        ind = _hw_style_ind(label)
        if ind is not None:
            color_map[label] = _HW_COLORS[ind % len(_HW_COLORS)]
            shape_map[label] = _HW_MARKERS[ind % len(_HW_MARKERS)]
        else:
            color_map[label] = _HW_COLORS[fallback_idx % len(_HW_COLORS)]
            shape_map[label] = _HW_MARKERS[fallback_idx % len(_HW_MARKERS)]
            fallback_idx += 1
        if _is_pure_fixed_angle_label(label):
            color_map[label] = _PURE_FIXED_ANGLE_COLOR

    label_map = {label: _compact_method_label(label) for label in color_labels_all}

    sample_inst = (
        df_points["instance_name"].dropna().iloc[0]
        if "instance_name" in df_points.columns and not df_points["instance_name"].dropna().empty
        else ""
    )
    graph_id = sample_inst[3:] if len(sample_inst) > 3 else f"N{num_nodes}"

    return {
        "df_points": df_points,
        "color_map": color_map,
        "shape_map": shape_map,
        "label_map": label_map,
        "graph_id": graph_id,
        "debug_summary": {
            "input_rows": int(n_rows_input),
            "rows_after_numeric_filters": int(n_rows_after_numeric_filters),
            "rows_after_grouping_and_label_filters": int(n_rows_after_label_filters),
            "n_panels": int(df_points["job_p"].nunique()) if not df_points.empty else 0,
            "job_p_values": sorted(df_points["job_p"].dropna().unique().tolist()) if not df_points.empty else [],
            "color_labels": color_labels_all,
            "numeric_debug": numeric_debug,
        },
    }


def plot_ibm_qaoa_performance_panels(
    plot_data: dict[str, object], save_dir: str | None = "plots"
) -> None:
    """Plot IBM QAOA performance panels split by QAOA depth.

    Parameters
    ----------
    plot_data : dict of str to object
        Shared plotting bundle returned by
        :func:`prepare_ibm_qaoa_plot_data`.
    save_dir : str or None, default="plots"
        Output directory for saved figures. When ``None``, figures are not
        written to disk.
    """
    fs_tick = 18
    fs_label = 20
    fs_title = 28
    fs_legend = 18
    raw_ms_map, centroid_ms_map = _optimization_size_maps()
    y_scale = 100.0
    y_cushion = 3.0
    eb_opts = {"mec": "k", "ecolor": "k", "capsize": 3, "elinewidth": 1.2}

    _ensure_save_dir(save_dir)

    df_points = plot_data["df_points"]
    color_map = plot_data["color_map"]
    shape_map = plot_data["shape_map"]
    label_map = plot_data["label_map"]
    graph_id = plot_data["graph_id"]
    debug_summary = plot_data.get("debug_summary", {})

    if df_points.empty:
        raise ValueError(
            "No valid IBM QAOA plot points remain after preprocessing. "
            f"Debug summary: {debug_summary}. "
            "Check that df_SB_final contains non-null 'total duration', "
            "'approximation_ratio', 'job_p', and 'file_name' values, and that "
            "the label filtering did not remove every method."
        )

    p_vals = sorted(df_points["job_p"].unique())
    n_panels = len(p_vals)
    if n_panels <= 0:
        raise ValueError(
            "No QAOA depth panels are available to plot. "
            f"Debug summary: {debug_summary}."
        )

    x_all_global = df_points["total_duration_s"].dropna()
    x_lo_global = max(x_all_global.min() / 3, 0.1)
    x_hi_global = x_all_global.max() * 3

    y_all_global = y_scale * df_points["approximation_ratio"].dropna()
    y_lo_global = max(0.0, float(y_all_global.min()) - y_cushion)
    y_hi_global = min(100.0, float(y_all_global.max()) + y_cushion)

    if y_hi_global - y_lo_global < 6.0:
        y_mid_global = 0.5 * (y_lo_global + y_hi_global)
        half_span = 3.0
        y_lo_global = max(0.0, y_mid_global - half_span)
        y_hi_global = min(100.0, y_mid_global + half_span)

    y_span_global = y_hi_global - y_lo_global
    if y_span_global <= 10.0:
        y_major = 1.0
    elif y_span_global <= 20.0:
        y_major = 2.0
    else:
        y_major = 5.0

    fig, axs = plt.subplots(1, n_panels, figsize=(9 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axs = [axs]

    legend_dict = {}

    for ax_idx, (ax, p) in enumerate(zip(axs, p_vals)):
        d = df_points[df_points["job_p"] == p]

        for (_, color_label), group in d.groupby(["group_label", "color_label"]):
            x = group["total_duration_s"]
            y = y_scale * group["approximation_ratio"]
            color = color_map[color_label]
            marker = shape_map[color_label]
            opt_level = _optimization_level(color_label)

            ax.errorbar(
                x,
                y,
                fmt=marker,
                linestyle="none",
                color=color,
                alpha=0.35,
                ms=raw_ms_map[opt_level],
                **eb_opts,
            )

            x_mean = float(x.mean())
            y_mean = float(y.mean())
            x_err = sem(x)
            y_err = sem(y)
            if x_err > 0 and x_mean - x_err <= 0:
                x_err = 0.9 * x_mean

            ax.errorbar(
                x_mean,
                y_mean,
                xerr=x_err,
                yerr=y_err,
                fmt=marker,
                ms=centroid_ms_map[opt_level],
                linestyle="none",
                color=color,
                zorder=10,
                **eb_opts,
            )

            expanded = label_map[color_label]
            if expanded not in legend_dict:
                legend_dict[expanded] = Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=color,
                    markeredgecolor="k",
                    lw=0,
                    markersize=centroid_ms_map[opt_level],
                    label=expanded,
                )

        panel_label = f"({chr(ord('a') + ax_idx)})"
        ax.text(
            0.02,
            0.97,
            panel_label,
            transform=ax.transAxes,
            fontsize=fs_title,
            fontweight="normal",
            va="top",
            ha="left",
        )

        ax.set_xlabel("Total duration (s)", fontsize=fs_label)
        ax.tick_params(axis="both", labelsize=fs_tick, labelleft=True)
        ax.set_xscale("log")
        ax.set_xlim(x_lo_global, x_hi_global)
        ax.set_ylim(y_lo_global, y_hi_global)
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
        ax.grid(True)

    axs[0].set_ylabel("Hardware approximation ratio (%)", fontsize=fs_label)

    sorted_items = sorted(legend_dict.items(), key=lambda x: x[0])
    all_legend_handles = [handle for _, handle in sorted_items]
    all_legend_handles.extend(
        _optimization_legend_handles(marker="o", markerfacecolor="white", markeredgecolor="k")
    )

    fig.legend(
        handles=all_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        borderaxespad=0,
        frameon=True,
        fontsize=fs_legend,
    )

    fig.tight_layout(rect=[0.035, 0.05, 1, 1])

    if save_dir:
        fname = f"{graph_id}_performance"
        fig.savefig(f"{save_dir}/{fname}.pdf", bbox_inches="tight")
        fig.savefig(f"{save_dir}/{fname}.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {save_dir}/{fname}.pdf")

    plt.show()


def plot_ibm_qaoa_overlay(
    plot_data: dict[str, object], save_dir: str | None = "plots"
) -> None:
    """Plot the IBM QAOA depth-transition overlay with centroid arrows.

    Parameters
    ----------
    plot_data : dict of str to object
        Shared plotting bundle returned by
        :func:`prepare_ibm_qaoa_plot_data`.
    save_dir : str or None, default="plots"
        Output directory for saved figures. When ``None``, figures are not
        written to disk.

    Raises
    ------
    ValueError
        If fewer than two QAOA depth levels are present.
    """
    fs_tick = 18
    fs_label = 20
    fs_legend = 15
    raw_ms_map, centroid_ms_map = _optimization_size_maps(
        raw_small=8.0,
        raw_medium=10.0,
        raw_large=12.0,
        centroid_small=14.0,
        centroid_medium=18.0,
        centroid_large=22.0,
    )
    y_scale = 100.0

    _ensure_save_dir(save_dir)

    overlay = plot_data["df_points"].copy()
    color_map = plot_data["color_map"]
    shape_map = plot_data["shape_map"]
    label_map = plot_data["label_map"]
    graph_id = plot_data["graph_id"]

    overlay["job_p"] = pd.to_numeric(overlay["job_p"], errors="coerce")
    overlay = overlay.dropna(subset=["total_duration_s", "approximation_ratio", "job_p"])

    p_levels = sorted(overlay["job_p"].astype(int).unique())
    if len(p_levels) < 2:
        raise ValueError("Need at least two p levels for arrow overlay plot.")

    p_max = max(p_levels)
    inst_high_p = overlay[overlay["job_p"].astype(int) == p_max]

    centroids = (
        overlay.groupby(["color_label", "job_p"], as_index=False)
        .agg(
            dur_mean=("total_duration_s", "mean"),
            ar_mean=("approximation_ratio", "mean"),
            dur_sem=("total_duration_s", sem),
            ar_sem=("approximation_ratio", sem),
        )
    )
    centroids["ar_mean"] *= y_scale
    centroids["ar_sem"] *= y_scale

    valid_methods = []
    for color_label, group in centroids.groupby("color_label"):
        if group["job_p"].nunique() >= 2:
            valid_methods.append(color_label)

    centroids = centroids[centroids["color_label"].isin(valid_methods)].copy()

    x_vals = overlay["total_duration_s"].dropna()
    x_lo = max(x_vals.min() / 3, 0.1)
    x_hi = x_vals.max() * 3
    y_lo = max(0.0, float(y_scale * overlay["approximation_ratio"].min()) - 3.0)

    fig, ax = plt.subplots(figsize=(18, 6))

    for color_label, group in inst_high_p.groupby("color_label"):
        if color_label not in valid_methods:
            continue
        opt_level = _optimization_level(color_label)
        ax.errorbar(
            group["total_duration_s"],
            y_scale * group["approximation_ratio"],
            fmt=shape_map[color_label],
            linestyle="none",
            color=color_map[color_label],
            alpha=0.28,
            ms=raw_ms_map[opt_level],
            mec="k",
            ecolor="k",
            capsize=2,
            elinewidth=0.9,
        )

    for color_label in sorted(valid_methods):
        group = (
            centroids[centroids["color_label"] == color_label]
            .sort_values("job_p")
            .reset_index(drop=True)
        )
        color = color_map[color_label]
        marker = shape_map[color_label]
        opt_level = _optimization_level(color_label)

        for _, row in group.iterrows():
            p = int(row["job_p"])
            if p == p_max:
                ax.errorbar(
                    row["dur_mean"],
                    row["ar_mean"],
                    xerr=row["dur_sem"],
                    yerr=row["ar_sem"],
                    fmt=marker,
                    ms=centroid_ms_map[opt_level],
                    linestyle="none",
                    color=color,
                    alpha=1.0,
                    zorder=11,
                    mec="k",
                    ecolor="k",
                    capsize=3,
                    elinewidth=1.2,
                )
            else:
                ax.plot(
                    row["dur_mean"],
                    row["ar_mean"],
                    marker=marker,
                    ms=raw_ms_map[opt_level],
                    linestyle="none",
                    mfc=color,
                    mec="k",
                    mew=0.9,
                    alpha=0.90,
                    zorder=10,
                )

        for i in range(len(group) - 1):
            row0 = group.iloc[i]
            row1 = group.iloc[i + 1]
            ax.annotate(
                "",
                xy=(row1["dur_mean"], row1["ar_mean"]),
                xytext=(row0["dur_mean"], row0["ar_mean"]),
                arrowprops=dict(
                    arrowstyle="->",
                    linestyle="--",
                    color=color,
                    lw=1.7,
                    alpha=0.80,
                    shrinkA=4,
                    shrinkB=10,
                    mutation_scale=14,
                ),
                zorder=12,
            )

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, 100.0)
    ax.yaxis.set_major_locator(MultipleLocator(10.0))
    ax.grid(True)
    ax.set_xlabel("Total duration (s)", fontsize=fs_label)
    ax.set_ylabel("Hardware approximation ratio (%)", fontsize=fs_label)
    ax.tick_params(axis="both", labelsize=fs_tick)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=shape_map[color_label],
            color=color_map[color_label],
            markeredgecolor="k",
            lw=0,
            markersize=centroid_ms_map[_optimization_level(color_label)],
            label=label_map[color_label],
        )
        for color_label in sorted(valid_methods)
    ]

    if len(p_levels) == 2:
        p_sizes = [9, 14]
    else:
        p_sizes = np.linspace(8, 15, len(p_levels))

    depth_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markeredgecolor="k",
            markerfacecolor="gray",
            lw=0,
            markersize=float(ms),
            label=f"p={int(p)}",
        )
        for p, ms in zip(p_levels, p_sizes)
    ]

    cue_handles = [
        Line2D([0, 1], [0, 0], color="black", lw=1.4, linestyle="--", label="p-shift arrows"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            markeredgecolor="k",
            lw=0,
            markersize=7,
            alpha=0.28,
            label=f"p={int(p_max)} instances",
        ),
    ]
    opt_handles = _optimization_legend_handles(
        marker="o",
        markerfacecolor="white",
        markeredgecolor="k",
    )

    ax.legend(
        handles=method_handles + depth_handles + cue_handles + opt_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        borderaxespad=0,
        frameon=True,
        fontsize=fs_legend,
    )

    fig.subplots_adjust(left=0.12, right=0.99, top=0.96, bottom=0.33)

    if save_dir:
        fname_overlay = f"{graph_id}_performance_overlay"
        fig.savefig(f"{save_dir}/{fname_overlay}.pdf", bbox_inches="tight")
        fig.savefig(f"{save_dir}/{fname_overlay}.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {save_dir}/{fname_overlay}.pdf")

    plt.show()


def prepare_training_bricks_data(
    df_flat: pd.DataFrame, df_hardware_new: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare aggregated training-brick data for IBM QAOA plots.

    Parameters
    ----------
    df_flat : pandas.DataFrame
        Flattened training log with ``level``, ``depth_step``, and
        ``duration`` columns.
    df_hardware_new : pandas.DataFrame
        Per-run hardware/training records used to attach method metadata.

    Returns
    -------
    tuple of (pandas.DataFrame, list of str)
        Aggregated plotting table ``agg`` and the ordered list of ``step_*``
        columns used to draw the stacked bars.
    """
    steps = (
        df_flat[df_flat["level"] == "inner"]
        .groupby(["file_name", "depth_step"])["duration"]
        .sum()
        .unstack(fill_value=0)
    )
    steps.columns = [f"step_{int(c)}" for c in steps.columns]
    steps = steps.reset_index()

    outer = (
        df_flat[(df_flat["level"] == "outer") & (df_flat["iteration"].isin([0, 1]))]
        .groupby("file_name")["duration"]
        .sum()
        .rename("outer_init")
        .reset_index()
    )

    df_rows = (
        df_hardware_new.merge(steps, on="file_name", how="left")
        .merge(outer, on="file_name", how="left")
        .fillna(0)
    )

    df_rows["method_base"] = df_rows["training_method"].str.replace(r"_\d+$", "", regex=True)
    step_cols = sorted(
        [col for col in df_rows.columns if col.startswith("step_")],
        key=lambda col: int(col.split("_")[1]),
    )

    for col in step_cols:
        k = int(col.split("_")[1])
        df_rows.loc[df_rows["job_p"] < k, col] = 0

    df_rows["brick_total"] = df_rows["outer_init"] + df_rows[step_cols].sum(axis=1)

    agg = df_rows.groupby(["job_p", "method_base"], as_index=False)[
        ["outer_init", "brick_total"] + step_cols
    ].mean()
    agg["sem_total"] = df_rows.groupby(["job_p", "method_base"])["brick_total"].sem().values

    agg = agg[~agg["method_base"].str.contains(r"_MPS(?!Aer)(?:_|$)", regex=True)]

    return agg, step_cols


def plot_ibm_qaoa_training_bricks(
    agg: pd.DataFrame,
    step_cols: list[str],
    color_map: dict,
    label_map: dict,
    graph_id: str,
    save_dir: str | None = "plots",
) -> None:
    """Plot the IBM QAOA depth-wise training-duration brick chart.

    Parameters
    ----------
    agg : pandas.DataFrame
        Aggregated training-duration table from
        :func:`prepare_training_bricks_data`.
    step_cols : list of str
        Ordered ``step_*`` columns used in the stacked bars.
    color_map : dict
        Mapping from method labels to colors.
    label_map : dict
        Mapping from method labels to legend labels.
    graph_id : str
        Identifier used in saved figure filenames.
    save_dir : str or None, default="plots"
        Output directory for saved figures. When ``None``, figures are not
        written to disk.
    """
    fs_tick = 20
    fs_label = 20
    fs_legend = 18
    y_scale = 100.0
    edge_lw = 0.9

    _ensure_save_dir(save_dir)

    methods = sorted(agg["method_base"].unique())
    depths = sorted(agg["job_p"].dropna().unique())
    x = np.arange(len(depths))

    shaded_depth = min(depths) if len(depths) == 2 else None
    shaded_idx = depths.index(shaded_depth) if shaded_depth is not None else None
    shaded_left = (shaded_idx - 0.5) if shaded_idx is not None else None
    shaded_right = (shaded_idx + 0.5) if shaded_idx is not None else None

    upper_bounds = (
        agg["brick_total"].to_numpy(dtype=float) + agg["sem_total"].fillna(0).to_numpy(dtype=float)
    )
    positive_bounds = np.sort(upper_bounds[np.isfinite(upper_bounds) & (upper_bounds > 0)])

    if len(positive_bounds) == 0:
        main_ymax = 1500.0
    elif len(positive_bounds) == 1:
        main_ymax = float(positive_bounds[0]) * 1.12
    else:
        min_side = 2 if len(positive_bounds) >= 6 else 1
        split_idx = None
        split_ratio = 1.0

        for idx in range(len(positive_bounds) - 1):
            left_size = idx + 1
            right_size = len(positive_bounds) - left_size
            if left_size < min_side or right_size < min_side:
                continue

            left = float(positive_bounds[idx])
            right = float(positive_bounds[idx + 1])
            if left <= 0:
                continue

            ratio = right / left
            if ratio > split_ratio:
                split_ratio = ratio
                split_idx = idx

        if split_idx is not None and split_ratio >= 2.0:
            main_ymax = float(positive_bounds[split_idx]) * 1.12
        else:
            main_ymax = float(np.quantile(positive_bounds, 0.7)) * 1.12

    max_total = float(positive_bounds[-1]) if len(positive_bounds) else main_ymax
    if len(positive_bounds):
        main_ymax = max(main_ymax, float(positive_bounds[0]) * 1.12, 1.0)
    main_ymax = min(main_ymax, max_total * 0.9)

    overflow_methods = sorted(
        [
            m
            for m in methods
            if any(
                (
                    agg[agg["method_base"] == m]["brick_total"].to_numpy(dtype=float)
                    + agg[agg["method_base"] == m]["sem_total"].fillna(0).to_numpy(dtype=float)
                )
                > main_ymax
            )
        ]
    )
    inset_ymin = main_ymax

    agg_overflow = agg[agg["method_base"].isin(overflow_methods)].copy()
    if not agg_overflow.empty:
        inset_top_needed = (
            agg_overflow["brick_total"] + agg_overflow["sem_total"].fillna(0)
        ).max()
        inset_ymax = max(float(inset_ymin) * 1.15, float(inset_top_needed) * 1.08)
    else:
        inset_ymax = max(float(inset_ymin) * 1.15, float(max_total) * 1.08)

    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.5, 1.5], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    inset_ax = fig.add_subplot(gs[1])

    if shaded_idx is not None:
        ax.axvspan(shaded_left, shaded_right, color="0.92", zorder=0)
        inset_ax.axvspan(shaded_left, shaded_right, color="0.92", zorder=0)

    _draw_training_bars(ax, agg, step_cols, methods, depths, color_map, edge_lw)
    ax.set_xticks(x)
    ax.set_xticklabels([int(d) for d in depths], fontsize=fs_tick)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.set_xlabel("QAOA depth p", fontsize=fs_label)
    ax.set_ylabel("Mean training duration (s)", fontsize=fs_label)
    ax.set_ylim(0, main_ymax)
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))

    n_all = len(methods)
    bw_all = 0.8 / max(1, n_all)
    fade_h = main_ymax * 0.11
    for bar_idx, method in enumerate(methods):
        sub = agg[agg["method_base"] == method]
        for depth_idx, depth in enumerate(depths):
            vals = sub[sub["job_p"] == depth]["brick_total"].values
            if len(vals) > 0 and float(vals[0]) > main_ymax:
                bar_center = depth_idx - n_all * bw_all / 2 + bw_all * (bar_idx + 0.5)
                x0 = bar_center - bw_all * 0.48
                width = bw_all * 0.96
                ax.add_patch(
                    plt.Rectangle(
                        (x0, main_ymax - fade_h),
                        width,
                        fade_h / 3,
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.18,
                        zorder=12,
                    )
                )
                ax.add_patch(
                    plt.Rectangle(
                        (x0, main_ymax - 2 * fade_h / 3),
                        width,
                        fade_h / 3,
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.32,
                        zorder=13,
                    )
                )
                ax.add_patch(
                    plt.Rectangle(
                        (x0, main_ymax - fade_h / 3),
                        width,
                        fade_h / 3,
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.46,
                        zorder=14,
                    )
                )

    d = 0.012
    kw_main = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1.6)
    ax.plot((-d, +d), (1 - d, 1 + d), **kw_main)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw_main)

    _draw_training_bars(
        inset_ax,
        agg,
        step_cols,
        methods,
        depths,
        color_map,
        edge_lw,
        method_subset=overflow_methods,
    )
    inset_ax.set_ylim(inset_ymin, inset_ymax)
    inset_ax.set_xticks(x)
    inset_ax.set_xticklabels([int(d) for d in depths], fontsize=fs_tick)
    inset_ax.tick_params(axis="y", labelsize=fs_tick)
    inset_ax.set_xlabel("QAOA depth p", fontsize=fs_label)
    inset_ax.set_ylabel("Mean training duration (s)", fontsize=fs_label)
    inset_ax.set_xlim(-0.5, x[-1] + 0.5)
    inset_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))

    kw_inset = dict(transform=inset_ax.transAxes, color="k", clip_on=False, lw=1.6)
    inset_ax.plot((-d, +d), (-d, +d), **kw_inset)
    inset_ax.plot((1 - d, 1 + d), (-d, +d), **kw_inset)

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color_map.get(method, plt.get_cmap("tab10")(i % 10)),
            edgecolor="black",
        )
        for i, method in enumerate(methods)
    ]
    legend_labels = [label_map.get(method, method) for method in methods]
    note_handles = [
        Line2D([0], [0], linestyle="", color="none", label="no superscript: no optimization"),
        Line2D([0], [0], linestyle="", color="none", label=r"$^{\dagger}$ method-parameter optimization"),
        Line2D([0], [0], linestyle="", color="none", label=r"$^{*}$ QAOA angle optimization"),
    ]
    fig.legend(
        handles + note_handles,
        legend_labels + [handle.get_label() for handle in note_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.00),
        ncol=4,
        borderaxespad=0,
        frameon=True,
        fontsize=fs_legend,
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.20)

    if save_dir:
        fname = f"{graph_id}_training_bricks"
        fig.savefig(f"{save_dir}/{fname}.pdf", bbox_inches="tight")
        fig.savefig(f"{save_dir}/{fname}.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {save_dir}/{fname}.pdf")

    plt.show()


def build_recommendation_data(
    df_points: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build centroid and Pareto-frontier data for IBM QAOA recommendations.

    Parameters
    ----------
    df_points : pandas.DataFrame
        Point-level IBM QAOA plotting data from
        :func:`prepare_ibm_qaoa_plot_data`.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        The centroid table ``df_centroids`` and the Pareto-frontier table
        ``df_frontier``.
    """
    df_centroids = (
        df_points.groupby(["group_label", "color_label", "job_p"], as_index=False)
        .agg(
            dur_mean=("total_duration_s", "mean"),
            dur_std=("total_duration_s", "std"),
            dur_count=("total_duration_s", "count"),
            ar_mean=("approximation_ratio", "mean"),
            ar_std=("approximation_ratio", "std"),
            ar_count=("approximation_ratio", "count"),
        )
    )
    df_centroids["dur_sem"] = df_centroids["dur_std"] / np.sqrt(df_centroids["dur_count"])
    df_centroids["ar_sem"] = df_centroids["ar_std"] / np.sqrt(df_centroids["ar_count"])

    df_pareto = (
        df_centroids[df_centroids["ar_mean"] <= 1.0].sort_values("dur_mean").reset_index(drop=True)
    )

    best_ar = -np.inf
    pareto_mask = []
    for _, row in df_pareto.iterrows():
        if row["ar_mean"] > best_ar:
            pareto_mask.append(True)
            best_ar = row["ar_mean"]
        else:
            pareto_mask.append(False)

    df_frontier = df_pareto[pareto_mask].reset_index(drop=True)
    return df_centroids, df_frontier


def plot_ibm_qaoa_recommendation(
    df_points: pd.DataFrame,
    df_frontier: pd.DataFrame,
    color_map: dict,
    shape_map: dict,
    label_map: dict,
    graph_id: str,
    save_dir: str | None = "plots",
) -> None:
    """Plot the IBM QAOA budget-based recommendation figure.

    Parameters
    ----------
    df_points : pandas.DataFrame
        Point-level IBM QAOA plotting data.
    df_frontier : pandas.DataFrame
        Pareto-frontier table returned by :func:`build_recommendation_data`.
    color_map : dict
        Mapping from color labels to colors.
    shape_map : dict
        Mapping from color labels to marker shapes.
    label_map : dict
        Mapping from color labels to legend labels.
    graph_id : str
        Identifier used in saved figure filenames.
    save_dir : str or None, default="plots"
        Output directory for saved figures. When ``None``, figures are not
        written to disk.
    """
    fs_tick = 18
    fs_label = 20
    fs_legend = 16
    y_scale = 100.0
    raw_alpha = 0.18
    raw_ms_map = {0: 8, 1: 10, 2: 12}
    frontier_ms_map = {0: 14, 1: 18, 2: 22}
    eb_opts = {"mec": "k", "ecolor": "k", "capsize": 3, "elinewidth": 1.2}

    _ensure_save_dir(save_dir)

    df_centroids, _ = build_recommendation_data(df_points)
    p_values = sorted(pd.to_numeric(df_points["job_p"], errors="coerce").dropna().astype(int).unique())
    depth_palette = plt.get_cmap("tab10")
    depth_color_map = {
        int(p): depth_palette(idx % depth_palette.N)
        for idx, p in enumerate(p_values)
    }

    x_all = df_centroids["dur_mean"].dropna()
    x_left = max(x_all.min() / 3, 0.1)
    if not df_frontier.empty:
        frontier_x = (
            pd.to_numeric(df_frontier["dur_mean"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        x_last = float(frontier_x[-1])
        if len(frontier_x) >= 2:
            tail_gap_decades = np.log10(max(frontier_x[-1], 1e-12)) - np.log10(
                max(frontier_x[-2], 1e-12)
            )
        else:
            tail_gap_decades = 0.18
        tail_gap_decades = float(np.clip(tail_gap_decades, 0.12, 0.30))
        x_right = 10 ** (np.log10(max(x_last, 1e-12)) + tail_gap_decades)
    else:
        x_right = float(x_all.max()) * 1.25 if not x_all.empty else 1e4

    fig, ax = plt.subplots(figsize=(18, 8))

    def _deterministic_log_jitter(values: pd.Series, width: float = 0.06) -> np.ndarray:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        n = len(arr)
        if n <= 1:
            return arr
        offsets = np.linspace(-width, width, n)
        return arr * np.power(10.0, offsets)

    if not df_frontier.empty:
        for seg_idx in range(len(df_frontier)):
            row = df_frontier.iloc[seg_idx]
            color_seg = depth_color_map[int(row["job_p"])]
            x_start = x_left if seg_idx == 0 else row["dur_mean"]
            x_end = (
                df_frontier.iloc[seg_idx + 1]["dur_mean"]
                if seg_idx < len(df_frontier) - 1
                else x_right
            )
            y_this = y_scale * row["ar_mean"]

            ax.plot(
                [x_start, x_end],
                [y_this, y_this],
                color=color_seg,
                linewidth=3,
                linestyle="--",
                alpha=0.85,
                zorder=9,
            )

            if seg_idx > 0:
                y_prev = y_scale * df_frontier.iloc[seg_idx - 1]["ar_mean"]
                ax.plot(
                    [row["dur_mean"], row["dur_mean"]],
                    [y_prev, y_this],
                    color=color_seg,
                    linewidth=3,
                    linestyle="--",
                    alpha=0.85,
                    zorder=9,
                )

    frontier_pairs = df_frontier[["color_label", "job_p"]].drop_duplicates()

    for _, pair in frontier_pairs.iterrows():
        color_label = pair["color_label"]
        p_val = int(pair["job_p"])
        inst_sub = df_points[
            (df_points["color_label"] == color_label) & (df_points["job_p"] == p_val)
        ]
        if inst_sub.empty:
            continue
        opt_level = _optimization_level(color_label)
        inst_sub = inst_sub.sort_values(
            ["total_duration_s", "approximation_ratio", "instance_name"],
            kind="mergesort",
        ).reset_index(drop=True)
        x_plot = _deterministic_log_jitter(inst_sub["total_duration_s"])
        ax.plot(
            x_plot,
            y_scale * inst_sub["approximation_ratio"],
            linestyle="",
            marker=shape_map[color_label],
            ms=raw_ms_map[opt_level],
            mfc=depth_color_map[p_val],
            mec="k",
            mew=0.8,
            alpha=raw_alpha,
            zorder=6,
            rasterized=True,
        )

    y_values = []
    if not df_frontier.empty:
        y_values.append(y_scale * df_frontier["ar_mean"].dropna().to_numpy(dtype=float))
        if "ar_sem" in df_frontier.columns:
            y_values.append(
                y_scale
                * (df_frontier["ar_mean"] + df_frontier["ar_sem"].fillna(0)).dropna().to_numpy(dtype=float)
            )
            y_values.append(
                y_scale
                * (df_frontier["ar_mean"] - df_frontier["ar_sem"].fillna(0)).dropna().to_numpy(dtype=float)
            )

    for _, pair in frontier_pairs.iterrows():
        color_label = pair["color_label"]
        p_val = int(pair["job_p"])
        inst_sub = df_points[
            (df_points["color_label"] == color_label) & (df_points["job_p"] == p_val)
        ]
        if not inst_sub.empty:
            y_values.append(
                y_scale * inst_sub["approximation_ratio"].dropna().to_numpy(dtype=float)
            )

    if y_values:
        y_all = np.concatenate([vals for vals in y_values if len(vals) > 0])
        y_min_data = float(np.min(y_all))
        y_max_data = float(np.max(y_all))
        y_span = y_max_data - y_min_data
        y_pad = max(0.8, 0.12 * y_span)
        y_lo = max(0.0, y_min_data - y_pad)
        y_hi = min(100.0, y_max_data + y_pad)

        if y_hi - y_lo < 4.0:
            y_mid = 0.5 * (y_lo + y_hi)
            half_span = 2.0
            y_lo = max(0.0, y_mid - half_span)
            y_hi = min(100.0, y_mid + half_span)
    else:
        y_lo = 50.0
        y_hi = 100.0

    for _, row in df_frontier.iterrows():
        p = int(row["job_p"])
        opt_level = _optimization_level(row["color_label"])
        x_val = float(row["dur_mean"])
        y_val = y_scale * float(row["ar_mean"])
        xerr = None if pd.isna(row.get("dur_sem", np.nan)) else float(row["dur_sem"])
        yerr = None if pd.isna(row.get("ar_sem", np.nan)) else y_scale * float(row["ar_sem"])
        if xerr is not None or yerr is not None:
            ax.errorbar(
                x_val,
                y_val,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                zorder=10,
                **eb_opts,
            )
        ax.plot(
            x_val,
            y_val,
            marker=shape_map[row["color_label"]],
            ms=frontier_ms_map[opt_level],
            mfc=depth_color_map[p],
            mec="k",
            mew=2.0,
            linestyle="",
            zorder=11,
        )

    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Total duration (s)", fontsize=fs_label)
    ax.set_ylabel("Hardware approximation ratio (%)", fontsize=fs_label)
    ax.set_xscale("log")
    ax.set_xlim(x_left, x_right)

    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=100))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=False))
    y_span = y_hi - y_lo
    if y_span <= 8.0:
        y_step = 1.0
    elif y_span <= 18.0:
        y_step = 2.0
    else:
        y_step = 5.0
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.minorticks_off()
    ax.tick_params(axis="both", which="major", labelsize=fs_tick, length=6)
    ax.grid(True)

    legend_pairs = (
        df_frontier[["color_label"]]
        .drop_duplicates()
        .sort_values(["color_label"])
    )
    method_handles = []
    for _, pair in legend_pairs.iterrows():
        color_label = pair["color_label"]
        opt_level = _optimization_level(color_label)
        method_handles.append(
            Line2D(
                [0],
                [0],
                marker=shape_map[color_label],
                color="black",
                markerfacecolor="white",
                markeredgecolor="k",
                markeredgewidth=1.5,
                markersize=frontier_ms_map[opt_level],
                linestyle="",
                label=label_map[color_label],
            )
        )

    depth_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=depth_color_map[p],
            markerfacecolor=depth_color_map[p],
            markeredgecolor="k",
            markeredgewidth=1.0,
            markersize=10,
            linestyle="",
            label=f"p={p}",
        )
        for p in p_values
    ]

    frontier_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="Pareto Frontier",
    )
    no_opt_note = Line2D(
        [0],
        [0],
        linestyle="",
        color="none",
        label="no superscript: no optimization",
    )
    dagger_note = Line2D(
        [0],
        [0],
        linestyle="",
        color="none",
        label=r"$^{\dagger}$ method-parameter optimization",
    )
    star_note = Line2D(
        [0],
        [0],
        linestyle="",
        color="none",
        label=r"$^{*}$ QAOA angle optimization",
    )

    method_legend = fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.24, 0.01),
        ncol=min(3, max(1, len(method_handles))),
        borderaxespad=0,
        fontsize=fs_legend,
        frameon=True,
        title="Method",
    )
    depth_legend = fig.legend(
        handles=depth_handles + [frontier_handle, no_opt_note, dagger_note, star_note],
        loc="upper center",
        bbox_to_anchor=(0.78, 0.01),
        ncol=min(3, max(2, len(depth_handles) + 4)),
        borderaxespad=0,
        fontsize=fs_legend,
        frameon=True,
        title="Depth / Frontier",
        handlelength=2.0,
    )
    fig.add_artist(method_legend)
    fig.add_artist(depth_legend)

    fig.tight_layout()
    fig.subplots_adjust(left=0.08, bottom=0.24)

    if save_dir:
        fname = f"{graph_id}_recommendation"
        fig.savefig(f"{save_dir}/{fname}.pdf", bbox_inches="tight")
        fig.savefig(f"{save_dir}/{fname}.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {save_dir}/{fname}.pdf")

    plt.show()


def print_budget_recommendations(df_frontier: pd.DataFrame) -> None:
    """Print the budget-based IBM QAOA recommendation summary table.

    Parameters
    ----------
    df_frontier : pandas.DataFrame
        Pareto-frontier table returned by :func:`build_recommendation_data`.
    """
    print("\n" + "=" * 80)
    print("BUDGET-BASED RECOMMENDATIONS")
    print("=" * 80)
    print(f"{'Budget Range (s)':<25} {'Recommended Method':<35} {'p':<5} {'Approx Ratio':<15}")
    print("-" * 80)

    for pos, (_, row) in enumerate(df_frontier.iterrows()):
        if pos == 0:
            budget_str = f"0 - {row['dur_mean']:.0f}"
        elif pos < len(df_frontier) - 1:
            prev = df_frontier.iloc[pos - 1]
            budget_str = f"{prev['dur_mean']:.0f} - {row['dur_mean']:.0f}"
        else:
            prev = df_frontier.iloc[pos - 1]
            budget_str = f"> {prev['dur_mean']:.0f}"

        print(
            f"{budget_str:<25} {row['group_label']:<35} "
            f"{int(row['job_p']):<5} {row['ar_mean']:.4f}"
        )

    print("=" * 80)
    print()
    print("FRONTIER POINTS")
    print("=" * 80)
    print(f"{'Method':<35} {'p':<5} {'Duration (s)':<15} {'Approx Ratio':<15}")
    print("-" * 80)
    for _, row in df_frontier.iterrows():
        print(
            f"{row['group_label']:<35} "
            f"{int(row['job_p']):<5} "
            f"{float(row['dur_mean']):<15.2f} "
            f"{float(row['ar_mean']):<15.4f}"
        )
    print("=" * 80)


def save_current_plot(name: str, plot_dir: str | Path, figure=None) -> None:
    """Save the current matplotlib figure as PDF and PNG."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig = figure if figure is not None else plt.gcf()
    for ext, kwargs in {"pdf": {}, "png": {"dpi": 300}}.items():
        path = plot_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
    print(f"Saved: {plot_dir / name}.pdf and .png")


def shared_approx_ylim(*series_list, pad_fraction: float = 0.08) -> tuple[float, float] | None:
    """Return a shared approximation-ratio y-limit across several series."""
    values: list[float] = []
    for series in series_list:
        if series is None:
            continue
        values.extend(pd.to_numeric(series, errors="coerce").dropna().tolist())
    if not values:
        return None
    y_min = min(values)
    y_max = max(values)
    pad = 0.01 if y_min == y_max else max((y_max - y_min) * pad_fraction, 0.005)
    return max(0.0, y_min - pad), min(1.0, y_max + pad)


def shared_approx_yticks(ylim: tuple[float, float] | None, n_ticks: int = 6) -> list[float] | None:
    """Return evenly spaced ticks for a shared approximation-ratio y-limit."""
    if ylim is None:
        return None
    y_min, y_max = ylim
    step = (y_max - y_min) / float(n_ticks - 1)
    return [y_min + step * i for i in range(n_ticks)]


def apply_shared_approx_axis(
    ax=None,
    *,
    ylim: tuple[float, float] | None = None,
    yticks: list[float] | None = None,
) -> None:
    """Apply shared approximation-ratio y-axis limits and formatting."""
    if ylim is None:
        return
    axis = ax if ax is not None else plt.gca()
    axis.set_ylim(*ylim)
    if yticks is not None:
        axis.yaxis.set_major_locator(FixedLocator(yticks))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def curve_label(df: pd.DataFrame, default: str) -> str:
    """Build a compact method label from strategy, simulator, and depth columns."""
    if df.empty:
        return default
    strategy = (
        df["strategy"].dropna().astype(str).iloc[0]
        if "strategy" in df and df["strategy"].notna().any()
        else default
    )
    simulator = (
        df["simulation_method"].dropna().astype(str).iloc[0]
        if "simulation_method" in df and df["simulation_method"].notna().any()
        else None
    )
    p_val = float(df["p"].dropna().iloc[0]) if "p" in df and df["p"].notna().any() else None
    parts = []
    if simulator is not None:
        parts.append(simulator)
    if p_val is not None:
        parts.append(f"p={p_val:g}")
    return f"{strategy} ({', '.join(parts)})" if parts else strategy


def prepare_monotone_curve(
    df: pd.DataFrame,
    resource_col: str = "resource",
    response_col: str = "response",
) -> pd.DataFrame:
    """Average duplicate resources and add a monotone best-so-far response."""
    keep_cols = [
        col
        for col in [resource_col, response_col, "response_lower", "response_upper"]
        if col in df.columns
    ]
    curve = (
        df.loc[:, keep_cols]
        .dropna(subset=[resource_col, response_col])
        .sort_values(resource_col)
        .groupby(resource_col, as_index=False)
        .mean(numeric_only=True)
    )
    curve["response_monotone"] = curve[response_col].cummax()
    return curve


def resolve_result_root(tag_or_path: str | Path, results_base: str | Path) -> Path:
    """Resolve either an absolute result path or a tag under a results base."""
    path = Path(tag_or_path)
    if not path.is_absolute():
        path = Path(results_base) / path
    return path.resolve()


def read_summary_csv(root: str | Path, name: str) -> pd.DataFrame:
    """Read a result summary CSV when present, otherwise return an empty frame."""
    path = Path(root) / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def attach_result_metadata(
    df: pd.DataFrame,
    result_tag: str,
    root: str | Path,
    label: str,
) -> pd.DataFrame:
    """Attach result metadata columns used by multi-strategy notebook plots."""
    if df.empty:
        return df
    out = df.copy()
    out["result_tag"] = result_tag
    out["result_root"] = str(root)
    out["method_label"] = label
    return out


def load_multi_strategy_summaries(
    result_tags: list[str | Path],
    results_base: str | Path,
) -> list[dict[str, object]]:
    """Load the summary CSV bundle for several PSS campaign result roots."""
    rows: list[dict[str, object]] = []
    missing = []
    for tag in result_tags:
        root = resolve_result_root(tag, results_base)
        if not root.exists():
            missing.append((tag, root, "root"))
            continue

        strategy_budget = read_summary_csv(root, "strategy_budget_summary_train.csv")
        actionable_lookup = read_summary_csv(root, "actionable_pss_lookup_train.csv")
        actionable_fit = read_summary_csv(root, "actionable_pss_fit_train.csv")
        window_sticker = read_summary_csv(root, "window_sticker_summary.csv")
        projection = read_summary_csv(root, "projection_summary.csv")
        virtual_best = read_summary_csv(root, "virtual_best_summary.csv")

        label_source = strategy_budget
        for candidate in [
            label_source,
            window_sticker,
            projection,
            actionable_lookup,
            actionable_fit,
            virtual_best,
        ]:
            if not candidate.empty:
                label_source = candidate
                break
        label = curve_label(label_source, Path(tag).name)
        result_tag = Path(tag).name

        rows.append(
            {
                "result_tag": result_tag,
                "root": root,
                "method_label": label,
                "strategy_budget": attach_result_metadata(strategy_budget, result_tag, root, label),
                "actionable_lookup": attach_result_metadata(actionable_lookup, result_tag, root, label),
                "actionable_fit": attach_result_metadata(actionable_fit, result_tag, root, label),
                "window_sticker": attach_result_metadata(window_sticker, result_tag, root, label),
                "projection": attach_result_metadata(projection, result_tag, root, label),
                "virtual_best": attach_result_metadata(virtual_best, result_tag, root, label),
            }
        )
    if missing:
        print("Missing result roots:")
        for tag, root, _ in missing:
            print(f"  {tag}: {root}")
    return rows


def concat_summary(strategy_summaries: list[dict[str, object]], key: str) -> pd.DataFrame:
    """Concatenate a named summary table across loaded strategy roots."""
    frames = [item[key] for item in strategy_summaries if not item[key].empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def curve_from_training_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a monotone curve from per-budget training summary rows."""
    required = {"method_label", "T", "response_mean"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["method_label", "T", "response"])
    curve = (
        df.loc[:, ["method_label", "T", "response_mean"]]
        .dropna(subset=["T", "response_mean"])
        .sort_values(["method_label", "T"])
        .groupby(["method_label", "T"], as_index=False)
        .first()
        .rename(columns={"response_mean": "response"})
    )
    curve["response_monotone"] = curve.groupby("method_label")["response"].cummax()
    return curve


def curve_from_window_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a monotone curve from Window Sticker projection rows."""
    required = {"method_label", "resource", "response"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["method_label", "resource", "response"])
    curve = (
        df.loc[:, ["method_label", "resource", "response"]]
        .dropna(subset=["resource", "response"])
        .sort_values(["method_label", "resource"])
        .groupby(["method_label", "resource"], as_index=False)
        .first()
    )
    curve["response_monotone"] = curve.groupby("method_label")["response"].cummax()
    return curve


def cross_strategy_envelope(
    curve_df: pd.DataFrame,
    resource_col: str,
    response_col: str = "response_monotone",
) -> pd.DataFrame:
    """Build the best-so-far envelope across method labels."""
    if curve_df.empty:
        return pd.DataFrame(columns=[resource_col, response_col, "method_label"])
    rows = []
    labels = sorted(curve_df["method_label"].dropna().unique())
    resources = sorted(pd.to_numeric(curve_df[resource_col], errors="coerce").dropna().unique())
    best_response = -np.inf
    best_label = None
    for resource in resources:
        candidates = []
        for label in labels:
            group = curve_df[curve_df["method_label"].eq(label)]
            eligible = group[pd.to_numeric(group[resource_col], errors="coerce") <= resource]
            if eligible.empty:
                continue
            row = eligible.sort_values(resource_col).iloc[-1]
            candidates.append((float(row[response_col]), label))
        if not candidates:
            continue
        response, label = max(candidates, key=lambda item: item[0])
        if response >= best_response:
            best_response = response
            best_label = label
        rows.append({resource_col: resource, response_col: best_response, "method_label": best_label})
    return pd.DataFrame(rows)


def plot_method_curves(
    curve_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
    resource_col: str,
    ylabel: str,
    title: str,
    filename: str,
    *,
    plot_dir: str | Path,
    approx_ylim: tuple[float, float] | None = None,
    approx_yticks: list[float] | None = None,
) -> None:
    """Plot per-method curves and their cross-strategy envelope."""
    if curve_df.empty:
        print(f"Skipping {title}: no curve data found.")
        return
    plt.figure(figsize=(8.5, 5))
    for label, group in curve_df.groupby("method_label"):
        group = group.sort_values(resource_col)
        plt.plot(group[resource_col], group["response_monotone"], linewidth=2.0, label=label)
    if not envelope_df.empty:
        plt.plot(
            envelope_df[resource_col],
            envelope_df["response_monotone"],
            color="black",
            linewidth=2.8,
            linestyle="--",
            label="Envelope across strategies",
        )
    plt.xscale("log")
    plt.xlabel(r"Resource $T_{\mathrm{proxy}}$")
    plt.ylabel(ylabel)
    apply_shared_approx_axis(ylim=approx_ylim, yticks=approx_yticks)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    save_current_plot(filename, plot_dir)
    plt.show()

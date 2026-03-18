import re
import math
import os
from typing import Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator, MultipleLocator


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

    # ---------- Filter ----------
    df = df_samples[(df_samples["instance_name"] == instance_name) &
                    (df_samples["job_p"] == job_p)].copy()
    if df.empty:
        raise ValueError("No rows found for that (instance_name, job_p).")

    # ---------- Top 1% probability mass per method ----------
    df_top = (
        df.sort_values(["training_method", "approximation_ratio"], ascending=[True, False])
          .groupby("training_method", group_keys=False)
          .apply(lambda g: g.head(1) if g["prob"].iloc[0] > 0.01
                 else g[g["prob"].cumsum() <= 0.01])
    )

    def _plot_one(dfin, title_suffix):
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
    "TQA": "TQA",
}
_REOPT_METHODS = {"FA", "TQA"}


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
    has_opt = "opt" in parts
    method_str = _METHOD_NAMES.get(method_key, method_key)
    if has_opt and method_key in _REOPT_METHODS:
        method_str = "reoptimized " + method_str
    return f"{method_str} with {evaluator}" if evaluator else method_str


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
    df_plot["total_duration_s"] = pd.to_numeric(df_plot["total duration"], errors="coerce")
    df_plot["approximation_ratio"] = pd.to_numeric(df_plot["approximation_ratio"], errors="coerce")
    df_plot["job_p"] = pd.to_numeric(df_plot["job_p"], errors="coerce")

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

    df_plot["group_label"] = df_plot.get(
        "trainer_label", df_plot.get("training_method", "method")
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
    fs_label = 26
    fs_title = 28
    fs_legend = 18
    centroid_ms = 22
    scatter_ms = 8
    y_max = 1.00
    y_major = 0.10
    y_cushion = 0.03
    eb_opts = {"mec": "k", "ecolor": "k", "capsize": 3, "elinewidth": 1.2}

    _ensure_save_dir(save_dir)

    df_points = plot_data["df_points"]
    color_map = plot_data["color_map"]
    shape_map = plot_data["shape_map"]
    label_map = plot_data["label_map"]
    graph_id = plot_data["graph_id"]

    p_vals = sorted(df_points["job_p"].unique())
    n_panels = len(p_vals)

    x_all_global = df_points["total_duration_s"].dropna()
    x_lo_global = max(x_all_global.min() / 3, 0.1)
    x_hi_global = x_all_global.max() * 3

    y_all_global = df_points["approximation_ratio"].dropna()
    y_lo_global = max(0.0, float(y_all_global.min()) - y_cushion)

    fig, axs = plt.subplots(1, n_panels, figsize=(9 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axs = [axs]

    legend_dict = {}

    for ax_idx, (ax, p) in enumerate(zip(axs, p_vals)):
        d = df_points[df_points["job_p"] == p]

        for (_, color_label), group in d.groupby(["group_label", "color_label"]):
            x = group["total_duration_s"]
            y = group["approximation_ratio"]
            color = color_map[color_label]
            marker = shape_map[color_label]

            ax.errorbar(
                x,
                y,
                fmt=marker,
                linestyle="none",
                color=color,
                alpha=0.35,
                ms=scatter_ms,
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
                ms=centroid_ms,
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
                    markersize=10,
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
        ax.set_ylim(y_lo_global, y_max)
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
        ax.grid(True)

    axs[0].set_ylabel("Approximation ratio", fontsize=fs_label)

    sorted_items = sorted(legend_dict.items(), key=lambda x: x[0])
    all_legend_handles = [handle for _, handle in sorted_items]

    fig.legend(
        handles=all_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        borderaxespad=0,
        frameon=True,
        fontsize=fs_legend,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

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
    fs_label = 26
    fs_legend = 15
    scatter_ms = 8

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

    valid_methods = []
    for color_label, group in centroids.groupby("color_label"):
        if group["job_p"].nunique() >= 2:
            valid_methods.append(color_label)

    centroids = centroids[centroids["color_label"].isin(valid_methods)].copy()

    x_vals = overlay["total_duration_s"].dropna()
    x_lo = max(x_vals.min() / 3, 0.1)
    x_hi = x_vals.max() * 3
    y_lo = max(0.0, float(overlay["approximation_ratio"].min()) - 0.03)

    fig, ax = plt.subplots(figsize=(18, 6))

    for color_label, group in inst_high_p.groupby("color_label"):
        if color_label not in valid_methods:
            continue
        ax.errorbar(
            group["total_duration_s"],
            group["approximation_ratio"],
            fmt=shape_map[color_label],
            linestyle="none",
            color=color_map[color_label],
            alpha=0.28,
            ms=scatter_ms,
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

        for _, row in group.iterrows():
            p = int(row["job_p"])
            if p == p_max:
                ax.errorbar(
                    row["dur_mean"],
                    row["ar_mean"],
                    xerr=row["dur_sem"],
                    yerr=row["ar_sem"],
                    fmt=marker,
                    ms=18,
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
                    ms=9,
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
    ax.set_ylim(y_lo, 1.00)
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.grid(True)
    ax.set_xlabel("Total duration (s)", fontsize=fs_label)
    ax.set_ylabel("Approximation ratio", fontsize=fs_label)
    ax.tick_params(axis="both", labelsize=fs_tick)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=shape_map[color_label],
            color=color_map[color_label],
            markeredgecolor="k",
            lw=0,
            markersize=10,
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

    ax.legend(
        handles=method_handles + depth_handles + cue_handles,
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
    fs_label = 26
    fs_legend = 18
    edge_lw = 0.9

    _ensure_save_dir(save_dir)

    methods = sorted(agg["method_base"].unique())
    depths = sorted(agg["job_p"].dropna().unique())
    x = np.arange(len(depths))

    idx_p5 = depths.index(5) if 5 in depths else None
    p5_left = (idx_p5 - 0.5) if idx_p5 is not None else None
    p5_right = (idx_p5 + 0.5) if idx_p5 is not None else None

    main_ymax = 1500.0
    overflow_methods = sorted(
        [m for m in methods if any(agg[agg["method_base"] == m]["brick_total"].values > main_ymax)]
    )
    inset_ymin = main_ymax

    agg_overflow = agg[agg["method_base"].isin(overflow_methods)].copy()
    if not agg_overflow.empty:
        inset_top_needed = (
            agg_overflow["brick_total"] + agg_overflow["sem_total"].fillna(0)
        ).max()
        inset_ymax = max(45000.0, float(inset_top_needed) * 1.08)
    else:
        inset_ymax = 45000.0

    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.5, 1.5], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    inset_ax = fig.add_subplot(gs[1])

    if idx_p5 is not None:
        ax.axvspan(p5_left, p5_right, color="0.92", zorder=0)
        inset_ax.axvspan(p5_left, p5_right, color="0.92", zorder=0)

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
    fig.legend(
        handles,
        legend_labels,
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
    fs_label = 26
    fs_legend = 18

    _ensure_save_dir(save_dir)

    df_centroids, _ = build_recommendation_data(df_points)

    p_levels = sorted(pd.to_numeric(df_centroids["job_p"], errors="coerce").dropna().astype(int).unique())
    if len(p_levels) <= 1:
        size_map = {int(p_levels[0]): 20} if len(p_levels) == 1 else {}
    else:
        sizes = np.linspace(16, 22, len(p_levels))
        size_map = {int(p): int(round(s)) for p, s in zip(p_levels, sizes)}
    scatter_size_map = {p: max(10, size_map[p] - 6) for p in size_map}

    x_all = df_centroids["dur_mean"].dropna()
    x_left = max(x_all.min() / 3, 0.1)
    x_right = 1e4

    fig, ax = plt.subplots(figsize=(18, 8))

    if not df_frontier.empty:
        for seg_idx in range(len(df_frontier)):
            row = df_frontier.iloc[seg_idx]
            color_seg = color_map[row["color_label"]]
            x_start = x_left if seg_idx == 0 else row["dur_mean"]
            x_end = (
                df_frontier.iloc[seg_idx + 1]["dur_mean"]
                if seg_idx < len(df_frontier) - 1
                else x_right
            )
            y_this = row["ar_mean"]

            ax.plot(
                [x_start, x_end],
                [y_this, y_this],
                color=color_seg,
                linewidth=3,
                linestyle="--",
                alpha=0.95,
                zorder=9,
            )

            if seg_idx > 0:
                y_prev = df_frontier.iloc[seg_idx - 1]["ar_mean"]
                ax.plot(
                    [row["dur_mean"], row["dur_mean"]],
                    [y_prev, y_this],
                    color=color_seg,
                    linewidth=3,
                    linestyle="--",
                    alpha=0.95,
                    zorder=9,
                )

    frontier_pairs = df_frontier[["color_label", "job_p"]].drop_duplicates()
    frontier_clabs = sorted(frontier_pairs["color_label"].unique())
    frontier_ps = sorted(int(p) for p in frontier_pairs["job_p"].dropna().unique())

    for _, pair in frontier_pairs.iterrows():
        color_label = pair["color_label"]
        p_val = int(pair["job_p"])
        inst_sub = df_points[
            (df_points["color_label"] == color_label) & (df_points["job_p"] == p_val)
        ]
        if inst_sub.empty:
            continue
        ax.plot(
            inst_sub["total_duration_s"],
            inst_sub["approximation_ratio"],
            linestyle="",
            marker=shape_map[color_label],
            ms=scatter_size_map.get(p_val, 10),
            mfc=color_map[color_label],
            mec="k",
            mew=1.1,
            alpha=0.35,
            zorder=6,
        )

    for _, row in df_frontier.iterrows():
        p = int(row["job_p"])
        ax.plot(
            row["dur_mean"],
            row["ar_mean"],
            marker=shape_map[row["color_label"]],
            ms=size_map.get(p, 20),
            mfc=color_map[row["color_label"]],
            mec="k",
            mew=2.0,
            linestyle="",
            zorder=11,
        )

    ax.set_xlabel("Total duration (s)", fontsize=fs_label)
    ax.set_ylabel("Approximation ratio", fontsize=fs_label)
    ax.set_xscale("log")
    ax.set_xlim(x_left, x_right)

    if not df_frontier.empty:
        y_min_data = float(df_frontier["ar_mean"].min())
        y_lo = max(0.50, y_min_data - 0.06)
    else:
        y_lo = 0.50
    ax.set_ylim(y_lo, 1.00)

    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=100))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=False))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.minorticks_off()
    ax.tick_params(axis="both", which="major", labelsize=fs_tick, length=6)
    ax.grid(True)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=shape_map[color_label],
            color=color_map[color_label],
            markeredgecolor="k",
            markersize=10,
            linestyle="",
            label=label_map[color_label],
        )
        for color_label in frontier_clabs
    ]

    for p in frontier_ps:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="gray",
                markeredgecolor="k",
                markersize=size_map.get(p, 16),
                linestyle="",
                label=f"p={p}",
            )
        )

    legend_handles.append(
        Line2D([0], [0], color="black", linewidth=2.5, linestyle="--", label="Pareto Frontier")
    )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=5,
        borderaxespad=0,
        fontsize=fs_legend,
        frameon=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)

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

import re
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.ticker import (
    FixedLocator,
    FormatStrFormatter,
    FuncFormatter,
    LogFormatterMathtext,
    LogLocator,
    MaxNLocator,
    MultipleLocator,
)
from matplotlib.transforms import Bbox

from .approx_ratio_calc import (
    extract_minmax_args as _extract_minmax_args,
    get_minmax as _get_minmax,
    maxcut_approximation_ratio as _maxcut_approximation_ratio,
    maxcut_energy_from_bitstring as _maxcut_energy_from_bitstring,
)

WINDOW_STICKER_LABEL_FONTSIZE = 16
WINDOW_STICKER_TICK_FONTSIZE = 14
WINDOW_STICKER_LEGEND_FONTSIZE = 13
WINDOW_STICKER_TITLE_FONTSIZE = 16

try:
    from qaoa_parameter_setting.utils.labels import (
        format_method_label_to as _qps_external_format_method_label_to,
        trainer_config_to_method_label as _qps_external_trainer_config_to_method_label,
    )
except Exception:
    _qps_external_format_method_label_to = None
    _qps_external_trainer_config_to_method_label = None


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


def _relativize_warning_filename(filename: str, workspace_root: str | Path) -> str:
    """Scrub a warning's absolute source-file path down to a portable one.

    ``warnings.warn``'s default formatter prints the absolute source-file
    path of the call site, bypassing dataframe-path scrubbing conventions
    like :func:`_relativize_paths` entirely -- e.g. a duplicate-resource
    warning from ``src/interpolate.py`` would otherwise leak a contributor's
    local filesystem layout into committed notebook output. This strips it
    down to a path relative to ``workspace_root`` (or the container
    ``/workspace`` mount prefix), matching :func:`_relativize_paths`' own
    convention.
    """
    try:
        return str(Path(filename).resolve().relative_to(workspace_root))
    except ValueError:
        pass
    if filename.startswith("/workspace/"):
        return filename[len("/workspace/"):]
    return filename


def _scrubbed_formatwarning(
    message, category, filename, lineno, line=None, *, workspace_root: str | Path
) -> str:
    """A ``warnings.formatwarning`` replacement that scrubs the source path.

    ``workspace_root`` is keyword-only and has no default since
    ``warnings.formatwarning`` is called positionally by the ``warnings``
    module; bind it with ``functools.partial`` before assigning, e.g.
    ``warnings.formatwarning = functools.partial(_scrubbed_formatwarning, workspace_root=WORKSPACE_ROOT)``.
    """
    return f"{_relativize_warning_filename(filename, workspace_root)}:{lineno}: {category.__name__}: {message}\n"


def _relativize_paths(df: pd.DataFrame, cols: Iterable[str], base: str | Path) -> pd.DataFrame:
    """Scrub absolute filesystem paths in ``cols`` down to a path relative to ``base``.

    Falls back to stripping a container ``/workspace/`` mount prefix when a
    path isn't under ``base`` (e.g. it came from a Nautilus job), and leaves
    a path unscrubbed (rather than raising) when neither applies, since this
    is a display convenience only.
    """
    df = df.copy()

    def _rel(p):
        if not pd.notna(p):
            return p
        try:
            return str(Path(p).resolve().relative_to(base))
        except ValueError:
            pass
        p_str = str(p)
        if p_str.startswith("/workspace/"):
            return p_str[len("/workspace/"):]
        return p_str

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(_rel)
    return df


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
        ax.set_xticklabels(
            [_compact_method_label(method) for method in methods],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("Approximation Ratio")
        ax.set_xlabel("Training method")
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


_METHOD_NAMES = {
    "FA": "fixed angles",
    "F": "Fourier",
    "I": "INTERP",
    "LR": "linear ramp",
    "PT": "parameter transfer",
    "RTS": "recursive transition states",
    "TQA": "TQA",
}
_QPS_METHOD_LABELS = {
    "F.json": "Fourier*",
    "FAer.json": "Fourier*",
    "FA_no_opt.json": "Fixed Angles†",
    "FA_opt.json": "Fixed Angles*",
    "FAAer_opt.json": "Fixed Angles*",
    "FAAer_no_opt.json": "Fixed Angles†",
    "I.json": "Interp.*",
    "I_opt.json": "Interp.*",
    "IAer.json": "Interp.*",
    "IAer_opt.json": "Interp.*",
    "I_no_opt.json": "Interp.†",
    "IAer_no_opt.json": "Interp.†",
    "LR_opt.json": "Linear Ramp",
    "LRAer_opt.json": "Linear Ramp",
    "LR_no_opt.json": "Linear Ramp†",
    "LRAer_no_opt.json": "Linear Ramp†",
    "LR_angle_opt.json": "Linear Ramp*",
    "LRAer_angle_opt.json": "Linear Ramp*",
    "RTS.json": "Recursive TS*",
    "RTSAer.json": "Recursive TS*",
    "TS.json": "Recursive TS*",
    "TQA_no_opt.json": "TQA†",
    "TQA_opt.json": "TQA*",
    "TQAAer_opt.json": "TQA*",
    "TQAAer_no_opt.json": "TQA†",
    "PT_AAAM.json": "Param. Transfer",
    "PT_AAA.json": "Param. Transfer",
    "PT_no_opt.json": "Param. Transfer†",
    "PT_AAAM_no_opt.json": "Param. Transfer†",
    "PT_AAA_no_opt.json": "Param. Transfer†",
}
_QPS_METHOD_PREFIXES = tuple(
    sorted(
        {name.removesuffix(".json").split("_", 1)[0] for name in _QPS_METHOD_LABELS},
        key=len,
        reverse=True,
    )
)
QPS_METHOD_COLORS = {
    "Fixed Angles*": "#4477AA",
    "Fixed Angles": "#4477AA",
    "Fixed Angles†": "#4477AA",
    "Fourier*": "#EE6677",
    "Interp.*": "#228833",
    "Interp.": "#228833",
    "Linear Ramp*": "#CCBB44",
    "Linear Ramp": "#CCBB44",
    "Linear Ramp†": "#CCBB44",
    "Recursive TS*": "#66CCEE",
    "TQA*": "#AA3377",
    "TQA": "#AA3377",
    "TQA†": "#AA3377",
    "Param. Transfer": "#BBBBBB",
    "Param. Transfer†": "#BBBBBB",
    "Parameter Transfer": "#BBBBBB",
}
QPS_EVALUATOR_MARKERS = {
    "SV": "o",
    "MPS (Quimb)": "P",
    "MPS (Aer)": "^",
    "PP": "s",
}
_STYLE_FALLBACK_COLORS = list(QPS_METHOD_COLORS.values()) + [
    "#44AA99",
    "#999933",
    "#882255",
    "#117733",
]


def _compact_method_label(label: str) -> str:
    """Build the compact display label used in IBM QAOA plot legends."""
    method_label = _method_label_from_training_method(label, format="latex")
    evaluator = _evaluation_label_from_training_method(label)
    return f"{method_label} with {evaluator}" if evaluator else method_label


def _plain_method_label_from_training_method(label: str) -> str:
    """Return the unformatted QPS method label used as a style lookup key."""
    config_name = _normalise_training_method_to_config(label)
    if _qps_external_trainer_config_to_method_label is not None:
        try:
            return _qps_external_trainer_config_to_method_label(config_name)
        except Exception:
            pass

    method_name = _method_config_to_method(config_name)
    method_label = _QPS_METHOD_LABELS.get(method_name)
    if method_label is None:
        parts = method_name.removesuffix(".json").split("_")
        method_key = parts[0] if parts else method_name.removesuffix(".json")
        method_label = _METHOD_NAMES.get(method_key, method_key)
    return method_label


def _method_color_from_training_method(label: str) -> str:
    """Return paper-style color keyed only by angle-setting method."""
    method_label = _plain_method_label_from_training_method(label)
    if method_label in QPS_METHOD_COLORS:
        return QPS_METHOD_COLORS[method_label]
    base_label = method_label.replace("*", "").replace("†", "")
    return QPS_METHOD_COLORS.get(
        base_label,
        _STYLE_FALLBACK_COLORS[abs(hash(base_label)) % len(_STYLE_FALLBACK_COLORS)],
    )


def _window_sticker_method_color(label: str) -> str:
    """Return paper-style color for Window Sticker labels that may include depth text."""
    label_str = str(label)
    cleaned = re.sub(r"\s*\(p\s*=\s*\d+\)\s*$", "", label_str)
    if cleaned in QPS_METHOD_COLORS:
        return QPS_METHOD_COLORS[cleaned]
    base_label = cleaned.replace("*", "").replace("†", "")
    if base_label in QPS_METHOD_COLORS:
        return QPS_METHOD_COLORS[base_label]
    return _method_color_from_training_method(cleaned)


def window_sticker_method_color(label: str) -> str:
    """Return the paper-style base color for a Window Sticker method label."""
    label_str = str(label)
    cleaned = re.sub(r"\s*\(p\s*=\s*\d+\)\s*$", "", label_str)
    base_label = re.sub(r"\$?\s*\^?\s*\{?\s*\\(?:star|dagger)\s*\}?\s*\$?", "", cleaned)
    base_label = re.sub(r"[\*†★⋆]", "", base_label).strip()
    base_label = re.sub(r"\s+", " ", base_label)
    base_lower = base_label.lower()
    if "fixed angles" in base_lower:
        return "#4477AA"
    if "linear ramp" in base_lower:
        return "#CCBB44"
    if "param. transfer" in base_lower or "parameter transfer" in base_lower:
        return "#BBBBBB"
    if "interp" in base_lower:
        return "#228833"
    if "fourier" in base_lower:
        return "#EE6677"
    if "recursive" in base_lower:
        return "#66CCEE"
    if "tqa" in base_lower:
        return "#AA3377"
    return _window_sticker_method_color(label_str)


def _window_sticker_label_base(label: str) -> str:
    """Return a normalized method key, without depth or optimization markers."""
    label_str = str(label)
    cleaned = re.sub(r"\s*\(p\s*=\s*\d+\)\s*$", "", label_str)
    cleaned = re.sub(r"\$?\s*\^?\s*\{?\s*\\(?:star|dagger)\s*\}?\s*\$?", "", cleaned)
    cleaned = re.sub(r"[\*†★⋆]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _window_sticker_label_depth(label: str) -> int | None:
    """Extract the plotted QAOA depth from a Window Sticker label, when present."""
    match = re.search(r"\(p\s*=\s*(\d+)\)\s*$", str(label))
    return int(match.group(1)) if match else None


def _shade_color(color: str, amount: float) -> tuple[float, float, float]:
    """Lighten (positive) or darken (negative) a color while keeping the hue."""
    c = np.array(mcolors.to_rgb(color), dtype=float)
    if amount >= 0:
        shaded = c + (1.0 - c) * amount
    else:
        shaded = np.clip(c * (1.0 + amount), 0.0, 1.0)
    return tuple(float(channel) for channel in shaded)


def window_sticker_method_color_map(labels: Iterable[str]) -> dict[str, object]:
    """Return method colors with small depth-based shade offsets for duplicates."""
    label_list = [str(label) for label in labels]
    grouped: dict[str, list[str]] = {}
    for label in label_list:
        grouped.setdefault(_window_sticker_label_base(label), []).append(label)

    color_map: dict[str, object] = {}
    for group_labels in grouped.values():
        unique_group_labels = sorted(
            set(group_labels),
            key=lambda item: (
                _window_sticker_label_depth(item) is None,
                _window_sticker_label_depth(item) if _window_sticker_label_depth(item) is not None else 10**9,
                item,
            ),
        )
        if len(unique_group_labels) == 1:
            color_map[unique_group_labels[0]] = window_sticker_method_color(unique_group_labels[0])
            continue

        midpoint = (len(unique_group_labels) - 1) / 2.0
        for idx, label in enumerate(unique_group_labels):
            # Keep variants close to the method's canonical color while separating depths.
            offset = (idx - midpoint) * 0.22
            offset = float(np.clip(offset, -0.26, 0.26))
            color_map[label] = _shade_color(window_sticker_method_color(label), offset)
    return color_map


def window_sticker_curve_colors(label: str) -> dict[str, object]:
    """Return distinct same-family colors for Window Sticker curves for one method."""
    base = window_sticker_method_color(label)
    return {
        "base": base,
        "virtual_best": base,
        "averaged_prescription": _lighten_color(base, 0.28),
        "actionable_prescription": _lighten_color(base, 0.48),
    }


def _marker_from_training_method(label: str) -> str:
    """Return paper-style marker keyed only by evaluator."""
    evaluator = _evaluation_label_from_training_method(label)
    return QPS_EVALUATOR_MARKERS.get(evaluator, "o")


def _style_plot_kwargs(label: str) -> dict[str, object]:
    """Return guide-compliant kwargs for line/point plots."""
    method_label = _plain_method_label_from_training_method(label)
    color = _method_color_from_training_method(label)
    marker = _marker_from_training_method(label)
    if "†" in method_label:
        return {
            "color": color,
            "marker": marker,
            "markerfacecolor": "white",
            "markeredgecolor": color,
            "markeredgewidth": 1.2,
        }
    return {
        "color": color,
        "marker": marker,
        "markerfacecolor": color,
        "markeredgecolor": "k" if "*" in method_label else "none",
        "markeredgewidth": 1.0 if "*" in method_label else 0.0,
    }


def _optimization_level(label: str) -> int:
    """Return 0=no opt, 1=method-parameter opt, 2=full angle opt."""
    method_label = _method_label_from_training_method(label, format="text")
    if "*" in method_label:
        return 2
    if "†" in method_label:
        return 0
    return 1


def _optimization_alpha(label: str) -> float:
    """Return opacity keyed by optimization level for bar plots."""
    return {0: 0.55, 1: 0.78, 2: 1.0}[_optimization_level(label)]


def _evaluator_edge_width(label: str, base_lw: float = 0.9) -> float:
    """Return solid bar edge thickness keyed by evaluator."""
    evaluator = _evaluation_label_from_training_method(label)
    if evaluator == "PP":
        return base_lw * 3.4
    if evaluator == "MPS (Aer)":
        return base_lw
    if evaluator == "MPS (Quimb)":
        return base_lw * 2.1
    return base_lw


def _qps_format_method_label(label: str, format: str = "latex") -> str:
    """Format QAOA Parameter Setting label markers for display."""
    if format in {"latex", "siunitx"}:
        return label.replace("*", r"$^\star$").replace("†", r"$^\dagger$")
    return label


def _normalise_training_method_to_config(label: str) -> str:
    """Convert IBM method strings/filenames to canonical QPS config names."""
    value = str(label).split("|", 1)[0].strip()
    # Our own zero-training strategy name (see ZERO_TRAINING_METHODS in
    # simulation_validation.py) spells out "linear_ramp" instead of the "LR"
    # prefix every other Linear Ramp config uses, so neither the prefix
    # matching below nor the external QPS label formatter (tried first in
    # _method_label_from_training_method) ever recognizes it -- it silently
    # falls through to displaying the raw method name. Translate it to the
    # equivalent canonical name directly.
    if value == "linear_ramp_no_opt":
        return "LR_no_opt.json"
    basename = Path(value).name
    stem = basename[:-5] if basename.endswith(".json") else basename
    parts = stem.split("_")

    start_idx = None
    for idx, part in enumerate(parts):
        if part in _QPS_METHOD_PREFIXES:
            start_idx = idx
            break
    if start_idx is not None:
        stem = "_".join(parts[start_idx:])

    stem = re.sub(r"_\d+$", "", stem)
    stem = re.sub(r"angleOpt(?:MW|BD)?\d*", "angle_opt", stem)
    stem = re.sub(r"noOpt(?:MW|BD)?\d*", "no_opt", stem)
    stem = re.sub(r"opt(?:MW|BD)\d*", "opt", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return f"{stem}.json"


def _method_config_to_method(config_name: str) -> str:
    """Drop the evaluator token from a canonical trainer config name."""
    return re.sub(r"_(?:MPSAer|MPS|PP|SV)(?=_|\.json$)", "", config_name)


def _method_label_from_training_method(label: str, format: str = "latex") -> str:
    """Map IBM method keys through the QPS method-label convention."""
    config_name = _normalise_training_method_to_config(label)
    if (
        _qps_external_trainer_config_to_method_label is not None
        and _qps_external_format_method_label_to is not None
    ):
        try:
            return _qps_external_format_method_label_to(
                _qps_external_trainer_config_to_method_label(config_name),
                format=format,
            )
        except Exception:
            pass

    method_name = _method_config_to_method(config_name)
    method_label = _QPS_METHOD_LABELS.get(method_name)

    if method_label is None:
        parts = method_name.removesuffix(".json").split("_")
        method_key = parts[0] if parts else method_name.removesuffix(".json")
        method_label = _METHOD_NAMES.get(method_key, method_key)

    return _qps_format_method_label(method_label, format=format)


def _evaluation_label_from_training_method(label: str) -> str:
    """Return the canonical evaluator label used by QPS plots."""
    config_name = _normalise_training_method_to_config(label)
    if re.search(r"_MPSAer(?=_|\.json$)", config_name):
        return "MPS (Aer)"
    if re.search(r"_MPS(?=_|\.json$)", config_name):
        return "MPS (Quimb)"
    if re.search(r"_PP(?=_|\.json$)", config_name):
        return "PP"
    if re.search(r"_SV(?=_|\.json$)", config_name):
        return "SV"
    return ""


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
        base = color_map.get(
            method,
            QPS_METHOD_COLORS.get(
                method,
                QPS_METHOD_COLORS.get(
                    method.replace("*", "").replace("†", ""),
                    plt.get_cmap("tab10")(methods.index(method) % 10),
                ),
            ),
        )
        alpha = _optimization_alpha(method)
        bar_edge_lw = _evaluator_edge_width(method, edge_lw)
        bottom = np.zeros(len(depths))

        outer_vals = sub["outer_init"].to_numpy()
        target_ax.bar(
            xpos,
            outer_vals,
            bw,
            bottom=bottom,
            color=base,
            edgecolor="black",
            linewidth=bar_edge_lw,
            alpha=alpha,
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
                linewidth=bar_edge_lw,
                alpha=alpha,
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

    for label in color_labels_all:
        color_map[label] = _method_color_from_training_method(label)
        shape_map[label] = _marker_from_training_method(label)

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
    fs_legend = 14
    raw_marker_size = 8.0
    centroid_marker_size = 18.0
    legend_marker_size = 8.5
    y_scale = 100.0
    y_cushion = 3.0
    eb_opts = {"ecolor": "k", "capsize": 3, "elinewidth": 1.2}

    _ensure_save_dir(save_dir)

    df_points = plot_data["df_points"]
    color_map = plot_data["color_map"]
    shape_map = plot_data["shape_map"]
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

    method_legend_dict = {}
    evaluator_legend_dict = {}

    for ax_idx, (ax, p) in enumerate(zip(axs, p_vals)):
        d = df_points[df_points["job_p"] == p]

        for (_, color_label), group in d.groupby(["group_label", "color_label"]):
            x = group["total_duration_s"]
            y = y_scale * group["approximation_ratio"]
            color = color_map[color_label]
            marker = shape_map[color_label]
            style_kwargs = _style_plot_kwargs(color_label)

            ax.errorbar(
                x,
                y,
                fmt=marker,
                linestyle="none",
                color=color,
                mfc=style_kwargs["markerfacecolor"],
                mec=style_kwargs["markeredgecolor"],
                mew=style_kwargs["markeredgewidth"],
                alpha=0.35,
                ms=raw_marker_size,
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
                ms=centroid_marker_size,
                linestyle="none",
                color=color,
                mfc=style_kwargs["markerfacecolor"],
                mec=style_kwargs["markeredgecolor"],
                mew=style_kwargs["markeredgewidth"],
                zorder=10,
                **eb_opts,
            )

            method_label = _method_label_from_training_method(color_label, format="latex")
            if method_label not in method_legend_dict:
                method_style = _style_plot_kwargs(color_label)
                method_legend_dict[method_label] = Line2D(
                    [0],
                    [0],
                    marker="s",
                    color=color,
                    markerfacecolor=method_style["markerfacecolor"],
                    markeredgecolor=method_style["markeredgecolor"],
                    markeredgewidth=max(float(method_style["markeredgewidth"]), 1.0),
                    lw=0,
                    markersize=legend_marker_size,
                    label=method_label,
                )
            evaluator_label = _evaluation_label_from_training_method(color_label)
            if evaluator_label and evaluator_label not in evaluator_legend_dict:
                evaluator_legend_dict[evaluator_label] = Line2D(
                    [0],
                    [0],
                    marker=_marker_from_training_method(color_label),
                    color="black",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    lw=0,
                    markersize=legend_marker_size,
                    label=evaluator_label,
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

    axs[0].set_ylabel("Hardware approximation\nratio (%)", fontsize=fs_label, labelpad=10)

    method_items = sorted(method_legend_dict.items(), key=lambda x: x[0])
    evaluator_items = sorted(evaluator_legend_dict.items(), key=lambda x: x[0])
    axs[0].legend(
        handles=[handle for _, handle in method_items],
        loc="lower left",
        ncol=4,
        borderaxespad=0.7,
        frameon=True,
        fontsize=fs_legend,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.4,
        labelspacing=0.25,
    )
    axs[-1].legend(
        handles=[handle for _, handle in evaluator_items],
        loc="lower left",
        ncol=1,
        borderaxespad=0.7,
        frameon=True,
        fontsize=fs_legend,
        handletextpad=0.6,
        columnspacing=0.9,
        labelspacing=0.35,
    )

    fig.tight_layout(rect=[0.06, 0.03, 1, 0.92])
    fig.subplots_adjust(left=0.11, bottom=0.14, top=0.88)

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
    legend_marker_size = 9.0
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
        style_kwargs = _style_plot_kwargs(color_label)
        ax.errorbar(
            group["total_duration_s"],
            y_scale * group["approximation_ratio"],
            fmt=shape_map[color_label],
            linestyle="none",
            color=color_map[color_label],
            mfc=style_kwargs["markerfacecolor"],
            mec=style_kwargs["markeredgecolor"],
            mew=style_kwargs["markeredgewidth"],
            alpha=0.28,
            ms=raw_ms_map[opt_level],
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
        style_kwargs = _style_plot_kwargs(color_label)

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
                    mfc=style_kwargs["markerfacecolor"],
                    mec=style_kwargs["markeredgecolor"],
                    mew=style_kwargs["markeredgewidth"],
                    alpha=1.0,
                    zorder=11,
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
                    mfc=style_kwargs["markerfacecolor"],
                    mec=style_kwargs["markeredgecolor"],
                    mew=style_kwargs["markeredgewidth"],
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

    method_legend_dict = {}
    evaluator_legend_dict = {}
    for color_label in sorted(valid_methods):
        method_label = _method_label_from_training_method(color_label, format="latex")
        if method_label not in method_legend_dict:
            style_kwargs = _style_plot_kwargs(color_label)
            method_legend_dict[method_label] = Line2D(
                [0],
                [0],
                marker="s",
                color=color_map[color_label],
                markerfacecolor=style_kwargs["markerfacecolor"],
                markeredgecolor=style_kwargs["markeredgecolor"],
                markeredgewidth=max(float(style_kwargs["markeredgewidth"]), 1.0),
                lw=0,
                markersize=legend_marker_size,
                label=method_label,
            )
        evaluator_label = _evaluation_label_from_training_method(color_label)
        if evaluator_label and evaluator_label not in evaluator_legend_dict:
            evaluator_legend_dict[evaluator_label] = Line2D(
                [0],
                [0],
                marker=_marker_from_training_method(color_label),
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                lw=0,
                markersize=legend_marker_size,
                label=evaluator_label,
            )

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
    method_items = sorted(method_legend_dict.items(), key=lambda item: item[0])
    evaluator_items = sorted(evaluator_legend_dict.items(), key=lambda item: item[0])
    method_legend = fig.legend(
        handles=[handle for _, handle in method_items],
        loc="upper left",
        bbox_to_anchor=(0.04, -0.01),
        ncol=4,
        borderaxespad=0,
        frameon=True,
        fontsize=fs_legend,
        handletextpad=0.7,
        columnspacing=1.2,
    )
    evaluator_legend = fig.legend(
        handles=[handle for _, handle in evaluator_items],
        loc="upper left",
        bbox_to_anchor=(0.62, 0.11),
        ncol=1,
        borderaxespad=0,
        frameon=True,
        fontsize=fs_legend,
        handletextpad=0.7,
        columnspacing=1.2,
    )
    cue_legend = fig.legend(
        handles=depth_handles + cue_handles,
        loc="upper left",
        bbox_to_anchor=(0.74, 0.11),
        ncol=2,
        borderaxespad=0,
        frameon=True,
        fontsize=fs_legend,
        handletextpad=0.7,
        columnspacing=1.2,
    )
    fig.add_artist(method_legend)
    fig.add_artist(evaluator_legend)
    fig.add_artist(cue_legend)

    fig.subplots_adjust(left=0.12, right=0.99, top=0.96, bottom=0.38)

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
    sem_df = (
        df_rows.groupby(["job_p", "method_base"])["brick_total"]
        .sem()
        .rename("sem_total")
        .reset_index()
    )
    agg = agg.merge(sem_df, on=["job_p", "method_base"], how="left")

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
        main_ymax = max(main_ymax, float(positive_bounds[0]) * 1.12, 1_000.0)
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
        inset_ymax = max(float(inset_ymin) * 1.4, float(inset_top_needed) * 1.15)
    else:
        inset_ymax = max(float(inset_ymin) * 1.4, float(max_total) * 1.15)

    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.5, 1.5], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    inset_ax = fig.add_subplot(gs[1])

    if shaded_idx is not None:
        ax.axvspan(shaded_left, shaded_right, color="0.86", zorder=0)
        inset_ax.axvspan(shaded_left, shaded_right, color="0.86", zorder=0)

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
    inset_ax.set_yscale("log")
    inset_ax.set_ylim(inset_ymin, inset_ymax)
    inset_ax.set_xticks(x)
    inset_ax.set_xticklabels([int(d) for d in depths], fontsize=fs_tick)
    inset_ax.tick_params(axis="y", labelsize=fs_tick)
    inset_ax.set_xlabel("QAOA depth p", fontsize=fs_label)
    inset_ax.set_ylabel("Mean training duration (s, log scale)", fontsize=fs_label)
    inset_ax.set_xlim(-0.5, x[-1] + 0.5)
    inset_ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    inset_ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    inset_ax.grid(True, which="major", axis="y", alpha=0.45)
    inset_ax.grid(True, which="minor", axis="y", alpha=0.18)

    kw_inset = dict(transform=inset_ax.transAxes, color="k", clip_on=False, lw=1.6)
    inset_ax.plot((-d, +d), (-d, +d), **kw_inset)
    inset_ax.plot((1 - d, 1 + d), (-d, +d), **kw_inset)

    method_legend_items = {}
    for i, method in enumerate(methods):
        method_label = _method_label_from_training_method(method, format="latex")
        if method_label not in method_legend_items:
            method_legend_items[method_label] = Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color_map.get(
                    method,
                    QPS_METHOD_COLORS.get(
                        method,
                        QPS_METHOD_COLORS.get(
                            method.replace("*", "").replace("†", ""),
                            plt.get_cmap("tab10")(i % 10),
                        ),
                    ),
                ),
                edgecolor="black",
                linewidth=_evaluator_edge_width(method, edge_lw),
                alpha=_optimization_alpha(method),
            )

    evaluator_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="black",
            linewidth=_evaluator_edge_width(method, edge_lw),
            label=_evaluation_label_from_training_method(method),
        )
        for method in sorted(
            {
                method
                for method in methods
                if _evaluation_label_from_training_method(method)
            },
            key=_evaluation_label_from_training_method,
        )
    ]
    deduped_evaluator_handles = {}
    for handle in evaluator_handles:
        deduped_evaluator_handles.setdefault(handle.get_label(), handle)

    method_legend = fig.legend(
        handles=list(method_legend_items.values()),
        labels=list(method_legend_items.keys()),
        loc="upper left",
        bbox_to_anchor=(0.04, 0.06),
        ncol=4,
        borderaxespad=0,
        frameon=True,
        fontsize=fs_legend,
        handletextpad=0.7,
        columnspacing=1.2,
    )
    evaluator_legend = fig.legend(
        handles=list(deduped_evaluator_handles.values()),
        loc="upper left",
        bbox_to_anchor=(0.62, 0.06),
        ncol=1,
        borderaxespad=0,
        frameon=True,
        fontsize=fs_legend,
        handletextpad=0.7,
        columnspacing=1.2,
    )
    fig.add_artist(method_legend)
    fig.add_artist(evaluator_legend)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.28)

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


def _best_bitstring_ar(
    row: pd.Series, *, minmax_path, graph_type, num_nodes, minmax_cache, instance_context_cache
) -> pd.Series:
    """Best-observed-bitstring approximation ratio for one hardware job row.

    Every unique bitstring a hardware job actually measured (via its raw
    counts histogram, in ``row["counts"]``) is scored and the single best one
    kept -- the same definition ``best_prefix_metrics`` uses for
    ``BestApproximationRatio``, just applied to a counts dict instead of an
    ordered shot stream (order doesn't matter for a global best).

    ``minmax_cache``/``instance_context_cache`` are mutated in place so
    repeated calls (one per row via ``DataFrame.apply``) reuse the per-instance
    minmax lookup and graph context instead of recomputing them every row.
    """
    _n = row["file_name"][:3]
    if _n not in minmax_cache:
        _mmp = _get_minmax(
            minmax_path, graph_type, _n, num_nodes,
            ER_probability="None", swap_layers="None", degree="None",
        )
        minmax_cache[_n] = _extract_minmax_args(_mmp)
    _min_cut, _max_cut, _sum_weights = minmax_cache[_n]
    _ctx = instance_context_cache[_n]
    _best_ar, _best_bs = -np.inf, None
    for _bs in row["counts"]:
        _e = _maxcut_energy_from_bitstring(_bs, _ctx)
        _ar = _maxcut_approximation_ratio(_min_cut, _max_cut, _sum_weights, _e)
        if _ar > _best_ar:
            _best_ar, _best_bs = _ar, _bs
    return pd.Series({"approximation_ratio": _best_ar, "best_bitstring": _best_bs})


def _build_hw_frontier(
    qpu_time_col: str, hardware_new_df: pd.DataFrame, hardware_df: pd.DataFrame, num_nodes: int
) -> pd.DataFrame:
    """Real-hardware Pareto frontier on a given QPU-time basis.

    Combines ``qpu_time_col`` with each row's ``total_train_cost`` into a
    ``"total duration"`` resource column, then routes through
    :func:`prepare_ibm_qaoa_plot_data`/:func:`build_recommendation_data`
    (the same pipeline the simulated curves use) to get a Pareto frontier on
    a comparable basis.
    """
    _df_sb = hardware_new_df.copy()
    _df_sb["total duration"] = _df_sb[qpu_time_col] + _df_sb["total_train_cost"]
    _df_sb = _df_sb.drop(columns=[
        "QPU_time (s)", "QPU_time_noiseless (s)", "QPU_time_noise_corrected (s)", "total_train_cost",
    ])
    _plot_data = prepare_ibm_qaoa_plot_data(_df_sb, hardware_df, num_nodes)
    _, _frontier = build_recommendation_data(_plot_data["df_points"])
    return _frontier


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
    fs_legend = 15
    y_scale = 100.0
    raw_alpha = 0.18
    raw_ms_map = {0: 8, 1: 8, 2: 8}
    frontier_base_ms_map = {0: 22, 1: 22, 2: 22}
    legend_marker_size = 9.0
    eb_opts = {"mec": "k", "ecolor": "k", "capsize": 3, "elinewidth": 1.2}

    _ensure_save_dir(save_dir)

    df_centroids, _ = build_recommendation_data(df_points)

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

    if not df_frontier.empty:
        plotted_pairs = set(
            df_frontier[["color_label", "job_p"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        scatter_df_for_xlim = df_points[
            df_points[["color_label", "job_p"]]
            .apply(tuple, axis=1)
            .isin(plotted_pairs)
        ]
    else:
        scatter_df_for_xlim = df_points
    scatter_x = pd.to_numeric(scatter_df_for_xlim["total_duration_s"], errors="coerce").dropna()
    if not scatter_x.empty:
        jittered_scatter_right = float(scatter_x.max()) * (10 ** 0.06)
        scatter_right = 10 ** (np.log10(max(jittered_scatter_right, 1e-12)) + 0.025)
        x_right = max(x_right, scatter_right)

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
            color_seg = color_map[row["color_label"]]
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
            mfc=color_map[color_label],
            mec="k" if "*" in _plain_method_label_from_training_method(color_label) else "none",
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

    frontier_annotation_points = []
    for _, row in df_frontier.iterrows():
        p = int(row["job_p"])
        opt_level = _optimization_level(row["color_label"])
        style_kwargs = _style_plot_kwargs(row["color_label"])
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
            ms=frontier_base_ms_map[opt_level],
            mfc=style_kwargs["markerfacecolor"],
            mec=style_kwargs["markeredgecolor"],
            mew=max(float(style_kwargs["markeredgewidth"]), 1.2),
            linestyle="",
            color=style_kwargs["color"],
            zorder=11,
        )
        frontier_annotation_points.append(
            {
                "x": x_val,
                "y": y_val,
                "p": p,
                "marker_size": frontier_base_ms_map[opt_level],
                "color_label": row["color_label"],
            }
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

    def _frontier_line_bboxes(axis, x_min, x_max, pad=5.0):
        bboxes = []
        if df_frontier.empty:
            return bboxes
        for seg_idx in range(len(df_frontier)):
            row = df_frontier.iloc[seg_idx]
            x_start = x_min if seg_idx == 0 else float(row["dur_mean"])
            x_end = (
                float(df_frontier.iloc[seg_idx + 1]["dur_mean"])
                if seg_idx < len(df_frontier) - 1
                else x_max
            )
            y_this = y_scale * float(row["ar_mean"])
            x0, y0 = axis.transData.transform((x_start, y_this))
            x1, y1 = axis.transData.transform((x_end, y_this))
            bboxes.append(
                Bbox.from_extents(
                    min(x0, x1) - pad,
                    min(y0, y1) - pad,
                    max(x0, x1) + pad,
                    max(y0, y1) + pad,
                )
            )
            if seg_idx > 0:
                y_prev = y_scale * float(df_frontier.iloc[seg_idx - 1]["ar_mean"])
                xv0, yv0 = axis.transData.transform((float(row["dur_mean"]), y_prev))
                xv1, yv1 = axis.transData.transform((float(row["dur_mean"]), y_this))
                bboxes.append(
                    Bbox.from_extents(
                        min(xv0, xv1) - pad,
                        min(yv0, yv1) - pad,
                        max(xv0, xv1) + pad,
                        max(yv0, yv1) + pad,
                    )
                )
        return bboxes

    def _annotate_frontier_depths(
        axis,
        points,
        offset_candidates,
        fontsize=11,
        color="0.15",
        marker_pad=7.0,
        avoid_frontier=True,
        force_fallback=False,
    ):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axis_bbox = axis.get_window_extent(renderer=renderer)
        marker_bboxes = []
        for point in points:
            x_disp, y_disp = axis.transData.transform((point["x"], point["y"]))
            half_px = 0.62 * point["marker_size"] * fig.dpi / 72.0 + marker_pad
            marker_bboxes.append(
                Bbox.from_extents(
                    x_disp - half_px,
                    y_disp - half_px,
                    x_disp + half_px,
                    y_disp + half_px,
                )
            )

        x_min, x_max = axis.get_xlim()
        frontier_line_bboxes = _frontier_line_bboxes(axis, x_min, x_max) if avoid_frontier else []
        placed_label_bboxes = []
        for point in points:
            fallback_candidate = None
            point_offsets = list(offset_candidates)
            point_label = _plain_method_label_from_training_method(point.get("color_label", ""))
            prefer_final_right = int(point["p"]) == 7 and point["x"] > 100.0
            prefer_clear_right = (
                int(point["p"]) == 7 and "Fixed Angles" in point_label
            ) or prefer_final_right
            if int(point["p"]) == 7 and "Param" in point_label:
                point_offsets = [(26, 8), (32, 10), (22, -10)] + point_offsets
            if int(point["p"]) == 5 and "Linear Ramp" in point_label:
                point_offsets = [(-20, 12), (-26, 14), (-18, -14)] + point_offsets
            if int(point["p"]) == 9 and "Linear Ramp" in point_label:
                point_offsets = [(0, 22), (14, 22), (-14, 22), (22, 20), (-22, 20)]
            if int(point["p"]) == 5 and "Fixed Angles" in point_label:
                point_offsets = [(18, -14), (24, -16), (18, 12)] + point_offsets
            if prefer_clear_right:
                point_offsets = [(0, 22), (14, 22), (-14, 22), (22, 20), (-22, 20)]
            for dx, dy in point_offsets:
                candidate = axis.annotate(
                    f"p={point['p']}",
                    xy=(point["x"], point["y"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top",
                    fontsize=fontsize,
                    color=color,
                    annotation_clip=True,
                    zorder=12,
                )
                bbox = candidate.get_window_extent(renderer=renderer).expanded(1.04, 1.12)
                overlaps_marker = any(bbox.overlaps(marker_bbox) for marker_bbox in marker_bboxes)
                overlaps_label = any(bbox.overlaps(label_bbox) for label_bbox in placed_label_bboxes)
                overlaps_frontier = False if prefer_clear_right else any(
                    bbox.overlaps(line_bbox) for line_bbox in frontier_line_bboxes
                )
                inside_axes = axis_bbox.contains(*bbox.get_points()[0]) and axis_bbox.contains(
                    *bbox.get_points()[1]
                )
                if inside_axes and not overlaps_marker and not overlaps_label and not overlaps_frontier:
                    if fallback_candidate is not None and fallback_candidate is not candidate:
                        fallback_candidate.remove()
                    placed_label_bboxes.append(bbox)
                    fallback_candidate = None
                    break
                if fallback_candidate is None and inside_axes and not overlaps_marker:
                    fallback_candidate = candidate
                else:
                    candidate.remove()
            if fallback_candidate is not None:
                if force_fallback:
                    bbox = fallback_candidate.get_window_extent(renderer=renderer).expanded(1.04, 1.12)
                    placed_label_bboxes.append(bbox)
                else:
                    fallback_candidate.remove()
            elif force_fallback and offset_candidates:
                dx, dy = offset_candidates[0]
                candidate = axis.annotate(
                    f"p={point['p']}",
                    xy=(point["x"], point["y"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top",
                    fontsize=fontsize,
                    color=color,
                    annotation_clip=True,
                    zorder=12,
                )
                placed_label_bboxes.append(
                    candidate.get_window_extent(renderer=renderer).expanded(1.04, 1.12)
                )

    if frontier_annotation_points:
        inset_cluster = [
            point
            for point in frontier_annotation_points
            if point["x"] <= 18.0 and 81.0 <= point["y"] <= 83.2
        ]
        if len(inset_cluster) > 3:
            inset_cluster = sorted(inset_cluster, key=lambda point: point["x"])
            middle_start = max(0, (len(inset_cluster) - 3) // 2)
            inset_cluster = inset_cluster[middle_start:middle_start + 3]
        elif len(inset_cluster) < 3:
            inset_cluster = []

        main_annotation_points = [
            point for point in frontier_annotation_points if point not in inset_cluster
        ]
        _annotate_frontier_depths(
            ax,
            main_annotation_points,
            [
                (12, 8),
                (12, -10),
                (-12, 8),
                (-12, -10),
                (0, 16),
                (0, -18),
                (18, 0),
                (-18, 0),
                (20, 10),
                (-20, 10),
                (20, -12),
                (-20, -12),
                (28, 0),
                (-28, 0),
            ],
            fontsize=14,
            color="#B00020",
            marker_pad=5.0,
            avoid_frontier=True,
            force_fallback=False,
        )

        if inset_cluster:
            inset_ax = ax.inset_axes([0.67, 0.12, 0.29, 0.30])
            inset_ax.set_facecolor((1, 1, 1, 0.94))
            inset_ax.set_xscale("log")
            x_cluster = np.array([point["x"] for point in inset_cluster], dtype=float)
            y_cluster = np.array([point["y"] for point in inset_cluster], dtype=float)
            x_pad_decades = 0.11
            inset_x0 = 10 ** (np.log10(x_cluster.min()) - x_pad_decades)
            inset_x1 = 10 ** (np.log10(x_cluster.max()) + x_pad_decades)
            y_cluster_span = max(float(y_cluster.max() - y_cluster.min()), 0.04)
            y_cluster_pad = max(0.10, 0.65 * y_cluster_span)
            inset_y0 = max(y_lo, float(y_cluster.min()) - y_cluster_pad)
            inset_y1 = min(y_hi, float(y_cluster.max()) + y_cluster_pad)
            inset_ax.set_xlim(inset_x0, inset_x1)
            inset_ax.set_ylim(inset_y0, inset_y1)

            for seg_idx in range(len(df_frontier)):
                row = df_frontier.iloc[seg_idx]
                color_seg = color_map[row["color_label"]]
                x_start = x_left if seg_idx == 0 else float(row["dur_mean"])
                x_end = (
                    float(df_frontier.iloc[seg_idx + 1]["dur_mean"])
                    if seg_idx < len(df_frontier) - 1
                    else x_right
                )
                y_this = y_scale * float(row["ar_mean"])
                inset_ax.plot(
                    [x_start, x_end],
                    [y_this, y_this],
                    color=color_seg,
                    linewidth=2.0,
                    linestyle="--",
                    alpha=0.85,
                    zorder=9,
                )
                if seg_idx > 0:
                    y_prev = y_scale * float(df_frontier.iloc[seg_idx - 1]["ar_mean"])
                    inset_ax.plot(
                        [float(row["dur_mean"]), float(row["dur_mean"])],
                        [y_prev, y_this],
                        color=color_seg,
                        linewidth=2.0,
                        linestyle="--",
                        alpha=0.85,
                        zorder=9,
                    )

            for point in inset_cluster:
                opt_level = _optimization_level(point["color_label"])
                style_kwargs = _style_plot_kwargs(point["color_label"])
                inset_ax.plot(
                    point["x"],
                    point["y"],
                    marker=shape_map[point["color_label"]],
                    ms=frontier_base_ms_map[opt_level],
                    mfc=style_kwargs["markerfacecolor"],
                    mec=style_kwargs["markeredgecolor"],
                    mew=max(float(style_kwargs["markeredgewidth"]), 1.0),
                    linestyle="",
                    color=style_kwargs["color"],
                    zorder=11,
                )

            _annotate_frontier_depths(
                inset_ax,
                inset_cluster,
                [
                    (12, 8),
                    (12, -10),
                    (-12, 8),
                    (-12, -10),
                    (0, 16),
                    (0, -18),
                    (18, 0),
                    (-18, 0),
                    (22, 10),
                    (-22, 10),
                    (22, -12),
                    (-22, -12),
                ],
                fontsize=14,
                color="#B00020",
                marker_pad=5.0,
                avoid_frontier=True,
                force_fallback=False,
            )
            inset_ax.xaxis.set_major_locator(
                LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6)
            )
            inset_ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=False))
            inset_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            inset_ax.tick_params(axis="both", which="major", labelsize=fs_tick, length=3)
            inset_ax.grid(True, alpha=0.35)
            inset_ax.minorticks_off()

    legend_pairs = (
        df_frontier[["color_label"]]
        .drop_duplicates()
        .sort_values(["color_label"])
    )
    method_legend_dict = {}
    evaluator_legend_dict = {}
    for _, pair in legend_pairs.iterrows():
        color_label = pair["color_label"]
        style_kwargs = _style_plot_kwargs(color_label)
        method_label = _method_label_from_training_method(color_label, format="latex")
        if method_label not in method_legend_dict:
            method_legend_dict[method_label] = Line2D(
                [0],
                [0],
                marker="s",
                color=color_map[color_label],
                markerfacecolor=style_kwargs["markerfacecolor"],
                markeredgecolor=style_kwargs["markeredgecolor"],
                markeredgewidth=max(float(style_kwargs["markeredgewidth"]), 1.0),
                markersize=legend_marker_size,
                linestyle="",
                label=method_label,
            )
        evaluator_label = _evaluation_label_from_training_method(color_label)
        if evaluator_label and evaluator_label not in evaluator_legend_dict:
            evaluator_legend_dict[evaluator_label] = Line2D(
                [0],
                [0],
                marker=_marker_from_training_method(color_label),
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                lw=0,
                markersize=legend_marker_size,
                label=evaluator_label,
            )

    frontier_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="Pareto Frontier",
    )
    method_items = sorted(method_legend_dict.items(), key=lambda item: item[0])
    evaluator_items = sorted(evaluator_legend_dict.items(), key=lambda item: item[0])

    legend_handles = [handle for _, handle in method_items]
    legend_handles += [handle for _, handle in evaluator_items]
    legend_handles.append(frontier_handle)
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.02),
        ncol=min(4, max(1, len(legend_handles))),
        borderaxespad=0.0,
        frameon=True,
        fontsize=fs_legend,
        handlelength=2.0,
        handletextpad=0.7,
        columnspacing=1.0,
    )

    fig.tight_layout()
    fig.subplots_adjust(left=0.08, bottom=0.11)

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
    display_dir = os.path.relpath(plot_dir, Path.cwd()) if plot_dir.is_absolute() else plot_dir
    print(f"Saved: {Path(display_dir) / name}.pdf and .png")


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


def _percent_approx_ylabel(ylabel: str) -> str:
    """Convert an approximation-ratio label to a percentage label."""
    label = str(ylabel)
    label = label.replace("Approximation ratio", "Approximation ratio (%)")
    label = label.replace("approximation ratio", "approximation ratio (%)")
    if "(%)" not in label:
        label = f"{label} (%)"
    return label


def _percent_axis_values(
    ylim: tuple[float, float] | None,
    yticks: list[float] | None,
) -> tuple[tuple[float, float] | None, list[float] | None]:
    """Convert optional ratio y-axis limits/ticks to percentage units."""
    percent_ylim = None if ylim is None else (ylim[0] * 100.0, ylim[1] * 100.0)
    percent_yticks = None if yticks is None else [tick * 100.0 for tick in yticks]
    return percent_ylim, percent_yticks


def _ws_display_method_label(label: str) -> str:
    """Convert an internal method label to a compact Window Sticker legend entry.

    The incoming label may carry LaTeX optimization markers (``$^\\star$``,
    ``$^\\dagger$``) produced by :func:`_method_label_from_training_method`.
    This function converts them to their plain Unicode equivalents and preserves
    the depth suffix ``(p = N)`` when present.

    Parameters
    ----------
    label : str
        Method label as stored in the ``method_label`` column, e.g.
        ``"Fixed Angles$^\\star$ (p=5)"``.

    Returns
    -------
    str
        Display label with the correct optimization marker:

        - ``*``  — full angle reoptimization via a Scipy trainer (e.g. COBYLA).
        - ``†``  — no angle optimization.
        - *(none)* — method-parameter optimization only (e.g. Linear Ramp's
          ramp-parameter sweep, or Parameter Transfer's classical preprocessing).
    """
    label_str = str(label)
    depth_match = re.search(r"\s*(\(p\s*=\s*\d+\))\s*$", label_str)
    depth_suffix = f" {depth_match.group(1)}" if depth_match else ""
    cleaned = re.sub(r"\s*\(p\s*=\s*\d+\)\s*$", "", label_str)

    has_star = bool(re.search(r"\$\^\\star\$|[\*★⋆]", cleaned))
    has_dagger = bool(re.search(r"\$\^\\dagger\$|[†]", cleaned))

    base_label = re.sub(r"\$?\s*\^?\s*\{?\s*\\(?:star|dagger)\s*\}?\s*\$?", "", cleaned)
    base_label = re.sub(r"[\*†★⋆]", "", base_label).strip()
    base_label = re.sub(r"\s+", " ", base_label)

    opt_marker = "†" if has_dagger else ("*" if has_star else "")
    return f"{base_label}{opt_marker}{depth_suffix}"


def _is_no_opt_metadata(value: object) -> bool:
    """Return whether a method/config/tag explicitly denotes no optimization."""
    text = str(value).lower()
    compact = re.sub(r"[^a-z0-9]+", "_", text)
    return "no_opt" in compact or "no_optimization" in compact or "noopt" in compact


def _force_dagger_label(label: str) -> str:
    """Replace optimization markers in a display label with a dagger marker."""
    text = str(label)
    suffix = ""
    suffix_match = re.search(r"\s+\([^)]*\)$", text)
    if suffix_match:
        suffix = suffix_match.group(0)
        text = text[: suffix_match.start()]

    text = re.sub(r"\$\^\\(?:star|dagger)\$", "", text)
    text = re.sub(r"\^\\(?:star|dagger)", "", text)
    text = text.replace(r"$^\star$", "").replace(r"$^\dagger$", "")
    text = text.replace("*", "").replace("†", "").strip()
    return f"{text}†{suffix}"


def curve_label(df: pd.DataFrame, default: str) -> str:
    """Build a compact method label from strategy, simulator, and depth columns."""
    if df.empty:
        if _is_no_opt_metadata(default):
            return _force_dagger_label(default)
        return default
    metadata_values = [default]
    for col in [
        "strategy",
        "method_label",
        "result_tag",
        "result_root",
        "root",
        "trainer_config",
        "training_method",
        "method_name",
        "config",
        "config_path",
    ]:
        if col in df and df[col].notna().any():
            metadata_values.extend(df[col].dropna().astype(str).head(5).tolist())
    strategy = (
        df["strategy"].dropna().astype(str).iloc[0]
        if "strategy" in df and df["strategy"].notna().any()
        else default
    )
    p_val = float(df["p"].dropna().iloc[0]) if "p" in df and df["p"].notna().any() else None
    try:
        label = _method_label_from_training_method(strategy, format="latex")
    except Exception:
        label = strategy
    parts = []
    if p_val is not None:
        parts.append(f"p={p_val:g}")
    label = f"{label} ({', '.join(parts)})" if parts else label
    if any(_is_no_opt_metadata(value) for value in metadata_values):
        label = _force_dagger_label(label)
    return label


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


def read_first_summary_csv(root: str | Path, names: list[str]) -> pd.DataFrame:
    """Read the first present summary CSV from a list of compatible filenames."""
    for name in names:
        df = read_summary_csv(root, name)
        if not df.empty:
            return df
    return pd.DataFrame()


def rebuild_strategy_budget_summary(root: str | Path) -> pd.DataFrame:
    """Rebuild the training budget summary from the saved raw frontier when needed."""
    root = Path(root)
    frontier_path = root / "pss_exp_raw_frontier.pkl"
    frontier_csv_path = root / "pss_exp_raw_frontier.csv"
    if not frontier_path.exists() and not frontier_csv_path.exists():
        return pd.DataFrame()

    try:
        from .simulation_validation import build_strategy_budget_summary
    except Exception:
        return pd.DataFrame()

    try:
        frontier_df = pd.read_pickle(frontier_path)
    except Exception:
        if not frontier_csv_path.exists():
            return pd.DataFrame()
        frontier_df = pd.read_csv(frontier_csv_path)
    strategy_budget = build_strategy_budget_summary(frontier_df, split="train")
    if not strategy_budget.empty:
        strategy_budget.to_csv(root / "strategy_budget_summary_train.csv", index=False)
    return strategy_budget


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
        if strategy_budget.empty:
            strategy_budget = rebuild_strategy_budget_summary(root)
        actionable_lookup = read_first_summary_csv(
            root,
            ["actionable_pss_lookup_train.csv", "actionable_recipe_train.csv"],
        )
        actionable_fit = read_first_summary_csv(
            root,
            ["actionable_pss_fit_train.csv", "fitted_actionable_recipe_train.csv"],
        )
        window_sticker = read_summary_csv(root, "window_sticker_summary.csv")
        projection = read_summary_csv(root, "projection_summary.csv")
        fitted_projection_test = read_summary_csv(root, "fitted_actionable_projection_test.csv")
        fitted_projection_train = read_summary_csv(root, "fitted_actionable_projection_train.csv")
        virtual_best = read_summary_csv(root, "virtual_best_summary.csv")
        virtual_best_train = read_summary_csv(root, "virtual_best_lookup_train.csv")

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
        result_tag = Path(tag).name
        label_source_for_label = label_source.copy()
        label_source_for_label["result_tag"] = result_tag
        label_source_for_label["result_root"] = str(root)
        label = curve_label(label_source_for_label, result_tag)

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
                "fitted_projection_test": attach_result_metadata(fitted_projection_test, result_tag, root, label),
                "fitted_projection_train": attach_result_metadata(fitted_projection_train, result_tag, root, label),
                "virtual_best": attach_result_metadata(virtual_best, result_tag, root, label),
                "virtual_best_train": attach_result_metadata(virtual_best_train, result_tag, root, label),
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
    keep_cols = ["method_label", "T", "response_mean"]
    optional_cols = [col for col in ["response_std", "n_instances"] if col in df.columns]
    curve = (
        df.loc[:, keep_cols + optional_cols]
        .dropna(subset=["T", "response_mean"])
        .sort_values(["method_label", "T"])
        .groupby(["method_label", "T"], as_index=False)
        .mean(numeric_only=True)
        .rename(columns={"response_mean": "response"})
    )
    curve["response_monotone"] = curve.groupby("method_label")["response"].cummax()
    if {"response_std", "n_instances"}.issubset(curve.columns):
        counts = pd.to_numeric(curve["n_instances"], errors="coerce").clip(lower=1)
        sem = pd.to_numeric(curve["response_std"], errors="coerce").fillna(0.0) / np.sqrt(counts)
        curve["response_lower"] = curve["response"] - 1.96 * sem
        curve["response_upper"] = curve["response"] + 1.96 * sem
        curve["response_lower_monotone"] = curve.groupby("method_label")["response_lower"].cummax()
        curve["response_upper_monotone"] = curve.groupby("method_label")["response_upper"].cummax()
    return curve


def curve_from_window_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a monotone curve from Window Sticker projection rows."""
    required = {"method_label", "resource", "response"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["method_label", "resource", "response"])
    keep_cols = ["method_label", "resource", "response"]
    optional_cols = [col for col in ["response_lower", "response_upper"] if col in df.columns]
    curve = (
        df.loc[:, keep_cols + optional_cols]
        .dropna(subset=["resource", "response"])
        .sort_values(["method_label", "resource"])
        .groupby(["method_label", "resource"], as_index=False)
        .mean(numeric_only=True)
    )
    curve["response_monotone"] = curve.groupby("method_label")["response"].cummax()
    for col in ["response_lower", "response_upper"]:
        if col in curve.columns:
            curve[f"{col}_monotone"] = curve.groupby("method_label")[col].cummax()
    return curve


def curve_from_response_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a monotone response curve from a summary table with resource/response columns."""
    return curve_from_window_summary(df)


def _rescale_resource(df: pd.DataFrame | None, scale_by_label: dict[str, float]) -> pd.DataFrame:
    """Multiply a ``resource`` column by a per-``method_label`` scale factor.

    Used to recalibrate a curve's resource axis onto a different shot-rate
    basis per method (e.g. simulated-vs-hardware time-per-shot). A row whose
    ``method_label`` has no entry in ``scale_by_label`` is left unscaled.
    """
    if df is None or df.empty or "resource" not in df.columns:
        return (df if df is not None else pd.DataFrame()).copy()
    df = df.copy()
    if "method_label" in df.columns:
        for lbl, factor in scale_by_label.items():
            mask = df["method_label"] == lbl
            df.loc[mask, "resource"] = df.loc[mask, "resource"] * factor
    return df


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


def _display_cross_strategy_envelope(
    curve_df: pd.DataFrame,
    resource_col: str,
    response_col: str = "response_monotone",
    *,
    num_points: int = 1000,
) -> pd.DataFrame:
    """Build an upper envelope of the linearly displayed method curves."""
    if curve_df.empty or resource_col not in curve_df or response_col not in curve_df:
        return pd.DataFrame(columns=[resource_col, response_col])

    valid = curve_df.copy()
    valid[resource_col] = pd.to_numeric(valid[resource_col], errors="coerce")
    valid[response_col] = pd.to_numeric(valid[response_col], errors="coerce")
    valid = valid.dropna(subset=[resource_col, response_col])
    valid = valid[valid[resource_col] > 0]
    if valid.empty:
        return pd.DataFrame(columns=[resource_col, response_col])

    x_min = float(valid[resource_col].min())
    x_max = float(valid[resource_col].max())
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min <= 0 or x_max < x_min:
        return pd.DataFrame(columns=[resource_col, response_col])

    if np.isclose(x_min, x_max):
        grid = np.array([x_min])
    else:
        grid = np.geomspace(x_min, x_max, num_points)

    interpolated = []
    log_grid = np.log10(grid)
    for _, group in valid.groupby("method_label"):
        group = (
            group.sort_values(resource_col)
            .drop_duplicates(subset=[resource_col], keep="first")
        )
        if group.empty:
            continue
        x = group[resource_col].to_numpy(dtype=float)
        y = group[response_col].to_numpy(dtype=float)
        if len(x) == 1:
            interp_y = np.full_like(grid, np.nan, dtype=float)
            interp_y[grid >= x[0]] = y[0]
        else:
            interp_y = np.interp(log_grid, np.log10(x), y, left=np.nan, right=y[-1])
            interp_y[grid < x[0]] = np.nan
        interpolated.append(interp_y)

    if not interpolated:
        return pd.DataFrame(columns=[resource_col, response_col])

    values = np.vstack(interpolated)
    finite_cols = np.isfinite(values).any(axis=0)
    if not finite_cols.any():
        return pd.DataFrame(columns=[resource_col, response_col])

    envelope = np.full(grid.shape, np.nan, dtype=float)
    envelope[finite_cols] = np.nanmax(values[:, finite_cols], axis=0)
    envelope[finite_cols] = np.maximum.accumulate(envelope[finite_cols])
    return pd.DataFrame({resource_col: grid[finite_cols], response_col: envelope[finite_cols]})


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
    envelope_label: str = "Virtual best",
) -> None:
    """Plot per-method Window Sticker curves and their cross-method virtual best."""
    if curve_df.empty:
        print(f"Skipping {title}: no curve data found.")
        return
    percent_ylim, percent_yticks = _percent_axis_values(approx_ylim, approx_yticks)
    plt.figure(figsize=(8.5, 5))
    for label, group in curve_df.groupby("method_label"):
        group = group.sort_values(resource_col)
        color = window_sticker_method_color(label)
        lower_col = "response_lower_monotone" if "response_lower_monotone" in group.columns else "response_lower"
        upper_col = "response_upper_monotone" if "response_upper_monotone" in group.columns else "response_upper"
        if lower_col in group.columns and upper_col in group.columns:
            band = group.copy()
            band[resource_col] = pd.to_numeric(band[resource_col], errors="coerce")
            band[lower_col] = pd.to_numeric(band[lower_col], errors="coerce")
            band[upper_col] = pd.to_numeric(band[upper_col], errors="coerce")
            band = band.dropna(subset=[resource_col, lower_col, upper_col])
            if not band.empty:
                plt.fill_between(
                    band[resource_col].to_numpy(dtype=float),
                    band[lower_col].to_numpy(dtype=float) * 100.0,
                    band[upper_col].to_numpy(dtype=float) * 100.0,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
        plt.plot(
            group[resource_col],
            pd.to_numeric(group["response_monotone"], errors="coerce") * 100.0,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=label,
        )
    if not envelope_df.empty:
        display_envelope_df = _display_cross_strategy_envelope(
            curve_df,
            resource_col,
        )
        if display_envelope_df.empty:
            display_envelope_df = envelope_df
        plt.plot(
            display_envelope_df[resource_col],
            pd.to_numeric(display_envelope_df["response_monotone"], errors="coerce") * 100.0,
            color="black",
            linewidth=2.8,
            linestyle="--",
            label=envelope_label,
        )
    plt.xscale("log")
    plt.xlabel(
        r"Resource ($T_{\mathrm{proxy}} = t_{\mathrm{preprocessing}} + t_{\mathrm{train}} + Qt_{\mathrm{shot}}$) [s]",
        fontsize=WINDOW_STICKER_LABEL_FONTSIZE,
    )
    plt.ylabel(_percent_approx_ylabel(ylabel), fontsize=WINDOW_STICKER_LABEL_FONTSIZE)
    plt.tick_params(axis="both", labelsize=WINDOW_STICKER_TICK_FONTSIZE)
    apply_shared_approx_axis(ylim=percent_ylim, yticks=percent_yticks)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if title:
        plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="best", frameon=True, fontsize=WINDOW_STICKER_LEGEND_FONTSIZE)
    save_current_plot(filename, plot_dir)
    plt.show()


def plot_multi_method_window_sticker_components(
    *,
    virtual_best_df: pd.DataFrame,
    averaged_prescription_df: pd.DataFrame | None = None,
    fitted_prescription_df: pd.DataFrame,
    plot_dir: str | Path,
    filename: str,
    ylabel: str,
    title: str | None = None,
    approx_ylim: tuple[float, float] | None = None,
    approx_yticks: list[float] | None = None,
    show_ci: bool = True,
) -> None:
    """Overlay each method's virtual best and actionable fitted prescription curves."""
    # Kept in the signature for older notebook calls; intentionally not plotted here.
    _ = averaged_prescription_df

    def _local_method_color(label: str, color_fn=window_sticker_method_color) -> str:
        return color_fn(label)

    curves = {
        "Actionable fit prescription": curve_from_response_summary(fitted_prescription_df),
        "Virtual best": curve_from_response_summary(virtual_best_df),
    }
    if all(curve.empty for curve in curves.values()):
        print(f"Skipping {filename}: no multi-method Window Sticker curves found.")
        return
    percent_ylim, percent_yticks = _percent_axis_values(approx_ylim, approx_yticks)

    labels = sorted(
        {
            label
            for curve in curves.values()
            if not curve.empty
            for label in curve["method_label"].dropna().astype(str).unique()
        }
    )
    color_map = window_sticker_method_color_map(labels)
    style_map = {
        "Virtual best": {
            "linestyle": "-",
            "marker": "o",
            "linewidth": 3.0,
            "markersize": 4.6,
            "alpha": 1.0,
            "zorder": 6,
        },
        "Actionable fit prescription": {
            "linestyle": "-.",
            "marker": "s",
            "linewidth": 1.8,
            "markersize": 3.6,
            "alpha": 0.8,
            "zorder": 3,
        },
    }

    fig, ax = plt.subplots(figsize=(8.5, 5))
    plotted_x: list[float] = []
    plotted_y: list[float] = []
    for curve_name, curve in curves.items():
        if curve.empty:
            continue
        style = style_map[curve_name]
        for label, group in curve.groupby("method_label"):
            group = group.loc[:, ~group.columns.duplicated()].copy()
            group["resource"] = pd.to_numeric(group["resource"], errors="coerce")
            group["response_monotone"] = pd.to_numeric(group["response_monotone"], errors="coerce")
            group = group.dropna(subset=["resource", "response_monotone"])
            group = group[group["resource"] > 0].sort_values("resource")
            if group.empty:
                continue
            lower_col = "response_lower_monotone" if "response_lower_monotone" in group.columns else "response_lower"
            upper_col = "response_upper_monotone" if "response_upper_monotone" in group.columns else "response_upper"
            if show_ci and lower_col in group.columns and upper_col in group.columns:
                lower = pd.to_numeric(group[lower_col], errors="coerce")
                upper = pd.to_numeric(group[upper_col], errors="coerce")
                band = group.assign(_lower=lower, _upper=upper).dropna(
                    subset=["resource", "_lower", "_upper"]
                )
                if not band.empty:
                    band_lower = band["_lower"].to_numpy(dtype=float) * 100.0
                    band_upper = band["_upper"].to_numpy(dtype=float) * 100.0
                    ax.fill_between(
                        band["resource"].to_numpy(dtype=float),
                        band_lower,
                        band_upper,
                        color=color_map[str(label)],
                        alpha=0.14 if curve_name == "Virtual best" else 0.08,
                        linewidth=0,
                        zorder=max(1, style["zorder"] - 2),
                    )
                    plotted_y.extend(band_lower)
                    plotted_y.extend(band_upper)
            response_percent = group["response_monotone"].to_numpy(dtype=float) * 100.0
            ax.plot(
                group["resource"],
                response_percent,
                color=color_map[str(label)],
                label=None,
                **style,
            )
            plotted_x.extend(group["resource"].to_numpy(dtype=float))
            plotted_y.extend(response_percent)

    ax.set_xscale("log")
    ax.set_xlabel(
        r"Resource ($T_{\mathrm{proxy}} = t_{\mathrm{preprocessing}} + t_{\mathrm{train}} + Qt_{\mathrm{shot}}$) [s]",
        fontsize=WINDOW_STICKER_LABEL_FONTSIZE,
    )
    display_ylabel = _percent_approx_ylabel(ylabel).replace(" on ", "\non ", 1)
    ax.set_ylabel(display_ylabel, fontsize=WINDOW_STICKER_LABEL_FONTSIZE, labelpad=10)
    ax.tick_params(axis="both", labelsize=WINDOW_STICKER_TICK_FONTSIZE)
    finite_x = np.asarray([x for x in plotted_x if np.isfinite(x) and x > 0], dtype=float)
    finite_y = np.asarray([y for y in plotted_y if np.isfinite(y)], dtype=float)
    if finite_x.size:
        ax.set_xlim(float(finite_x.min()), float(finite_x.max()))
    if percent_ylim is not None:
        apply_shared_approx_axis(ax, ylim=percent_ylim, yticks=percent_yticks)
    elif finite_y.size:
        y_min = float(finite_y.min())
        y_max = float(finite_y.max())
        y_span = y_max - y_min
        y_pad = max(0.3, 0.06 * y_span) if y_span > 0 else 0.3
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    _ = title
    ax.grid(alpha=0.25)

    method_handles = [
        Line2D([0], [0], color=color_map[label], linewidth=2.6, label=_ws_display_method_label(label))
        for label in labels
    ]
    curve_handles = [
        Line2D([0], [0], color="black", label=name, **style)
        for name, style in style_map.items()
    ]
    ax.legend(
        handles=[*curve_handles, *method_handles],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        frameon=True,
        ncol=1,
        fontsize=max(10, WINDOW_STICKER_LEGEND_FONTSIZE - 2),
        handlelength=2.4,
        handletextpad=0.6,
        labelspacing=0.35,
    )
    fig.tight_layout()
    save_current_plot(filename, plot_dir)
    plt.show()


# FA_star/FA_dagger shade ranges were originally (0.38, 0.92) / (0.32, 0.58),
# which overlap on 0.38-0.58 -- at matched depths (e.g. both at p=6) the two
# curves could land on nearly the same Blues shade. Narrowed to disjoint
# bands (both still plt.cm.Blues, matching the paper's single Fixed-Angles
# blue) so every depth of one variant is visually distinct from every depth
# of the other; PT/Interp are unchanged from the original.
#
# Linear Ramp has three optimization tiers (see _optimization_level /
# _QPS_METHOD_LABELS): LR_dagger = no optimization at all (fixed ramp
# slopes), LR = ramp-parameter optimization only (LR_PP_opt), LR_star =
# ramp-parameter + full angle optimization (LR_PP_angle_opt). Same YlOrBr
# colormap as before, split into three disjoint bands -- more optimization
# reads as a darker/more saturated shade, matching FA's convention.
_FAMILY_CMAP_SPEC: dict[str, tuple] = {
    "FA_star":   (plt.cm.Blues,   0.55, 0.92),
    "FA_dagger": (plt.cm.Blues,   0.15, 0.42),
    "PT":        (plt.cm.Greys,   0.35, 0.60),
    "LR_star":   (plt.cm.YlOrBr,  0.65, 0.92),
    "LR":        (plt.cm.YlOrBr,  0.42, 0.58),
    "LR_dagger": (plt.cm.YlOrBr,  0.15, 0.32),
    "Interp":    (plt.cm.Greens,  0.35, 0.85),
}
_FAMILY_DISPLAY: dict[str, str] = {
    "FA_star":   r"Fixed Angles$^*$",
    "FA_dagger": r"Fixed Angles$^\dagger$",
    "PT":        "Param. Transfer",
    "LR_star":   r"Linear Ramp$^*$",
    "LR":        "Linear Ramp",
    "LR_dagger": r"Linear Ramp$^\dagger$",
    "Interp":    "Interpolation",
}
_FAMILY_ORDER = ["FA_star", "FA_dagger", "PT", "LR_star", "LR", "LR_dagger", "Interp"]


def _detect_method_family(label: str) -> str:
    s = str(label).lower()
    if "fixed angles" in s and re.search(r"[†]|\$\^\\dagger\$|dagger", s):
        return "FA_dagger"
    if "fixed angles" in s:
        return "FA_star"
    if re.search(r"(?<![a-z])pt(?![a-z])|param|transfer", s):
        return "PT"
    if "linear" in s or "ramp" in s:
        if re.search(r"[†]|\$\^\\dagger\$|dagger", s):
            return "LR_dagger"
        if re.search(r"\$\^\\star\$|[\*★⋆]", s):
            return "LR_star"
        return "LR"
    if "interp" in s or "i_mps" in s or re.search(r"(?<![a-z])i_", s):
        return "Interp"
    return s[:20]


def _label_depth(label: str) -> int | None:
    m = re.search(r"\(p\s*=\s*(\d+)\)", str(label))
    return int(m.group(1)) if m else None


def _build_family_color_map(
    labels: Iterable[str],
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, list[int]]]:
    """Depth-gradient colour map: within each method family lighter = lower p, darker = higher p."""
    family_labels: dict[str, list[str]] = {}
    for lbl in labels:
        family_labels.setdefault(_detect_method_family(lbl), []).append(lbl)

    family_p_vals: dict[str, list[int]] = {}
    for fam, fam_lbls in family_labels.items():
        ps = sorted({_label_depth(l) for l in fam_lbls if _label_depth(l) is not None})
        family_p_vals[fam] = ps

    color_map: dict[str, Any] = {}
    for fam, fam_lbls in family_labels.items():
        cmap_fn, lo, hi = _FAMILY_CMAP_SPEC.get(fam, (plt.cm.viridis, 0.3, 0.9))
        p_vals = family_p_vals[fam]
        for lbl in fam_lbls:
            p = _label_depth(lbl)
            if p is not None and len(p_vals) > 1:
                t = p_vals.index(p) / (len(p_vals) - 1)
                color_map[lbl] = cmap_fn(lo + t * (hi - lo))
            elif p is not None:
                color_map[lbl] = cmap_fn((lo + hi) / 2)
            else:
                color_map[lbl] = window_sticker_method_color(lbl)
    return color_map, family_labels, family_p_vals


def _pareto_envelope_and_owner(
    entries: list[tuple[str, Any, np.ndarray, np.ndarray]],
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Running-max Pareto envelope across method entries, and which entry holds it.

    ``entries`` is a list of (label, color, resource_array, response_array)
    tuples with natural (non-extrapolated) data, NaN before a method's data
    starts and after it ends. The envelope is a running maximum over resource
    so a method's record, once set, persists until a higher value from any
    method supersedes it, matching the fact that per-method curves are
    already forced non-decreasing via cummax().
    """
    n_grid = len(grid)
    matrix = np.full((len(entries), n_grid), np.nan)
    for i, entry in enumerate(entries):
        # Index-based rather than a strict 4-tuple unpack, so callers that
        # carry extra per-entry fields (e.g. CI bounds) past position 3 can
        # pass their entries straight through without truncating first.
        xs, ys = entry[2], entry[3]
        if len(xs) < 2:
            continue
        in_range = (grid >= xs[0]) & (grid <= xs[-1])
        matrix[i, in_range] = np.interp(grid[in_range], xs, ys)

    envelope = np.full(n_grid, np.nan)
    best_idx = np.full(n_grid, -1, dtype=int)
    best_value_so_far = -np.inf
    best_idx_so_far = -1
    with np.errstate(all="ignore"):
        col_argmax = np.where(
            np.all(np.isnan(matrix), axis=0), -1,
            np.nanargmax(np.nan_to_num(matrix, nan=-np.inf), axis=0),
        )
    for col in range(n_grid):
        i = col_argmax[col]
        if i >= 0 and matrix[i, col] > best_value_so_far:
            best_value_so_far = matrix[i, col]
            best_idx_so_far = i
        if best_idx_so_far >= 0:
            envelope[col] = best_value_so_far
            best_idx[col] = best_idx_so_far
    return envelope, best_idx


def _pareto_envelope_bounds(
    entries_with_bounds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    grid: np.ndarray,
    best_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper CI envelope consistent with an already-computed best_idx.

    ``entries_with_bounds`` is a list of (resource_array, lower_array,
    upper_array) tuples, index-aligned with the ``entries`` list that
    produced ``best_idx`` via :func:`_pareto_envelope_and_owner`. Reuses
    ``best_idx`` directly (rather than re-deriving ownership) so the CI band
    always matches the entry that actually owns the envelope at each column,
    including the flat holds where that entry's own value has plateaued.
    """
    n_grid = len(grid)
    lower_matrix = np.full((len(entries_with_bounds), n_grid), np.nan)
    upper_matrix = np.full((len(entries_with_bounds), n_grid), np.nan)
    for i, (xs, ys_lower, ys_upper) in enumerate(entries_with_bounds):
        if len(xs) < 2:
            continue
        in_range = (grid >= xs[0]) & (grid <= xs[-1])
        lower_matrix[i, in_range] = np.interp(grid[in_range], xs, ys_lower)
        upper_matrix[i, in_range] = np.interp(grid[in_range], xs, ys_upper)

    lower_env = np.full(n_grid, np.nan)
    upper_env = np.full(n_grid, np.nan)
    lower_so_far = np.nan
    upper_so_far = np.nan
    prev_idx = -1
    for col in range(n_grid):
        idx = int(best_idx[col])
        if idx < 0:
            continue
        if idx != prev_idx:
            lower_so_far = lower_matrix[idx, col]
            upper_so_far = upper_matrix[idx, col]
            prev_idx = idx
        lower_env[col] = lower_so_far
        upper_env[col] = upper_so_far
    return lower_env, upper_env


def _sim_entries(df_test: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per-method (label, resource, response%, ci_lower, ci_upper) curves for a Pareto overlay.

    Drops methods with fewer than two resource points (they can't form a
    curve). ``response_lower``/``response_upper`` (from
    :func:`curve_from_response_summary`, see
    ``_build_response_summary_from_rec_params``) are a 95% CI
    (response +/- 1.96*SEM); the CI half-width is rescaled down to 1 SEM so
    these whiskers are on the same statistical basis as a real-hardware
    curve's ``dur_sem``/``ar_sem``.
    """
    curve = curve_from_response_summary(df_test)
    if curve.empty:
        return []
    lower_col = "response_lower_monotone" if "response_lower_monotone" in curve.columns else "response_lower"
    upper_col = "response_upper_monotone" if "response_upper_monotone" in curve.columns else "response_upper"
    has_ci = lower_col in curve.columns and upper_col in curve.columns
    entries = []
    for label, group in curve.groupby("method_label"):
        group = group.loc[:, ~group.columns.duplicated()].copy()
        group["resource"] = pd.to_numeric(group["resource"], errors="coerce")
        group["response_monotone"] = pd.to_numeric(group["response_monotone"], errors="coerce")
        if has_ci:
            group[lower_col] = pd.to_numeric(group[lower_col], errors="coerce")
            group[upper_col] = pd.to_numeric(group[upper_col], errors="coerce")
        group = group.dropna(subset=["resource", "response_monotone"])
        group = group[group["resource"] > 0].sort_values("resource")
        if len(group) < 2:
            continue
        response_percent = group["response_monotone"].to_numpy(dtype=float) * 100.0
        if has_ci:
            _ci_lower_raw = group[lower_col].to_numpy(dtype=float) * 100.0
            _ci_upper_raw = group[upper_col].to_numpy(dtype=float) * 100.0
            _sem_half_width = (_ci_upper_raw - _ci_lower_raw) / 2.0 / 1.96
            ci_lower = response_percent - _sem_half_width
            ci_upper = response_percent + _sem_half_width
        else:
            ci_lower = np.full_like(response_percent, np.nan)
            ci_upper = np.full_like(response_percent, np.nan)
        entries.append((
            str(label),
            group["resource"].to_numpy(dtype=float),
            response_percent,
            ci_lower,
            ci_upper,
        ))
    return entries


def _label_hw_frontier(frontier: pd.DataFrame) -> pd.DataFrame:
    """Tag a hardware-frontier table with the shared "<family> (p=<depth>)" method_label.

    Uses the same label format the simulated curves use (via
    :func:`_method_label_from_training_method`) so both datasets can share
    one family colour map instead of an arbitrary flat colour.
    """
    _hw = frontier.copy()
    _hw["method_label"] = [
        f"{_method_label_from_training_method(color_label, format='latex')} (p={int(job_p)})"
        for color_label, job_p in zip(_hw["color_label"], _hw["job_p"])
    ]
    return _hw


def _sim_winners(
    entries: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    color_map: dict[str, object],
) -> set[str]:
    """Which method labels actually own some stretch of the simulated Pareto envelope.

    Used to restrict the plotted colorbars/legend to methods that are
    actually visible on the frontier, rather than every method with data.
    """
    colored = [(lbl, color_map[lbl], xs, ys) for lbl, xs, ys, _, _ in entries if len(xs) >= 2]
    if not colored:
        return set()
    x_lo = min(e[2][0] for e in colored)
    x_hi = max(e[2][-1] for e in colored)
    if x_lo <= 0 or x_hi <= x_lo:
        return set()
    grid = np.logspace(np.log10(x_lo), np.log10(x_hi), 800)
    envelope, best_idx = _pareto_envelope_and_owner(colored, grid)
    finite = np.isfinite(envelope)
    return {colored[i][0] for i in np.unique(best_idx[finite]) if i >= 0}


_CB_MARGIN = 0.05
_CB_SPACING = 0.03
_CB_MIN_W = 0.10


def _family_colorbar_row_count(n_cb: int) -> int:
    """How many colorbar rows _draw_family_colorbars will use for n_cb families.

    Callers need this before the figure exists (to size it), so it has to
    match _draw_family_colorbars's own row-wrap decision exactly rather than
    duplicating a separately-hardcoded threshold that could drift out of sync.
    """
    if n_cb == 0:
        return 0
    max_single_row = int((1.0 - 2 * _CB_MARGIN + _CB_SPACING) // (_CB_MIN_W + _CB_SPACING))
    return 1 if n_cb <= max_single_row else 2


def _draw_family_colorbars(
    fig,
    family_labels: dict[str, list[str]],
    family_p_vals: dict[str, list[int]],
    *,
    row_h: float = 0.042,
    row_gap: float = 0.15,
    bottom: float = 0.05,
) -> None:
    """Draw one horizontal colorbar per method family, showing its p-depth gradient.

    Every colorbar is the same width, whether there's one row or two -- a row
    with fewer items is centered rather than stretched to fill the width, so
    a 2-item row doesn't end up visibly wider than a 3-item row next to it.
    Wraps onto a second row only once a single row would push colorbars
    narrower than ``min_cb_w`` (families' titles, usually wider than the bar
    itself, ran into each other when a wide single row got squeezed).
    ``row_h``/``row_gap``/``bottom`` are fractions of the figure height;
    callers reserve enough total height for however many rows this ends up
    drawing (see the ``n_rows`` calc at each call site).
    """
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable

    cb_families = [f for f in _FAMILY_ORDER if f in family_labels]
    n_cb = len(cb_families)
    if n_cb == 0:
        return
    margin, spacing = _CB_MARGIN, _CB_SPACING
    n_rows = _family_colorbar_row_count(n_cb)
    per_row = -(-n_cb // n_rows)  # ceil: the widest row sets the uniform width
    rows = [cb_families[i : i + per_row] for i in range(0, n_cb, per_row)]
    cb_w = (1.0 - 2 * margin - (per_row - 1) * spacing) / per_row

    for row_idx, row_families in enumerate(rows):
        n_in_row = len(row_families)
        # Center a shorter row instead of stretching its items to fill the
        # width the widest row uses.
        content_w = n_in_row * cb_w + (n_in_row - 1) * spacing
        row_start = margin + ((1.0 - 2 * margin) - content_w) / 2
        # Row 0 is the bottom-most row; later rows stack upward.
        cb_bottom = bottom + row_idx * row_gap
        for i, fam in enumerate(row_families):
            p_vals = family_p_vals.get(fam, [])
            cmap_fn, lo, hi = _FAMILY_CMAP_SPEC.get(fam, (plt.cm.viridis, 0.3, 0.9))
            # Build a 2-stop gradient matching the curve colours for this family.
            c_lo = cmap_fn(lo)
            c_hi = cmap_fn(hi)
            grad_cmap = LinearSegmentedColormap.from_list("", [c_lo, c_hi])
            ax_cb = fig.add_axes([
                row_start + i * (cb_w + spacing),
                cb_bottom,
                cb_w,
                row_h,
            ])
            p_min = min(p_vals) if p_vals else 0
            p_max = max(p_vals) if p_vals else 1
            norm = Normalize(vmin=p_min - 0.5, vmax=p_max + 0.5)
            sm = ScalarMappable(cmap=grad_cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
            cb.set_ticks(p_vals if p_vals else [p_min, p_max])
            cb.ax.tick_params(labelsize=11)
            ax_cb.set_title(_FAMILY_DISPLAY.get(fam, fam), fontsize=12, pad=3)
            # "circuit depth p" as an axis label below the tick numbers (not
            # folded into the title above) so it's clear those numbers are p,
            # not just floating digits.
            ax_cb.set_xlabel("circuit depth $p$", fontsize=11, labelpad=2)


def plot_multi_method_window_sticker_component_panels(
    *,
    training_virtual_best_df: pd.DataFrame,
    training_fitted_prescription_df: pd.DataFrame,
    test_virtual_best_df: pd.DataFrame,
    test_fitted_prescription_df: pd.DataFrame,
    plot_dir: str | Path,
    filename: str,
    approx_ylim: tuple[float, float] | None = None,
    approx_yticks: list[float] | None = None,
    show_ci: bool = True,
    xlim: tuple[float, float] | None = None,
    extend_curves_to_xlim: bool = False,
) -> None:
    """Plot training and test multi-method Window Sticker curves as shared-y panels.

    Depths are annotated in red on each virtual-best curve so the legend can be
    collapsed to one entry per method family.  The legend is placed in the gap
    between the two panels.
    """

    style_map = {
        "Virtual best": {
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.6,
            "markersize": 3.8,
            "alpha": 1.0,
            "zorder": 6,
        },
        "Actionable fit prescription": {
            "linestyle": "-.",
            "marker": "s",
            "linewidth": 1.7,
            "markersize": 3.0,
            "alpha": 0.85,
            "zorder": 3,
        },
    }

    panel_data = [
        (
            "Training instances",
            {
                "Actionable fit prescription": curve_from_response_summary(training_fitted_prescription_df),
                "Virtual best": curve_from_response_summary(training_virtual_best_df),
            },
        ),
        (
            "Test instances",
            {
                "Actionable fit prescription": curve_from_response_summary(test_fitted_prescription_df),
                "Virtual best": curve_from_response_summary(test_virtual_best_df),
            },
        ),
    ]
    if all(curve.empty for _, curves in panel_data for curve in curves.values()):
        print(f"Skipping {filename}: no multi-method Window Sticker curves found.")
        return

    labels = sorted(
        {
            label
            for _, curves in panel_data
            for curve in curves.values()
            if not curve.empty
            for label in curve["method_label"].dropna().astype(str).unique()
        }
    )
    _ = approx_ylim, approx_yticks

    color_map, family_labels, family_p_vals = _build_family_color_map(labels)

    # Colorbars wrap onto a second row once a single row would push them
    # narrower than a readable minimum (see _family_colorbar_row_count /
    # _draw_family_colorbars) -- cramming 5-6 into one row squeezed each
    # colorbar's title (usually wider than the bar) into its neighbours.
    # Grow the figure to make room rather than shrinking the main panels.
    _base_h = 5.8
    _two_cb_rows = _family_colorbar_row_count(len(family_labels)) == 2
    _fig_h = _base_h + 1.5 if _two_cb_rows else _base_h
    fig, axes = plt.subplots(1, 2, figsize=(13.2, _fig_h), sharey=True)

    all_y: list[float] = []
    panel_annotations: list[list[tuple[float, float, str]]] = [[], []]
    # Actionable prescription data collected per panel for Pareto envelope + bg shading.
    panel_actionable_curves: list[list[tuple[str, str, np.ndarray, np.ndarray]]] = [[], []]

    for panel_idx, (panel_label, curves) in enumerate(panel_data):
        ax = axes[panel_idx]
        panel_x: list[float] = []
        panel_y: list[float] = []
        for curve_name, curve in curves.items():
            if curve.empty:
                continue
            style = style_map[curve_name]
            for label, group in curve.groupby("method_label"):
                group = group.loc[:, ~group.columns.duplicated()].copy()
                group["resource"] = pd.to_numeric(group["resource"], errors="coerce")
                group["response_monotone"] = pd.to_numeric(group["response_monotone"], errors="coerce")
                group = group.dropna(subset=["resource", "response_monotone"])
                group = group[group["resource"] > 0].sort_values("resource")
                if xlim is not None:
                    group = group[(group["resource"] >= xlim[0]) & (group["resource"] <= xlim[1])]
                if group.empty:
                    continue
                # Natural (pre-extension) endpoint and series: used for the depth
                # annotation and for the background-dominance matrix below, so a
                # method never gets credited past where it actually has data.
                natural_last_x = float(group["resource"].iloc[-1])
                natural_last_y = float(group["response_monotone"].iloc[-1]) * 100.0
                natural_resource = group["resource"].to_numpy(dtype=float)
                natural_response_percent = group["response_monotone"].to_numpy(dtype=float) * 100.0

                # Extend to right boundary: hold last value constant so every curve
                # reaches the plot edge regardless of where its data ends. This is a
                # display-only convenience for the drawn line; the background/Pareto
                # dominance computation below uses the natural series instead so it
                # doesn't treat this flat visual hold as evidence of real performance.
                # Drawn dashed and at reduced alpha below (not the solid natural-data
                # style) so a reader doesn't mistake the flat hold for a measurement.
                extend_to = None
                if extend_curves_to_xlim and xlim is not None and float(group["resource"].iloc[-1]) < xlim[1]:
                    extend_to = xlim[1]

                lower_col = "response_lower_monotone" if "response_lower_monotone" in group.columns else "response_lower"
                upper_col = "response_upper_monotone" if "response_upper_monotone" in group.columns else "response_upper"
                if show_ci and lower_col in group.columns and upper_col in group.columns:
                    lower = pd.to_numeric(group[lower_col], errors="coerce")
                    upper = pd.to_numeric(group[upper_col], errors="coerce")
                    band = group.assign(_lower=lower, _upper=upper).dropna(
                        subset=["resource", "_lower", "_upper"]
                    )
                    if not band.empty:
                        band_lower = band["_lower"].to_numpy(dtype=float) * 100.0
                        band_upper = band["_upper"].to_numpy(dtype=float) * 100.0
                        ax.fill_between(
                            band["resource"].to_numpy(dtype=float),
                            band_lower,
                            band_upper,
                            color=color_map[str(label)],
                            alpha=0.14 if curve_name == "Virtual best" else 0.08,
                            linewidth=0,
                            zorder=max(1, style["zorder"] - 2),
                        )
                response_percent = group["response_monotone"].to_numpy(dtype=float) * 100.0
                ax.plot(
                    group["resource"],
                    response_percent,
                    color=color_map[str(label)],
                    label=None,
                    **style,
                )
                panel_x.extend(group["resource"].to_numpy(dtype=float))
                panel_y.extend(response_percent)

                if extend_to is not None:
                    ext_style = dict(style)
                    ext_style["linestyle"] = "--"
                    ext_style["marker"] = None
                    ext_style["alpha"] = style.get("alpha", 1.0) * 0.5
                    ax.plot(
                        [natural_last_x, extend_to],
                        [natural_last_y, natural_last_y],
                        color=color_map[str(label)],
                        label=None,
                        **ext_style,
                    )
                    panel_x.append(extend_to)
                    panel_y.append(natural_last_y)

                # Collect depth annotation for virtual-best curves only.
                if curve_name == "Virtual best":
                    depth_match = re.search(r"\(p\s*=\s*(\d+)\)", str(label))
                    if depth_match:
                        panel_annotations[panel_idx].append(
                            (natural_last_x, natural_last_y, f"p={depth_match.group(1)}")
                        )

                # Collect actionable prescription data for Pareto envelope. Uses the
                # natural (pre-extension) series so the flat visual-continuity hold
                # doesn't count as evidence for the background-dominance computation.
                if curve_name == "Actionable fit prescription":
                    panel_actionable_curves[panel_idx].append((
                        str(label),
                        color_map[str(label)],
                        natural_resource,
                        natural_response_percent,
                    ))

        finite_x = np.asarray([x for x in panel_x if np.isfinite(x) and x > 0], dtype=float)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        elif finite_x.size:
            ax.set_xlim(float(finite_x.min()), float(finite_x.max()))
        all_y.extend(panel_y)
        ax.set_xscale("log")
        ax.set_xlabel(
            r"Resource ($T_{\mathrm{proxy}} = t_{\mathrm{preprocessing}} + t_{\mathrm{train}} + Qt_{\mathrm{shot}}$) [s]",
            fontsize=WINDOW_STICKER_LABEL_FONTSIZE,
        )
        ax.tick_params(axis="both", labelsize=WINDOW_STICKER_TICK_FONTSIZE)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(alpha=0.25)
        ax.text(
            0.03,
            0.97,
            f"({chr(ord('a') + panel_idx)}) {panel_label}",
            transform=ax.transAxes,
            fontsize=WINDOW_STICKER_LABEL_FONTSIZE,
            va="top",
            ha="left",
        )

    # Pareto envelope of actionable prescriptions + per-method background shading.
    # Drawn after ylim is set so background spans cover the full y range.
    for panel_idx, ax in enumerate(axes):
        entries = panel_actionable_curves[panel_idx]
        if not entries:
            continue
        x_lo, x_hi = ax.get_xlim()
        if x_lo <= 0 or x_hi <= x_lo:
            continue
        n_grid = 800
        grid = np.logspace(np.log10(x_lo), np.log10(x_hi), n_grid)
        method_colors = [e[1] for e in entries]
        envelope, best_idx = _pareto_envelope_and_owner(entries, grid)

        # Draw background spans for each contiguous dominance segment.
        i = 0
        while i < n_grid:
            seg = int(best_idx[i])
            j = i + 1
            while j < n_grid and int(best_idx[j]) == seg:
                j += 1
            if seg >= 0:
                ax.axvspan(
                    grid[i], grid[min(j, n_grid - 1)],
                    alpha=0.22, color=method_colors[seg], zorder=0, linewidth=0,
                )
            i = j

        # Draw Pareto envelope as black dotted line on top of everything.
        valid = np.isfinite(envelope)
        if valid.any():
            ax.plot(
                grid[valid], envelope[valid],
                color="black", linestyle=":", linewidth=2.2, zorder=8, label=None,
            )

    if all_y:
        finite_y = np.asarray([y for y in all_y if np.isfinite(y)], dtype=float)
        if finite_y.size:
            y_min = float(finite_y.min())
            y_max = float(finite_y.max())
            y_span = y_max - y_min
            y_pad_low = max(0.05, 0.01 * y_span) if y_span > 0 else 0.05
            y_pad_high = max(0.15, 0.025 * y_span) if y_span > 0 else 0.15
            for ax in axes:
                ax.set_ylim(y_min - y_pad_low, y_max + y_pad_high)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    axes[0].set_ylabel(
        "Approximation ratio (%)",
        fontsize=WINDOW_STICKER_LABEL_FONTSIZE,
        labelpad=10,
    )

    # Curve-type legend (Virtual best / Actionable / Pareto) -- method families
    # are identified by the colorbars below, so no per-family colour handles.
    curve_handles = [
        Line2D([0], [0], color="black", label=name, **style)
        for name, style in style_map.items()
    ]
    pareto_handle = Line2D(
        [0], [0], color="black", linestyle=":", linewidth=2.2,
        label="Pareto frontier (actionable)",
    )

    # Reserve bottom space: legend row + colorbar row(s). A two-row colorbar
    # strip needs more vertical room above it for the legend.
    cb_area_top = 0.27 if _two_cb_rows else 0.12
    leg_row_h = 0.10  # fraction for legend row
    bottom_reserved = cb_area_top + leg_row_h + 0.02

    fig.tight_layout(rect=[0.0, bottom_reserved, 1.0, 1.0], w_pad=2.0)

    # Place curve-type legend above colorbars.
    fig.legend(
        handles=curve_handles + [pareto_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, cb_area_top + 0.01),
        bbox_transform=fig.transFigure,
        frameon=True,
        ncol=len(curve_handles) + 1,
        fontsize=WINDOW_STICKER_LEGEND_FONTSIZE,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.2,
        labelspacing=0.4,
    )

    _draw_family_colorbars(fig, family_labels, family_p_vals)

    save_current_plot(filename, plot_dir)
    plt.show()


def plot_pareto_frontier_overlay(
    *,
    calibrations: list[dict[str, Any]],
    plot_dir: str | Path,
    filename: str,
    approx_ylim: tuple[float, float] | None = None,
) -> None:
    """Overlay the actionable Pareto frontier from several resource calibrations.

    Each entry in ``calibrations`` is a dict with keys ``label`` (legend name,
    e.g. "Noiseless"), ``linestyle`` (matplotlib linestyle distinguishing this
    calibration), ``training_fitted_prescription_df``, and
    ``test_fitted_prescription_df``. An optional ``marker`` key (e.g. "o" for
    Noiseless, "D" for Noise-corrected) adds a point marker along the line so
    the calibration reads clearly even where two curves sit close together;
    omit it to draw a plain line. The frontier is colour-coded by whichever
    method family holds it at each resource level, using the same family
    colours as plot_multi_method_window_sticker_component_panels, so a reader
    can read both which calibration a segment belongs to (linestyle/marker)
    and which strategy is recommended there (colour) off a single line.
    """
    panel_data = [
        ("Training instances", "training_fitted_prescription_df"),
        ("Test instances", "test_fitted_prescription_df"),
    ]

    all_labels = sorted({
        label
        for cal in calibrations
        for _, df_key in panel_data
        if not cal[df_key].empty
        for label in curve_from_response_summary(cal[df_key])["method_label"].dropna().astype(str).unique()
    })
    if not all_labels:
        print(f"Skipping {filename}: no calibrations with actionable-fit data found.")
        return
    color_map, family_labels, family_p_vals = _build_family_color_map(all_labels)

    def _build_entries(
        curve: pd.DataFrame,
    ) -> list[tuple[str, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Per-label (label, color, resource, response, ci_lower, ci_upper) entries.

        ci_lower/ci_upper are NaN-filled when the source has no CI columns, so
        callers can always index them without a separate has-CI branch.
        """
        lower_col = "response_lower_monotone" if "response_lower_monotone" in curve.columns else "response_lower"
        upper_col = "response_upper_monotone" if "response_upper_monotone" in curve.columns else "response_upper"
        has_ci = lower_col in curve.columns and upper_col in curve.columns
        entries: list[tuple[str, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for label, group in curve.groupby("method_label"):
            group = group.loc[:, ~group.columns.duplicated()].copy()
            group["resource"] = pd.to_numeric(group["resource"], errors="coerce")
            group["response_monotone"] = pd.to_numeric(group["response_monotone"], errors="coerce")
            if has_ci:
                group[lower_col] = pd.to_numeric(group[lower_col], errors="coerce")
                group[upper_col] = pd.to_numeric(group[upper_col], errors="coerce")
            group = group.dropna(subset=["resource", "response_monotone"])
            group = group[group["resource"] > 0].sort_values("resource")
            if group.empty:
                continue
            resource = group["resource"].to_numpy(dtype=float)
            response_percent = group["response_monotone"].to_numpy(dtype=float) * 100.0
            if has_ci:
                # response_lower/upper are a 95% CI (response +/- 1.96*SEM,
                # see _build_response_summary_from_rec_params); rescale the
                # half-width down to 1 SEM so error-bar whiskers are on the
                # same statistical basis wherever they're drawn.
                _ci_lower_raw = group[lower_col].to_numpy(dtype=float) * 100.0
                _ci_upper_raw = group[upper_col].to_numpy(dtype=float) * 100.0
                _sem_half_width = (_ci_upper_raw - _ci_lower_raw) / 2.0 / 1.96
                ci_lower = response_percent - _sem_half_width
                ci_upper = response_percent + _sem_half_width
            else:
                ci_lower = np.full_like(response_percent, np.nan)
                ci_upper = np.full_like(response_percent, np.nan)
            entries.append((str(label), color_map[str(label)], resource, response_percent, ci_lower, ci_upper))
        return entries

    def _envelope_winners(entries: list[tuple]) -> set[str]:
        """Labels that actually own at least one point of the drawn envelope."""
        multi_point_entries = [e for e in entries if len(e[2]) >= 2]
        if not multi_point_entries:
            return set()
        x_lo = min(e[2][0] for e in multi_point_entries)
        x_hi = max(e[2][-1] for e in multi_point_entries)
        if x_lo <= 0 or x_hi <= x_lo:
            return set()
        grid = np.logspace(np.log10(x_lo), np.log10(x_hi), 800)
        envelope, best_idx = _pareto_envelope_and_owner(entries, grid)
        finite = np.isfinite(envelope)
        return {entries[i][0] for i in np.unique(best_idx[finite]) if i >= 0}

    # The colorbar legend only shows families that actually own a segment of
    # the drawn envelope -- a strategy that's present in the comparison data
    # but never wins at any resource level (never becomes the visible line
    # colour) has no business taking up a colorbar. Figure out who actually
    # wins in a pre-pass so the figure can be sized before drawing.
    winning_labels: set[str] = set()
    for _, df_key in panel_data:
        for cal in calibrations:
            curve = curve_from_response_summary(cal[df_key])
            if curve.empty:
                continue
            winning_labels |= _envelope_winners(_build_entries(curve))

    dynamic_family_labels = {
        fam: [lbl for lbl in labels if lbl in winning_labels]
        for fam, labels in family_labels.items()
        if any(lbl in winning_labels for lbl in labels)
    }
    dynamic_family_p_vals = {
        fam: sorted({p for p in (_label_depth(lbl) for lbl in labels) if p is not None})
        for fam, labels in dynamic_family_labels.items()
    }

    # See plot_multi_method_window_sticker_component_panels for the row-wrap
    # logic; grow the figure to make room rather than shrinking the main panels.
    _base_h = 5.4
    _two_cb_rows = _family_colorbar_row_count(len(dynamic_family_labels)) == 2
    _fig_h = _base_h + 1.5 if _two_cb_rows else _base_h
    fig, axes = plt.subplots(1, 2, figsize=(13.2, _fig_h), sharey=True)
    all_y: list[float] = []

    for panel_idx, (panel_label, df_key) in enumerate(panel_data):
        ax = axes[panel_idx]
        panel_x: list[float] = []

        for cal in calibrations:
            curve = curve_from_response_summary(cal[df_key])
            if curve.empty:
                continue
            entries = _build_entries(curve)
            for e in entries:
                panel_x.extend(e[2].tolist())

            multi_point_entries = [e for e in entries if len(e[2]) >= 2]
            if not multi_point_entries:
                continue
            x_lo = min(e[2][0] for e in multi_point_entries)
            x_hi = max(e[2][-1] for e in multi_point_entries)
            if x_lo <= 0 or x_hi <= x_lo:
                continue
            n_grid = 800
            grid = np.logspace(np.log10(x_lo), np.log10(x_hi), n_grid)
            envelope, best_idx = _pareto_envelope_and_owner(entries, grid)
            method_colors = [e[1] for e in entries]
            bounds_entries = [(e[2], e[4], e[5]) for e in entries]
            ci_lower, ci_upper = _pareto_envelope_bounds(bounds_entries, grid, best_idx)

            # Draw the frontier as contiguous segments, colour = owning
            # family, linestyle = this calibration. One marker per segment,
            # placed at the start -- i.e. exactly where a strategy takes
            # over the frontier -- rather than evenly spaced along the line.
            _marker_idx: list[int] = []
            i = 0
            while i < n_grid:
                seg = int(best_idx[i])
                j = i + 1
                while j < n_grid and int(best_idx[j]) == seg:
                    j += 1
                if seg >= 0 and np.isfinite(envelope[i:j]).any():
                    ax.plot(
                        grid[i:j], envelope[i:j],
                        color=method_colors[seg],
                        linestyle=cal["linestyle"],
                        linewidth=2.6,
                        solid_capstyle="round",
                        zorder=5,
                        marker=cal.get("marker"),
                        markevery=[0] if cal.get("marker") else None,
                        markersize=10,
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                    )
                    if cal.get("marker"):
                        _marker_idx.append(i)
                i = j
            all_y.extend(envelope[np.isfinite(envelope)].tolist())

            # Error-bar whiskers at exactly the marker positions, coloured by
            # whichever family owns the envelope at that point.
            for _idx in _marker_idx:
                if not np.isfinite(envelope[_idx]) or best_idx[_idx] < 0:
                    continue
                _lo, _hi, _env = ci_lower[_idx], ci_upper[_idx], envelope[_idx]
                if not (np.isfinite(_lo) and np.isfinite(_hi)):
                    continue
                ax.errorbar(
                    grid[_idx], _env,
                    yerr=[[max(0.0, _env - _lo)], [max(0.0, _hi - _env)]],
                    fmt="none", ecolor=method_colors[int(best_idx[_idx])],
                    capsize=3, elinewidth=1.1, zorder=5.5,
                )

        finite_x = np.asarray([x for x in panel_x if np.isfinite(x) and x > 0], dtype=float)
        if finite_x.size:
            ax.set_xlim(float(finite_x.min()), float(finite_x.max()))
        ax.set_xscale("log")
        ax.set_xlabel(
            r"Resource ($T_{\mathrm{proxy}} = t_{\mathrm{preprocessing}} + t_{\mathrm{train}} + Qt_{\mathrm{shot}}$) [s]",
            fontsize=WINDOW_STICKER_LABEL_FONTSIZE,
        )
        ax.tick_params(axis="both", labelsize=WINDOW_STICKER_TICK_FONTSIZE)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(alpha=0.25)
        ax.text(
            0.03, 0.97, f"({chr(ord('a') + panel_idx)}) {panel_label}",
            transform=ax.transAxes, fontsize=WINDOW_STICKER_LABEL_FONTSIZE,
            va="top", ha="left",
        )

    if approx_ylim is not None:
        for ax in axes:
            ax.set_ylim(*approx_ylim)
    elif all_y:
        finite_y = np.asarray(all_y, dtype=float)
        y_min, y_max = float(finite_y.min()), float(finite_y.max())
        y_span = y_max - y_min
        pad_low = max(0.05, 0.01 * y_span) if y_span > 0 else 0.05
        pad_high = max(0.15, 0.025 * y_span) if y_span > 0 else 0.15
        for ax in axes:
            ax.set_ylim(y_min - pad_low, y_max + pad_high)

    axes[0].set_ylabel("Approximation ratio (%)", fontsize=WINDOW_STICKER_LABEL_FONTSIZE, labelpad=10)

    # Linestyle legend distinguishes calibrations; family colour is read off
    # the colorbars below, shared with the component-panel figures.
    calibration_handles = [
        Line2D(
            [0], [0], color="black", linestyle=cal["linestyle"], linewidth=2.6,
            marker=cal.get("marker"), markersize=10, label=cal["label"],
        )
        for cal in calibrations
    ]

    cb_area_top = 0.27 if _two_cb_rows else 0.12
    leg_row_h = 0.10
    bottom_reserved = cb_area_top + leg_row_h + 0.02
    fig.tight_layout(rect=[0.0, bottom_reserved, 1.0, 1.0], w_pad=2.0)

    fig.legend(
        handles=calibration_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, cb_area_top + 0.01),
        bbox_transform=fig.transFigure,
        frameon=True,
        ncol=len(calibration_handles),
        fontsize=WINDOW_STICKER_LEGEND_FONTSIZE,
        handlelength=2.4,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    _draw_family_colorbars(fig, dynamic_family_labels, dynamic_family_p_vals)

    save_current_plot(filename, plot_dir)
    plt.show()


def _prepare_parameter_curve(
    df: pd.DataFrame,
    *,
    resource_col: str,
    parameter_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Collapse duplicate resource rows for parameter-prescription plotting."""
    required = {resource_col, *parameter_cols}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=[resource_col, *parameter_cols])
    curve = df.loc[:, [resource_col, *parameter_cols]].copy()
    curve[resource_col] = pd.to_numeric(curve[resource_col], errors="coerce")
    for col in parameter_cols:
        curve[col] = pd.to_numeric(curve[col], errors="coerce")
    curve = curve.dropna(subset=[resource_col, *parameter_cols])
    curve = curve[curve[resource_col] > 0]
    if curve.empty:
        return curve
    return (
        curve.groupby(resource_col, as_index=False)
        .median(numeric_only=True)
        .sort_values(resource_col)
        .reset_index(drop=True)
    )


def plot_parameter_prescriptions_by_resource(
    *,
    candidate_df: pd.DataFrame,
    virtual_best_df: pd.DataFrame,
    actionable_df: pd.DataFrame,
    plot_dir: str | Path,
    filename: str,
    title: str,
    xlabel: str,
    lookup_df: pd.DataFrame | None = None,
    candidate_resource_col: str = "T",
    curve_resource_col: str = "resource",
    parameter_cols: tuple[str, ...] = ("N", "M", "Q"),
    parameter_ylims: dict[str, tuple[float, float]] | None = None,
    connect_points: bool = True,
    connect_lookup: bool = True,
    show_virtual_best: bool = True,
    show_lookup: bool = False,
    actionable_marker: str | None = "s",
) -> None:
    """Plot N/M/Q prescriptions against a resource axis."""
    parameter_ylims = parameter_ylims or {}
    if candidate_df.empty or candidate_resource_col not in candidate_df.columns:
        print(f"Skipping {title}: candidate parameter data not found.")
        return

    parameter_sources = [candidate_df, virtual_best_df, actionable_df]
    if lookup_df is not None:
        parameter_sources.append(lookup_df)
    active_parameter_cols = tuple(
        col for col in parameter_cols
        if any(
            col in source.columns and (pd.to_numeric(source[col], errors="coerce") > 0).any()
            for source in parameter_sources
            if source is not None and not source.empty
        )
    )
    if not active_parameter_cols:
        print(f"Skipping {title}: no non-empty prescription parameter columns found.")
        return

    candidate_cols = [candidate_resource_col, *[col for col in active_parameter_cols if col in candidate_df.columns]]
    candidates = candidate_df.loc[:, candidate_cols].copy()
    candidates[candidate_resource_col] = pd.to_numeric(candidates[candidate_resource_col], errors="coerce")
    for col in active_parameter_cols:
        if col not in candidates.columns:
            candidates[col] = np.nan
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    candidates = candidates.dropna(subset=[candidate_resource_col])
    candidates = candidates[candidates[candidate_resource_col] > 0].drop_duplicates()
    if candidates.empty:
        print(f"Skipping {title}: no positive candidate resource values.")
        return

    method_label = None
    for source in [virtual_best_df, actionable_df, lookup_df, candidate_df]:
        if source is None or source.empty:
            continue
        if "method_label" in source.columns and source["method_label"].notna().any():
            method_label = str(source["method_label"].dropna().iloc[0])
            break
        if "strategy" in source.columns and source["strategy"].notna().any():
            method_label = str(source["strategy"].dropna().iloc[0])
            break
    curve_colors = window_sticker_curve_colors(method_label or "")
    curves = {}
    if show_virtual_best:
        curves["Average virtual-best winner"] = (
            _prepare_parameter_curve(
                virtual_best_df,
                resource_col=curve_resource_col,
                parameter_cols=active_parameter_cols,
            ),
            {"color": curve_colors["virtual_best"], "marker": "o", "linewidth": 2.1, "linestyle": "-"},
        )
    if show_lookup and lookup_df is not None:
        curves["Averaged virtual-best prescription"] = (
            _prepare_parameter_curve(
                lookup_df,
                resource_col=curve_resource_col,
                parameter_cols=active_parameter_cols,
            ),
            {"color": curve_colors["averaged_prescription"], "marker": "^", "linewidth": 1.8, "linestyle": "--"},
        )
    curves["Actionable fit prescription"] = (
        _prepare_parameter_curve(
            actionable_df,
            resource_col=curve_resource_col,
            parameter_cols=active_parameter_cols,
        ),
        {"color": "black", "marker": actionable_marker, "linewidth": 3.2, "linestyle": "-"},
    )

    fig, axes = plt.subplots(
        len(active_parameter_cols),
        1,
        figsize=(13.5, 3.6 * len(active_parameter_cols) + 1.0),
        sharex=True,
        constrained_layout=True,
    )
    if len(active_parameter_cols) == 1:
        axes = [axes]

    for ax, parameter in zip(axes, active_parameter_cols):
        candidate_points = candidates.dropna(subset=[parameter])
        ax.scatter(
            candidate_points[candidate_resource_col],
            candidate_points[parameter],
            s=24,
            color="0.28",
            alpha=0.55,
            linewidths=0,
            label="Candidate settings",
            zorder=1,
        )
        for label, (curve, style) in curves.items():
            if curve.empty:
                continue
            draw_connected = connect_points
            if label == "Averaged virtual-best prescription":
                draw_connected = connect_lookup
            if draw_connected:
                ax.plot(
                    curve[curve_resource_col],
                    curve[parameter],
                    label=label,
                    markersize=6,
                    zorder=3,
                    **style,
                )
            else:
                ax.scatter(
                    curve[curve_resource_col],
                    curve[parameter],
                    label=label,
                    s=42,
                    color=style["color"],
                    marker=style["marker"] or "o",
                    alpha=0.95,
                    zorder=3,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(parameter, fontsize=WINDOW_STICKER_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=WINDOW_STICKER_TICK_FONTSIZE)
        if parameter in parameter_ylims:
            ax.set_ylim(*parameter_ylims[parameter])
        else:
            y_values = []
            y_values.extend(pd.to_numeric(candidate_points[parameter], errors="coerce").dropna().tolist())
            for curve, _style in curves.values():
                if curve.empty or parameter not in curve.columns:
                    continue
                y_values.extend(pd.to_numeric(curve[parameter], errors="coerce").dropna().tolist())
            y_values = [float(value) for value in y_values if np.isfinite(value) and value > 0]
            if y_values:
                y_min = min(y_values)
                y_max = max(y_values)
                if np.isclose(y_min, y_max):
                    ax.set_ylim(y_min / 1.5, y_max * 1.5)
                else:
                    ax.set_ylim(y_min / 1.25, y_max * 1.25)
        ax.grid(alpha=0.35, which="major", linestyle="-.")
        ax.grid(alpha=0.12, which="minor", linestyle=":")

    axes[-1].set_xlabel(xlabel, fontsize=WINDOW_STICKER_LABEL_FONTSIZE)
    if title:
        fig.suptitle(title, fontsize=WINDOW_STICKER_TITLE_FONTSIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        frameon=True,
        fontsize=WINDOW_STICKER_LEGEND_FONTSIZE,
    )
    save_current_plot(filename, plot_dir)
    plt.show()

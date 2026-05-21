import json
import logging
import os

import df_utils
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import training
from matplotlib.collections import LineCollection

monotone = False
plot_vb_CI = True
dir_path = os.path.dirname(os.path.realpath(__file__))
ws_style = os.path.join(dir_path, "ws.mplstyle")

plt.style.use(ws_style)

logger = logging.getLogger(__name__)

MANIFEST_NAME = "plotting_manifest.json"
BASELINE_STEM = "baseline"


def _read_plot_csv(path):
    df = pd.read_csv(path)
    if "resource" not in df.columns and "index" in df.columns:
        df.rename(columns={"index": "resource"}, inplace=True)

    drop_cols = [
        col
        for col in df.columns
        if col.startswith("Unnamed:")
        or col == "level_0"
        or (col == "index" and "resource" in df.columns)
    ]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    return df


def _is_preprocessed_params_stem(stem):
    return stem.endswith("_preproc") or stem.endswith("params")


class Plotting:
    """
    Plotting helpers that read exported plotting data from CSV files.

    Parameters
    ----------
    checkpoints_dir : str or path-like
        Directory containing the plotting CSV subdirectories.
    """

    def __init__(self, checkpoints_dir):
        if hasattr(checkpoints_dir, "here") and hasattr(
            checkpoints_dir.here, "checkpoints"
        ):
            if hasattr(checkpoints_dir, "export_plot_csvs"):
                checkpoints_dir.export_plot_csvs(monotone=monotone)
            checkpoints_dir = checkpoints_dir.here.checkpoints

        self.checkpoints_dir = os.fspath(checkpoints_dir)
        self.params_dir = os.path.join(self.checkpoints_dir, "params_plotting")
        self.perf_dir = os.path.join(self.checkpoints_dir, "performance_plotting")
        self.meta_dir = os.path.join(self.checkpoints_dir, "meta_params_plotting")

        self.manifest = self._load_manifest()
        self.baseline_name = self.manifest.get("baseline_name", "baseline")
        self.response_key = self.manifest.get("response_key", "response")
        self.parameter_names = (
            self.manifest.get("parameter_names") or self._infer_parameter_names()
        )

        self.experiment_info = {
            exp["name"]: exp for exp in self.manifest.get("experiments", [])
        }
        self.experiment_names = (
            list(self.experiment_info) or self._infer_experiment_names()
        )

        self.colors = sns.color_palette("tab10", max(len(self.experiment_names), 1))
        self.baseline_color = "black"
        self._assign_experiment_colors()
        self.xscale = "log"

    def _load_manifest(self):
        manifest_path = os.path.join(self.checkpoints_dir, MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            return {}
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            logger.warning("Could not parse plotting manifest at %s", manifest_path)
            return {}

    def _infer_parameter_names(self):
        baseline_csv = self._params_path(BASELINE_STEM)
        if not os.path.exists(baseline_csv):
            return []
        df = _read_plot_csv(baseline_csv)
        return [col for col in df.columns if col != "resource"]

    def _infer_experiment_names(self):
        names = []
        for directory in [self.params_dir, self.perf_dir]:
            if not os.path.exists(directory):
                continue
            for filename in sorted(os.listdir(directory)):
                stem, ext = os.path.splitext(filename)
                if (
                    ext != ".csv"
                    or stem == BASELINE_STEM
                    or _is_preprocessed_params_stem(stem)
                ):
                    continue
                if stem not in names:
                    names.append(stem)
        return names

    def _assign_experiment_colors(self):
        if len(self.colors) == 0:
            self.experiment_colors = {}
            return
        self.experiment_colors = {
            name: self.colors[idx % len(self.colors)]
            for idx, name in enumerate(self.experiment_names)
        }

    def _params_path(self, name):
        return os.path.join(self.params_dir, f"{name}.csv")

    def _performance_path(self, name):
        return os.path.join(self.perf_dir, f"{name}.csv")

    def _meta_path(self, name):
        return os.path.join(self.meta_dir, f"{name}.csv")

    def _preprocessed_params_path(self, name):
        candidates = [
            os.path.join(self.params_dir, f"{name}_preproc.csv"),
            os.path.join(self.params_dir, f"{name}params.csv"),
        ]
        return next(
            (path for path in candidates if os.path.exists(path)), candidates[0]
        )

    def _preprocessed_meta_path(self, name):
        return os.path.join(self.meta_dir, f"{name}_preproc.csv")

    def _experiment_has_meta(self, name):
        info = self.experiment_info.get(name, {})
        if "has_meta" in info:
            return info["has_meta"]
        return os.path.exists(self._meta_path(name))

    def _meta_parameter_names(self, name, metaparams_df):
        info = self.experiment_info.get(name, {})
        if info.get("meta_parameter_names"):
            return [
                col
                for col in info["meta_parameter_names"]
                if col in metaparams_df.columns
            ]

        ignored = {
            "TotalBudget",
            "ExplorationBudget",
            "resource",
            "response",
            "response_lower",
            "response_upper",
            "count",
        }
        return [
            col
            for col in metaparams_df.columns
            if col not in ignored and not col.startswith("Key=")
        ]

    def _meta_resource_column(self, name, metaparams_df):
        info = self.experiment_info.get(name, {})
        resource_col = info.get("meta_resource", "TotalBudget")
        if resource_col in metaparams_df.columns:
            return resource_col
        if "TotalBudget" in metaparams_df.columns:
            return "TotalBudget"
        return "resource"

    def set_colors(self, cp):
        """
        Sets color palette and reassigns colors to experiments.
        """
        self.colors = cp
        self._assign_experiment_colors()

    def set_xlims(self, xlims):
        """
        Sets limits for shared x axis.
        """
        self.xlims = xlims

    def make_legend(self, ax, baseline_bool, experiment_bools):
        """
        Makes legend entries for the baseline and selected experiments.
        """
        if baseline_bool:
            color_patches = [
                mpatches.Patch(color=self.baseline_color, label=self.baseline_name)
            ]
        else:
            color_patches = []

        for idx, name in enumerate(self.experiment_names):
            if experiment_bools[idx]:
                color_patches.append(
                    mpatches.Patch(color=self.experiment_colors[name], label=name)
                )

        ax.legend(handles=color_patches)

    def apply_shared(self, p, baseline_bool=True, experiment_bools=None):
        """
        Apply shared plot components such as x-scale, x-limits, and legends.
        """
        if experiment_bools is None:
            experiment_bools = [True] * len(self.experiment_names)

        if type(p) is dict:
            for k, v in p.items():
                p[k] = self.apply_shared(v, baseline_bool, experiment_bools)
            return p

        p = p.scale(x=self.xscale)
        if hasattr(self, "xlims"):
            p = p.limit(x=self.xlims)

        fig = plt.figure()
        p = p.on(fig).plot()
        ax = fig.axes[0]
        self.make_legend(ax, baseline_bool, experiment_bools)

        return fig

    def _plot_baseline_parameter_trace(self, ax, params_df, eval_df, param):
        points = np.array(
            [params_df["resource"].values, params_df[param].values]
        ).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(eval_df["response"].min(), eval_df["response"].max())
        lc = LineCollection(segments, cmap="Spectral", norm=norm)
        lc.set_array(eval_df["response"].values)
        lc.set_label(self.baseline_name)
        lc.set_linewidth(8)
        lc.set_alpha(0.75)
        line = ax.add_collection(lc)
        ax.plot(params_df["resource"], params_df[param], "o", ms=2, mec="k", alpha=0.25)
        return line

    def _plot_experiment_parameter_traces(self, axes):
        for experiment_name in self.experiment_names:
            params_path = self._params_path(experiment_name)
            if not os.path.exists(params_path):
                continue

            params_df = _read_plot_csv(params_path)
            preproc_path = self._preprocessed_params_path(experiment_name)
            has_meta = self._experiment_has_meta(experiment_name)

            for param in self.parameter_names:
                if param not in params_df.columns:
                    continue

                if not has_meta:
                    axes[param].plot(
                        params_df["resource"],
                        params_df[param],
                        "o-",
                        ms=2,
                        lw=1.5,
                        color=self.experiment_colors[experiment_name],
                        label=experiment_name,
                    )

                if os.path.exists(preproc_path):
                    preproc_params = _read_plot_csv(preproc_path)
                    if param in preproc_params.columns:
                        axes[param].plot(
                            preproc_params["resource"],
                            preproc_params[param],
                            color=self.experiment_colors[experiment_name],
                            marker="x",
                            linestyle=":",
                            ms=2,
                            lw=1.5,
                        )

    def plot_parameters_together(self):
        """Plot the parameters in one figure with a subplot per parameter."""
        if not self.parameter_names:
            raise ValueError("No parameter CSV data found for plotting")

        fig, axes_list = plt.subplots(len(self.parameter_names), 1)
        axes_array = np.atleast_1d(axes_list).ravel()
        axes = {
            param: axes_array[ind] for ind, param in enumerate(self.parameter_names)
        }

        params_df = _read_plot_csv(self._params_path(BASELINE_STEM))
        eval_df = _read_plot_csv(self._performance_path(BASELINE_STEM))
        eval_df = df_utils.monotone_df(eval_df, "resource", "response", 1)

        line = None
        for param in self.parameter_names:
            line = self._plot_baseline_parameter_trace(
                axes[param], params_df, eval_df, param
            )

        if line is not None:
            cbar = fig.colorbar(line, ax=axes_array.tolist())
            cbar.ax.tick_params()
            cbar.set_label(self.response_key)

        self._plot_experiment_parameter_traces(axes)

        for param in self.parameter_names:
            axes[param].grid(axis="y")
            axes[param].set_ylabel(param)
            axes[param].set_xscale(self.xscale)
            axes[param].set_xlabel("Resource")
            if hasattr(self, "xlims"):
                axes[param].set_xlim(self.xlims)

        handles, labels = axes_array[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=[0.5, 0], loc="upper center")
        return fig, axes

    def plot_parameters_separate(self):
        """Plot each parameter in a separate figure."""
        if not self.parameter_names:
            raise ValueError("No parameter CSV data found for plotting")

        figs = {}
        axes = {}

        for param in self.parameter_names:
            figs[param], axes[param] = plt.subplots(1, 1)

        params_df = _read_plot_csv(self._params_path(BASELINE_STEM))
        eval_df = _read_plot_csv(self._performance_path(BASELINE_STEM))
        eval_df = df_utils.monotone_df(eval_df, "resource", "response", 1)

        for param in self.parameter_names:
            line = self._plot_baseline_parameter_trace(
                axes[param], params_df, eval_df, param
            )
            cbar = figs[param].colorbar(line, ax=axes[param])
            cbar.ax.tick_params()
            cbar.set_label(self.response_key)

        self._plot_experiment_parameter_traces(axes)

        for param in self.parameter_names:
            axes[param].grid(axis="y")
            axes[param].set_ylabel(param)
            axes[param].set_xscale(self.xscale)
            axes[param].set_xlabel("Resource")
            if hasattr(self, "xlims"):
                axes[param].set_xlim(self.xlims)
            axes[param].legend()
            figs[param].tight_layout()

        return figs, axes

    def plot_parameters_distance(self):
        """
        Plots the scaled distance between recommended parameters and virtual best.
        """
        recipes = _read_plot_csv(self._params_path(BASELINE_STEM))

        all_params_list = []
        for count, experiment_name in enumerate(self.experiment_names):
            params_path = self._params_path(experiment_name)
            if not os.path.exists(params_path):
                continue
            params_df = _read_plot_csv(params_path)
            params_df["exp_idx"] = count
            all_params_list.append(params_df)

        if not all_params_list:
            raise ValueError("No experiment parameter CSV data found for plotting")

        all_params = pd.concat(all_params_list, ignore_index=True)
        dist_params_list = []

        for _, recipe in recipes.iterrows():
            res_df = all_params[all_params["resource"] == recipe["resource"]]
            temp_df_eval = training.scaled_distance(
                res_df, recipe, self.parameter_names
            )
            temp_df_eval.loc[:, "resource"] = recipe["resource"]
            dist_params_list.append(temp_df_eval)
        all_params = pd.concat(dist_params_list, ignore_index=True)

        fig, axs = plt.subplots(1, 1)
        axs.plot(all_params["resource"], all_params["distance_scaled"])

        for idx, experiment_name in enumerate(self.experiment_names):
            params_df = all_params[all_params["exp_idx"] == idx]
            if len(params_df) == 0:
                continue

            if self._experiment_has_meta(experiment_name):
                marker = "x"
                linestyle = ":"
            else:
                marker = "o"
                linestyle = "-"

            axs.plot(
                params_df["resource"],
                params_df["distance_scaled"],
                marker=marker,
                linestyle=linestyle,
                color=self.experiment_colors[experiment_name],
                label=experiment_name,
            )

        axs.grid(axis="y")
        axs.set_ylabel("distance_scaled")
        axs.set_xscale(self.xscale)
        axs.set_xlabel("Resource")
        axs.legend(loc="best")
        fig.tight_layout()

        return fig, axs

    def plot_performance(self):
        """
        Plots monotonized performance for each experiment and the baseline.
        """
        eval_df = _read_plot_csv(self._performance_path(BASELINE_STEM))

        fig, axs = plt.subplots(1, 1)
        if plot_vb_CI:
            axs.fill_between(
                eval_df["resource"],
                eval_df["response_lower"],
                eval_df["response_upper"],
                alpha=0.25,
                color="k",
                lw=0,
            )
        axs.plot(
            eval_df["resource"],
            eval_df["response"],
            "o-",
            ms=5,
            lw=1,
            color=self.baseline_color,
            label=self.baseline_name,
        )

        for experiment_name in self.experiment_names:
            save_file = self._performance_path(experiment_name)
            if not os.path.exists(save_file):
                continue

            eval_df = _read_plot_csv(save_file)
            axs.fill_between(
                eval_df["resource"],
                eval_df["response_lower"],
                eval_df["response_upper"],
                alpha=0.25,
                color=self.experiment_colors[experiment_name],
                lw=0,
            )
            axs.plot(
                eval_df["resource"],
                eval_df["response"],
                "o-",
                ms=5,
                lw=1,
                color=self.experiment_colors[experiment_name],
                label=experiment_name,
            )

        axs.grid(axis="y")
        axs.set_ylabel(self.response_key)
        axs.set_xscale(self.xscale)
        axs.set_xlabel("Resource")
        if hasattr(self, "xlims"):
            axs.set_xlim(self.xlims)
        axs.legend(loc="lower right")
        fig.tight_layout()
        return fig, axs

    def plot_meta_parameters(self):
        """
        Plots meta parameters for experiments that have them.
        """
        figs = {}
        axes = {}

        for experiment_name in self.experiment_names:
            exp_figs = {}
            exp_axes = {}
            save_file = self._meta_path(experiment_name)
            if not os.path.exists(save_file):
                figs[experiment_name] = exp_figs
                axes[experiment_name] = exp_axes
                continue

            metaparams_df = _read_plot_csv(save_file)
            resource_col = self._meta_resource_column(experiment_name, metaparams_df)

            preproc_file = self._preprocessed_meta_path(experiment_name)
            metaparams_preproc_df = None
            if os.path.exists(preproc_file):
                metaparams_preproc_df = _read_plot_csv(preproc_file)

            for param in self._meta_parameter_names(experiment_name, metaparams_df):
                fig, axs = plt.subplots(1, 1)
                axs.plot(
                    metaparams_df[resource_col],
                    metaparams_df[param],
                    color=self.experiment_colors[experiment_name],
                    marker="o",
                    label=experiment_name,
                )
                if (
                    metaparams_preproc_df is not None
                    and param in metaparams_preproc_df.columns
                    and resource_col in metaparams_preproc_df.columns
                ):
                    axs.plot(
                        metaparams_preproc_df[resource_col],
                        metaparams_preproc_df[param],
                        color=self.experiment_colors[experiment_name],
                        marker="x",
                        linestyle="--",
                    )
                axs.grid(axis="y")
                axs.set_ylabel(param)
                axs.set_xscale(self.xscale)
                axs.set_xlabel(resource_col)
                axs.legend(loc="best")
                fig.tight_layout()
                exp_figs[param] = fig
                exp_axes[param] = axs

            figs[experiment_name] = exp_figs
            axes[experiment_name] = exp_axes
        return figs, axes

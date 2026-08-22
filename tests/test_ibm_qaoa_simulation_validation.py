import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IBM_QAOA_ROOT = REPO_ROOT / "examples" / "IBM_QAOA"
for path in (REPO_ROOT / "src", IBM_QAOA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.simulation_validation import (  # noqa: E402
    SAMPLED_BACKEND_METHODS,
    _cached_fa_grid_point,
    _pool_adjacent_violators,
    build_bound_circuit_simulator,
    fit_recommended_recipe_curves,
)
from src.utils import (  # noqa: E402
    _FAMILY_CMAP_SPEC,
    _build_family_color_map,
    _detect_method_family,
    _label_depth,
    _method_label_from_training_method,
    _pareto_envelope_and_owner,
)


class TestDetectMethodFamily:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("Fixed Angles$^\\star$ (p=5)", "FA_star"),
            ("Fixed Angles† (p=3)", "FA_dagger"),
            ("Fixed Angles$^\\dagger$ (p=2)", "FA_dagger"),
            ("Param. Transfer (p=5)", "PT"),
            ("PT_PP_AAA", "PT"),
            ("Parameter Transfer (p=1)", "PT"),
            ("Linear Ramp (p=5)", "LR"),
            ("linear_ramp_no_opt", "LR"),
            ("Interpolation (p=7)", "Interp"),
            ("I_MPSAer (p=7)", "Interp"),
            ("Linear Ramp (p=9)", "LR"),
            (r"Linear Ramp$^\star$ (p=5)", "LR_star"),
            ("Linear Ramp* (p=5)", "LR_star"),
            (r"Linear Ramp$^\dagger$ (p=9)", "LR_dagger"),
            ("Linear Ramp† (p=9)", "LR_dagger"),
        ],
    )
    def test_known_families(self, label, expected):
        assert _detect_method_family(label) == expected

    def test_pt_does_not_match_opt_substring(self):
        # 'FA_PP_opt' contains the substring 'pt' but is not a PT strategy;
        # \b treats '_' as a word char so a naive \bpt\b would still miss it,
        # this guards the (?<![a-z])pt(?![a-z]) boundary logic specifically.
        assert _detect_method_family("FA_PP_opt (p=5)") != "PT"

    def test_unrecognized_label_falls_back_to_truncated_string(self):
        label = "some totally unknown strategy name"
        assert _detect_method_family(label) == label[:20]


class TestLabelDepth:
    def test_extracts_depth(self):
        assert _label_depth("Fixed Angles (p=5)") == 5

    def test_extracts_depth_with_spaces(self):
        assert _label_depth("Fixed Angles (p = 12)") == 12

    def test_no_depth_returns_none(self):
        assert _label_depth("Fixed Angles") is None


class TestBuildFamilyColorMap:
    def test_fa_star_and_fa_dagger_do_not_collide(self):
        # Regression test for the color-collision bug: FA_star and FA_dagger
        # used to both draw from overlapping stretches of plt.cm.Blues
        # (0.38-0.92 vs 0.32-0.58), making e.g. Fixed Angles*(p=6) and
        # Fixed Angles-dagger(p=6) nearly indistinguishable even though that
        # contrast is the point of the figure. Both stay on plt.cm.Blues
        # (matching the paper's single Fixed-Angles blue) with disjoint bands.
        labels = [
            "Fixed Angles$^\\star$ (p=2)", "Fixed Angles$^\\star$ (p=5)", "Fixed Angles$^\\star$ (p=6)",
            "Fixed Angles† (p=2)", "Fixed Angles† (p=3)", "Fixed Angles† (p=6)",
            "Param. Transfer (p=2)", "Param. Transfer (p=5)",
        ]
        color_map, _, _ = _build_family_color_map(labels)

        def dist(c1, c2):
            return sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3])) ** 0.5

        fa_star = [l for l in labels if _detect_method_family(l) == "FA_star"]
        fa_dagger = [l for l in labels if _detect_method_family(l) == "FA_dagger"]
        worst = min(dist(color_map[a], color_map[b]) for a in fa_star for b in fa_dagger)
        assert worst > 0.1, f"FA_star/FA_dagger colors collide (distance={worst:.3f})"

    def test_lr_star_and_lr_do_not_collide(self):
        # Same collision check as FA_star/FA_dagger, for Linear Ramp's three
        # optimization tiers (LR_dagger = no opt, LR = ramp-param opt only,
        # LR_star = ramp-param + full angle opt) -- these used to all collapse
        # into one undifferentiated "LR" family/color.
        labels = [
            "Linear Ramp (p=5)", "Linear Ramp (p=9)",
            r"Linear Ramp$^\star$ (p=5)", r"Linear Ramp$^\star$ (p=9)",
        ]
        color_map, _, _ = _build_family_color_map(labels)

        def dist(c1, c2):
            return sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3])) ** 0.5

        lr = [l for l in labels if _detect_method_family(l) == "LR"]
        lr_star = [l for l in labels if _detect_method_family(l) == "LR_star"]
        assert lr and lr_star, "expected both LR and LR_star labels to be detected"
        worst = min(dist(color_map[a], color_map[b]) for a in lr for b in lr_star)
        assert worst > 0.1, f"LR/LR_star colors collide (distance={worst:.3f})"

    def test_linear_ramp_no_opt_maps_to_dagger_not_plain_lr(self):
        # Regression test: our own zero-training strategy name
        # ("linear_ramp_no_opt", see ZERO_TRAINING_METHODS in
        # simulation_validation.py) spells out "linear_ramp" instead of the
        # "LR" prefix every other Linear Ramp config uses, so it fell through
        # both the external QPS label formatter and the internal lookup table
        # unrecognized -- silently landing in the same "LR" (no-opt-tier)
        # family/color as LR_PP_opt instead of its own "LR_dagger" shade.
        latex_label = _method_label_from_training_method("linear_ramp_no_opt", format="latex")
        assert "dagger" in latex_label, f"expected a dagger marker, got {latex_label!r}"
        assert _detect_method_family(latex_label + " (p=9)") == "LR_dagger"

    def test_single_label_family_uses_midpoint_color(self):
        color_map, family_labels, family_p_vals = _build_family_color_map(["Linear Ramp (p=5)"])
        cmap_fn, lo, hi = _FAMILY_CMAP_SPEC["LR"]
        assert color_map["Linear Ramp (p=5)"] == cmap_fn((lo + hi) / 2)
        assert family_labels == {"LR": ["Linear Ramp (p=5)"]}
        assert family_p_vals == {"LR": [5]}

    def test_multi_depth_family_spans_low_to_high(self):
        labels = ["Linear Ramp (p=2)", "Linear Ramp (p=5)", "Linear Ramp (p=8)"]
        color_map, _, family_p_vals = _build_family_color_map(labels)
        cmap_fn, lo, hi = _FAMILY_CMAP_SPEC["LR"]
        assert color_map["Linear Ramp (p=2)"] == cmap_fn(lo)
        assert color_map["Linear Ramp (p=8)"] == cmap_fn(hi)
        assert family_p_vals["LR"] == [2, 5, 8]

    def test_label_without_depth_falls_back_to_method_color(self):
        color_map, _, _ = _build_family_color_map(["some totally unknown strategy"])
        # Should not raise, and should return a usable (non-None) color.
        assert color_map["some totally unknown strategy"] is not None


class TestParetoEnvelopeAndOwner:
    def test_running_max_switches_owner_at_crossover(self):
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        entries = [
            ("A", "blue", np.array([0.0, 4.0]), np.array([1.0, 1.0])),
            ("B", "red", np.array([0.0, 4.0]), np.array([0.0, 3.0])),
        ]
        envelope, best_idx = _pareto_envelope_and_owner(entries, grid)
        # A starts ahead (1.0 > 0.0), B overtakes once its interpolated value exceeds 1.0.
        assert best_idx[0] == 0
        assert best_idx[-1] == 1
        assert np.all(np.isfinite(envelope))

    def test_single_point_entry_is_ignored(self):
        grid = np.array([0.0, 1.0, 2.0])
        entries = [
            ("A", "blue", np.array([1.0]), np.array([5.0])),
        ]
        envelope, best_idx = _pareto_envelope_and_owner(entries, grid)
        assert np.all(np.isnan(envelope))
        assert np.all(best_idx == -1)

    def test_all_nan_before_any_entry_starts(self):
        grid = np.array([0.0, 1.0, 2.0, 3.0])
        entries = [
            ("A", "blue", np.array([2.0, 3.0]), np.array([1.0, 2.0])),
        ]
        envelope, best_idx = _pareto_envelope_and_owner(entries, grid)
        assert np.isnan(envelope[0]) and np.isnan(envelope[1])
        assert best_idx[0] == -1 and best_idx[1] == -1
        assert envelope[2] == 1.0
        assert envelope[3] == 2.0

    def test_running_max_persists_after_owning_methods_data_ends(self):
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        entries = [
            ("A", "blue", np.array([0.0, 1.0]), np.array([5.0, 5.0])),
            ("B", "red", np.array([2.0, 4.0]), np.array([1.0, 1.0])),
        ]
        envelope, best_idx = _pareto_envelope_and_owner(entries, grid)
        # A's record of 5.0 persists across the gap and past B's lower values.
        assert envelope[-1] == 5.0
        assert best_idx[-1] == 0


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit_aer") is None,
    reason="requires qiskit-aer (see requirements-examples.txt)",
)
class TestBuildBoundCircuitSimulator:
    def test_sv_backend_uses_statevector_method(self):
        sim = build_bound_circuit_simulator("SV")
        assert sim.options.method == SAMPLED_BACKEND_METHODS["SV"]

    def test_mps_alias_maps_to_mpsaer_method(self):
        sim = build_bound_circuit_simulator("MPS")
        assert sim.options.method == SAMPLED_BACKEND_METHODS["MPSAer"]

    def test_unsampled_backend_raises(self):
        with pytest.raises(ValueError):
            build_bound_circuit_simulator("not_a_real_backend")

    def test_simulator_options_are_forwarded(self):
        sim = build_bound_circuit_simulator("MPSAer", {"matrix_product_state_max_bond_dimension": 20})
        assert sim.options.matrix_product_state_max_bond_dimension == 20


class TestCachedFaGridPoint:
    def _group_df(self, points):
        """points: list of (n, m, [q1, q2, ...]) -> one row per (n, m, q)."""
        rows = []
        for n, m, qs in points:
            for q in qs:
                rows.append({"N": n, "M": m, "Q": q, "num_objective_evaluations": 42})
        return pd.DataFrame(rows)

    def test_fully_covered_point_is_returned(self):
        df = self._group_df([(10, 50, [100, 200])])
        result = _cached_fa_grid_point(df, n_value=10, m_value=50, q_values=[100, 200])
        assert result is not None
        assert len(result) == 2

    def test_partially_covered_point_is_not_cached(self):
        # Only Q=100 present; Q=200 was requested too, so this point isn't
        # resumable -- a single grid point's Q rows all come from one sampling
        # pass, there's no partial-Q state to pick back up from.
        df = self._group_df([(10, 50, [100])])
        result = _cached_fa_grid_point(df, n_value=10, m_value=50, q_values=[100, 200])
        assert result is None

    def test_missing_point_returns_none(self):
        df = self._group_df([(10, 50, [100])])
        result = _cached_fa_grid_point(df, n_value=20, m_value=50, q_values=[100])
        assert result is None

    def test_empty_group_returns_none(self):
        result = _cached_fa_grid_point(pd.DataFrame(), n_value=10, m_value=50, q_values=[100])
        assert result is None

    def test_does_not_match_other_n_m_combinations(self):
        # N=10,M=100 and N=20,M=50 both exist but N=10,M=50 (the one being
        # asked about) doesn't -- a naive OR-based filter could wrongly match.
        df = self._group_df([(10, 100, [100]), (20, 50, [100])])
        result = _cached_fa_grid_point(df, n_value=10, m_value=50, q_values=[100])
        assert result is None


class TestPoolAdjacentViolators:
    def test_already_monotonic_is_unchanged(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        fit = _pool_adjacent_violators(y, np.ones(4))
        assert np.allclose(fit, y)

    def test_violating_sequence_is_pooled_to_weighted_mean(self):
        # 5.0 -> 2.0 violates monotonicity; PAVA pools them (and anything
        # between) to their weighted mean rather than just clipping.
        y = np.array([1.0, 5.0, 2.0, 6.0])
        fit = _pool_adjacent_violators(y, np.ones(4))
        assert np.all(np.diff(fit) >= -1e-12)
        assert fit[1] == fit[2] == pytest.approx(3.5)

    def test_weights_affect_pooled_value(self):
        # A heavily-weighted low value should pull the pooled mean down more
        # than an unweighted PAVA would.
        y = np.array([1.0, 5.0, 2.0])
        fit_unweighted = _pool_adjacent_violators(y, np.array([1.0, 1.0, 1.0]))
        fit_weighted = _pool_adjacent_violators(y, np.array([1.0, 1.0, 10.0]))
        assert fit_weighted[1] < fit_unweighted[1]


class TestFitRecommendedRecipeCurvesMonotonic:
    def test_bump_shaped_data_produces_monotonic_fit(self):
        # Regression test for the bug this replaces: an unconstrained degree-2
        # log-log polynomial fit to data that rises then falls would itself
        # rise then fall (a parabola can't represent "rise then plateau"),
        # producing a nonsensical "use fewer test shots with more budget"
        # prescription. Isotonic regression must stay non-decreasing.
        resource = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512], dtype=float)
        q = np.array([300, 500, 1000, 2500, 5000, 6000, 5800, 4000, 3000, 2500], dtype=float)
        recipe_df = pd.DataFrame({"resource": resource, "Q": q, "N": resource * 2, "M": resource})

        fitted, model = fit_recommended_recipe_curves(
            recipe_df, fit_parameter_cols=("N", "M", "Q"), resource_col="resource"
        )
        assert np.all(np.diff(fitted["Q"].to_numpy()) >= -1e-9)
        assert set(model["parameter"]) == {"N", "M", "Q"}

    def test_monotonic_input_is_fit_closely(self):
        resource = np.array([1, 2, 4, 8, 16], dtype=float)
        n = np.array([10, 20, 40, 80, 160], dtype=float)
        recipe_df = pd.DataFrame({"resource": resource, "N": n})

        fitted, _ = fit_recommended_recipe_curves(
            recipe_df, fit_parameter_cols=("N",), resource_col="resource"
        )
        assert np.allclose(fitted["N"].to_numpy(), n, rtol=1e-6)

    def test_empty_input_returns_empty_frames(self):
        fitted, model = fit_recommended_recipe_curves(pd.DataFrame(), resource_col="resource")
        assert fitted.empty
        assert model.empty

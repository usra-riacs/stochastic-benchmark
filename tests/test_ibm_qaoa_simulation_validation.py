import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IBM_QAOA_ROOT = REPO_ROOT / "examples" / "IBM_QAOA"
for path in (REPO_ROOT / "src", IBM_QAOA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.simulation_validation import (  # noqa: E402
    SAMPLED_BACKEND_METHODS,
    build_bound_circuit_simulator,
)
from src.utils import _pareto_envelope_and_owner  # noqa: E402


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

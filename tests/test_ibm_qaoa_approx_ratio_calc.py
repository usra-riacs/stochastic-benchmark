"""Tests for the functions in examples/IBM_QAOA/src/approx_ratio_calc.py that
aren't already exercised by test_ibm_qaoa_processing.py's
test_maxcut_helpers_compute_energy_counts_and_ratio (which covers
load_maxcut_instance_context, counts_from_bitstring_samples,
maxcut_energy_from_bitstring, and maxcut_approximation_ratio).

Step 5 of the IBM_QAOA cleanup plan: get_minmax, extract_minmax_args, and
best_prefix_metrics had no coverage at all before this file.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IBM_QAOA_ROOT = REPO_ROOT / "examples" / "IBM_QAOA"
for path in (REPO_ROOT / "src", IBM_QAOA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.approx_ratio_calc import (  # noqa: E402
    best_prefix_metrics,
    extract_minmax_args,
    get_minmax,
)


# ---------------------------------------------------------------------------
# get_minmax
# ---------------------------------------------------------------------------

class TestGetMinmax:
    def test__get_minmax__given_heavy_hex__matches_by_instance_and_node_count(self, tmp_path):
        # ARRANGE
        graph_dir = tmp_path / "heavy_hex"
        graph_dir.mkdir()
        expected = graph_dir / "001_seed7_heavyhex_144nodes_v2.json"
        expected.touch()
        (graph_dir / "002_seed7_heavyhex_144nodes_v2.json").touch()

        # ACT
        result = get_minmax(str(tmp_path), "heavy_hex", "001", "144")

        # ASSERT
        assert result == expected

    def test__get_minmax__given_erdos_renyi__uses_er_probability_suffix(self, tmp_path):
        # ARRANGE
        graph_dir = tmp_path / "erdos_renyi"
        graph_dir.mkdir()
        expected = graph_dir / "001_20nodes_erdosrenyi30percent.json"
        expected.touch()
        (graph_dir / "001_20nodes_erdosrenyi50percent.json").touch()

        # ACT
        result = get_minmax(str(tmp_path), "erdos_renyi", "001", "20", ER_probability="30")

        # ASSERT
        assert result == expected

    def test__get_minmax__given_line_to_full__uses_swap_layers_suffix(self, tmp_path):
        # ARRANGE
        graph_dir = tmp_path / "line_to_full"
        graph_dir.mkdir()
        expected = graph_dir / "001_10nodes_3swap_layers.json"
        expected.touch()

        # ACT
        result = get_minmax(str(tmp_path), "line_to_full", "001", "10", swap_layers="3")

        # ASSERT
        assert result == expected

    def test__get_minmax__given_random_regular__uses_degree_suffix(self, tmp_path):
        # ARRANGE
        graph_dir = tmp_path / "random_regular"
        graph_dir.mkdir()
        expected = graph_dir / "001_10nodes_random3regular.json"
        expected.touch()

        # ACT
        result = get_minmax(str(tmp_path), "random_regular", "001", "10", degree="3")

        # ASSERT
        assert result == expected

    def test__get_minmax__given_unknown_graph_type__raises_valueerror(self, tmp_path):
        # ARRANGE / ACT / ASSERT
        with pytest.raises(ValueError):
            get_minmax(str(tmp_path), "not_a_graph_type", "001", "10")

    def test__get_minmax__given_no_matching_file__raises_filenotfounderror(self, tmp_path):
        # ARRANGE
        (tmp_path / "heavy_hex").mkdir()

        # ACT / ASSERT
        with pytest.raises(FileNotFoundError):
            get_minmax(str(tmp_path), "heavy_hex", "001", "144")

    def test__get_minmax__given_multiple_matching_files__raises_runtimeerror(self, tmp_path):
        # ARRANGE
        graph_dir = tmp_path / "heavy_hex"
        graph_dir.mkdir()
        (graph_dir / "001_a_heavyhex_144nodes.json").touch()
        (graph_dir / "001_b_heavyhex_144nodes.json").touch()

        # ACT / ASSERT
        with pytest.raises(RuntimeError):
            get_minmax(str(tmp_path), "heavy_hex", "001", "144")


# ---------------------------------------------------------------------------
# extract_minmax_args
# ---------------------------------------------------------------------------

class TestExtractMinmaxArgs:
    def test__extract_minmax_args__given_all_keys_present__returns_min_max_sum_tuple(self, tmp_path):
        # ARRANGE
        path = tmp_path / "minmax.json"
        path.write_text(json.dumps({"min_cut": 1.0, "max_cut": 5.0, "sum_of_weights": 6.0}))

        # ACT
        result = extract_minmax_args(path)

        # ASSERT
        assert result == (1.0, 5.0, 6.0)

    def test__extract_minmax_args__given_a_missing_key__returns_none(self, tmp_path):
        # ARRANGE -- no "sum_of_weights"
        path = tmp_path / "minmax.json"
        path.write_text(json.dumps({"min_cut": 1.0, "max_cut": 5.0}))

        # ACT
        result = extract_minmax_args(path)

        # ASSERT
        assert result is None


# ---------------------------------------------------------------------------
# best_prefix_metrics
#
# Uses the same 3-node/2-edge synthetic MaxCut instance as
# test_ibm_qaoa_notebook_helpers.py's TestBestBitstringAr (u=[0,1], v=[1,2],
# w=[1,1], sum_weights=2.0; min_cut=0, max_cut=2), so approximation ratios
# are hand-verifiable: "001" -> 0.5, "111" -> 0.0, "010" -> 1.0.
# ---------------------------------------------------------------------------

@pytest.fixture
def triangle_instance_context():
    return {
        "u": np.array([0, 1]),
        "v": np.array([1, 2]),
        "w": np.array([1.0, 1.0]),
        "sum_weights": 2.0,
    }


class TestBestPrefixMetrics:
    def test__best_prefix_metrics__given_empty_bitstrings__returns_empty_list(self, triangle_instance_context):
        # ARRANGE / ACT
        result = best_prefix_metrics([], [1], triangle_instance_context, 0.0, 2.0, 2.0)

        # ASSERT
        assert result == []

    def test__best_prefix_metrics__given_several_checkpoints__best_so_far_persists_until_beaten(self, triangle_instance_context):
        # ARRANGE -- shot 2 ("111", ratio 0.0) is worse than the running best
        # ("001", ratio 0.5) set at shot 1, so the Q=2 checkpoint should still
        # report "001" as the best bitstring; shot 3 ("010", ratio 1.0) then
        # beats it.
        bitstrings = ["001", "111", "010"]

        # ACT
        result = best_prefix_metrics(
            bitstrings, [1, 2, 3], triangle_instance_context, 0.0, 2.0, 2.0
        )

        # ASSERT
        assert [row["Q"] for row in result] == [1, 2, 3]
        assert result[0]["best_bitstring"] == "001"
        assert result[0]["best_approx_ratio"] == pytest.approx(0.5)
        assert result[0]["counts"] == {"001": 1}

        assert result[1]["best_bitstring"] == "001"  # unchanged: "111" didn't beat it
        assert result[1]["best_approx_ratio"] == pytest.approx(0.5)
        assert result[1]["counts"] == {"001": 1, "111": 1}

        assert result[2]["best_bitstring"] == "010"
        assert result[2]["best_approx_ratio"] == pytest.approx(1.0)
        assert result[2]["counts"] == {"001": 1, "111": 1, "010": 1}

    def test__best_prefix_metrics__given_checkpoint_beyond_stream_length__clips_to_stream_length(self, triangle_instance_context):
        # ARRANGE
        bitstrings = ["010"]

        # ACT
        result = best_prefix_metrics(
            bitstrings, [1000], triangle_instance_context, 0.0, 2.0, 2.0
        )

        # ASSERT -- clipped to the one available shot, so Q=1 not Q=1000
        assert [row["Q"] for row in result] == [1]

    def test__best_prefix_metrics__given_nonpositive_checkpoints__are_dropped(self, triangle_instance_context):
        # ARRANGE
        bitstrings = ["010", "001"]

        # ACT
        result = best_prefix_metrics(
            bitstrings, [0, -1], triangle_instance_context, 0.0, 2.0, 2.0
        )

        # ASSERT
        assert result == []

    def test__best_prefix_metrics__given_duplicate_checkpoints__collapses_to_one_row_each(self, triangle_instance_context):
        # ARRANGE -- checkpoints [2, 2, 3] on a 3-shot stream should produce
        # rows at Q=2 and Q=3 only, not two rows at Q=2.
        bitstrings = ["001", "111", "010"]

        # ACT
        result = best_prefix_metrics(
            bitstrings, [2, 2, 3], triangle_instance_context, 0.0, 2.0, 2.0
        )

        # ASSERT
        assert [row["Q"] for row in result] == [2, 3]

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

from src.Processing import (  # noqa: E402
    QAOATraining,
    expected_energy_from_counts,
    load_problem_instance,
    set_data_path,
)
from src.approx_ratio_calc import (  # noqa: E402
    counts_from_bitstring_samples,
    load_maxcut_instance_context,
    maxcut_approximation_ratio,
    maxcut_energy_from_bitstring,
)


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "ibm_qaoa"


def test_set_data_path_builds_training_and_hardware_paths(tmp_path):
    base = tmp_path / "qaoa_data"

    hardware_path, training_path = set_data_path(
        str(base),
        hardware=True,
        training=True,
        graph_type="heavy_hex",
    )

    assert hardware_path == str(base / "hardware" / "heavy_hex")
    assert training_path == str(base / "training" / "heavy_hex")


def test_load_problem_instance_filters_random_regular_degree(tmp_path):
    graph_dir = tmp_path / "instances" / "random_regular"
    graph_dir.mkdir(parents=True)
    expected = graph_dir / "001_10nodes_random3regular.json"
    wrong_degree = graph_dir / "001_10nodes_random4regular.json"
    expected.write_text("{}", encoding="utf-8")
    wrong_degree.write_text("{}", encoding="utf-8")

    matches = load_problem_instance(
        str(tmp_path / "instances"),
        graph_type="random_regular",
        num_nodes="10",
        instance="1",
        degree="3",
    )

    assert matches == [expected]


def test_qaoa_training_locate_and_load_fixture():
    matches = QAOATraining.locate_training_instance(
        str(FIXTURES_DIR),
        graph_type="random_regular",
        instance="1",
        num_nodes="10",
        p="1",
        degree="3",
    )

    assert len(matches) == 1
    training = QAOATraining.load_training_instance(matches[0])

    assert training.file_name == "001N10R3R_MC_FA_SV_noOpt_1.json"
    assert training.physical_file_name == matches[0].name
    assert training.train_duration_per_iter == [pytest.approx(0.04679393768310547)]
    assert training.expected_energy_per_iter == [pytest.approx(2.2201059527008287)]
    assert training.trainer_name_per_iter[0]["trainer_name"] == "FixedAngleConjecture"
    assert training.stage_summaries[0]["stage"] == 0
    assert training.stage_summaries[0]["optimized_qaoa_angles"] == [
        pytest.approx(0.3926720292447629),
        pytest.approx(0.615533629093832),
    ]


def test_expected_energy_from_counts_full_and_top_modes():
    def objective(bitstring, _context):
        return {"00": 0.0, "01": 1.0, "10": 2.0}[bitstring]

    counts = {"00": 1, "01": 1, "10": 2}

    assert expected_energy_from_counts(counts, objective) == pytest.approx(1.25)
    assert expected_energy_from_counts(counts, objective, energy_mode="top", top_pc=50) == pytest.approx(2.0)
    assert expected_energy_from_counts({}, objective) is None
    with pytest.raises(ValueError):
        expected_energy_from_counts(counts, objective, energy_mode="invalid")


def test_maxcut_helpers_compute_energy_counts_and_ratio(tmp_path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "edge list": [
                    {"nodes": [0, 1], "weight": 2.0},
                    {"nodes": [1, 2], "weight": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    context = load_maxcut_instance_context(graph_path)
    assert context["sum_weights"] == pytest.approx(3.0)
    assert np.array_equal(context["u"], np.array([0, 1]))
    assert counts_from_bitstring_samples(["010", "010", "111"]) == {"010": 2, "111": 1}

    energy = maxcut_energy_from_bitstring("010", context)
    assert energy == pytest.approx(1.5)
    assert maxcut_approximation_ratio(0.0, 3.0, 3.0, energy) == pytest.approx(1.0)

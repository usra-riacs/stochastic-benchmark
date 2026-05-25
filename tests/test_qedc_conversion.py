import importlib.util
import pickle
import sys
import types
from pathlib import Path


CONVERSION_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "QEDC_to_WS_conversion"
    / "conversion.py"
)


def load_conversion_module(module_name):
    spec = importlib.util.spec_from_file_location(module_name, CONVERSION_PATH)
    conversion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conversion)
    return conversion


def test_qedc_json_to_pkl_uses_qedc_metrics_module(tmp_path, monkeypatch):
    maxcut_benchmark = types.ModuleType("maxcut_benchmark")
    maxcut_benchmark.load_all_metrics = lambda folder: {
        "max_iter": 2,
        "max_circuits": 1,
        "min_qubits": 2,
        "objective_func_type": "approx_ratio",
    }

    metrics = types.ModuleType("qedclib.metrics")
    metrics.circuit_metrics_detail_2 = {
        "2": {
            1: {
                0: {"approx_ratio": 0.25},
                1: {"approx_ratio": 0.75},
            }
        }
    }
    qedclib = types.ModuleType("qedclib")
    qedclib.metrics = metrics

    monkeypatch.setitem(sys.modules, "maxcut_benchmark", maxcut_benchmark)
    monkeypatch.setitem(sys.modules, "qedclib", qedclib)
    monkeypatch.setitem(sys.modules, "qedclib.metrics", metrics)

    conversion = load_conversion_module("qedc_conversion_with_qedc")

    target_folder = tmp_path / "converted"
    df, gen_prop = conversion.json_to_pkl(
        tmp_path / "raw",
        load_from_json=True,
        target_folder=target_folder,
        target_file_name="all_metrics.pkl",
    )

    assert gen_prop["objective_func_type"] == "approx_ratio"
    assert df[1].tolist() == [0.25, 0.75]

    with (target_folder / "all_metrics.pkl").open("rb") as converted:
        stored_df = pickle.load(converted)
        stored_gen_prop = pickle.load(converted)

    assert stored_df[1].tolist() == [0.25, 0.75]
    assert stored_gen_prop == gen_prop


def test_qedc_json_to_pkl_reads_bundled_fixture_without_qedc_modules(tmp_path):
    conversion = load_conversion_module("qedc_conversion_without_qedc")
    conversion.maxcut_benchmark = None
    conversion.metrics = None

    raw_folder = (
        CONVERSION_PATH.parent
        / "__results"
        / "instance=0"
        / "approx_ratio"
        / "rounds-2_shots-50"
    )
    target_folder = tmp_path / "converted"

    df, gen_prop = conversion.json_to_pkl(
        raw_folder,
        load_from_json=True,
        target_folder=target_folder,
        target_file_name="all_metrics.pkl",
    )

    assert gen_prop["objective_func_type"] == "approx_ratio"
    assert gen_prop["num_shots"] == 50
    assert list(df.index) == [1, 2]
    assert df[1].tolist() == [0.52, 0.61]

    with (target_folder / "all_metrics.pkl").open("rb") as converted:
        stored_df = pickle.load(converted)
        stored_gen_prop = pickle.load(converted)

    assert stored_df[1].tolist() == [0.52, 0.61]
    assert stored_gen_prop == gen_prop

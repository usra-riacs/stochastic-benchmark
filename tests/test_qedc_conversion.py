import importlib.util
import pickle
import sys
import types
from pathlib import Path


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

    conversion_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "QEDC_to_WS_conversion"
        / "conversion.py"
    )
    spec = importlib.util.spec_from_file_location("qedc_conversion", conversion_path)
    conversion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conversion)

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

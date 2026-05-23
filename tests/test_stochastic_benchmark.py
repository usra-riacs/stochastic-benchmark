import pandas as pd
import pytest

import stochastic_benchmark as sb_module


def test_reduce_mem_bootstrap_recovers_checkpoint_pickles_without_exp_raw(
    tmp_path,
):
    """Existing bootstrap pickles should make reruns independent of exp_raw."""
    sb = sb_module.stochastic_benchmark(
        parameter_names=["param"],
        here=tmp_path,
        instance_cols=["instance"],
        reduce_mem=True,
        recover=True,
    )

    checkpoint_path = (
        tmp_path / "checkpoints" / "bootstrapped_results_inst=1.pkl"
    )
    pd.DataFrame({"param": [1], "instance": [1], "resource": [1]}).to_pickle(
        checkpoint_path
    )

    sb.run_Bootstrap(bsParams_iter=object())

    assert sb.bs_results == [str(checkpoint_path)]


def test_reduce_mem_bootstrap_recovers_expected_checkpoint_pickles_with_exp_raw(
    tmp_path, monkeypatch
):
    """Raw inputs should only recover the matching complete checkpoint set."""
    sb = sb_module.stochastic_benchmark(
        parameter_names=["param"],
        here=tmp_path,
        instance_cols=["instance"],
        reduce_mem=True,
        recover=True,
    )

    raw_dir = tmp_path / "exp_raw"
    raw_dir.mkdir()
    (raw_dir / "raw_results_inst=1.pkl").write_text("not a pickle")

    checkpoint_path = tmp_path / "checkpoints" / "bootstrapped_results_inst=1.pkl"
    pd.DataFrame({"param": [1], "instance": [1], "resource": [1]}).to_pickle(
        checkpoint_path
    )

    def fail_bootstrap_from_raw(*args, **kwargs):
        pytest.fail(
            "run_Bootstrap should not use exp_raw when checkpoint pickles exist"
        )

    monkeypatch.setattr(
        sb_module.bootstrap, "Bootstrap_reduce_mem", fail_bootstrap_from_raw
    )

    sb.run_Bootstrap(bsParams_iter=object(), group_name_fcn=lambda _: "inst=1")

    assert sb.bs_results == [str(checkpoint_path)]


def test_reduce_mem_bootstrap_does_not_recover_partial_checkpoint_set(
    tmp_path, monkeypatch
):
    """Partial checkpoints should not silently drop raw-data groups."""
    sb = sb_module.stochastic_benchmark(
        parameter_names=["param"],
        here=tmp_path,
        instance_cols=["instance"],
        reduce_mem=True,
        recover=True,
    )

    raw_dir = tmp_path / "exp_raw"
    raw_dir.mkdir()
    raw_files = [
        raw_dir / "raw_results_inst=1.pkl",
        raw_dir / "raw_results_inst=2.pkl",
    ]
    for raw_file in raw_files:
        raw_file.write_text("not a pickle")

    checkpoint_path = tmp_path / "checkpoints" / "bootstrapped_results_inst=1.pkl"
    pd.DataFrame({"param": [1], "instance": [1], "resource": [1]}).to_pickle(
        checkpoint_path
    )

    expected_results = ["rebuilt-results"]

    def bootstrap_from_raw(raw_data, *args, **kwargs):
        assert sorted(raw_data) == sorted(str(raw_file) for raw_file in raw_files)
        return expected_results

    monkeypatch.setattr(sb_module.bootstrap, "Bootstrap_reduce_mem", bootstrap_from_raw)

    def group_name(raw_filename):
        if raw_filename.endswith("inst=1.pkl"):
            return "inst=1"
        return "inst=2"

    sb.run_Bootstrap(bsParams_iter=object(), group_name_fcn=group_name)

    assert sb.bs_results == expected_results

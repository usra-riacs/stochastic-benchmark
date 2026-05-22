import os

import pandas as pd
import pytest

import stochastic_benchmark as sb_module


def test_reduce_mem_bootstrap_prefers_checkpoint_pickles_over_stale_exp_raw(
    tmp_path, monkeypatch
):
    """Existing bootstrap pickles should make reruns independent of exp_raw."""
    sb = sb_module.stochastic_benchmark(
        parameter_names=["param"],
        here=tmp_path,
        instance_cols=["instance"],
        reduce_mem=True,
        recover=True,
    )

    raw_dir = tmp_path / "exp_raw"
    raw_dir.mkdir()
    (raw_dir / "raw_results_inst=stale.pkl").write_text("not a pickle")

    checkpoint_path = (
        tmp_path / "checkpoints" / "bootstrapped_results_inst=1.pkl"
    )
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

    sb.run_Bootstrap(bsParams_iter=object(), group_name_fcn=lambda _: "missing")

    assert sb.bs_results == [os.fspath(checkpoint_path)]

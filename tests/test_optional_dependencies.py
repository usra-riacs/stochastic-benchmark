"""Tests for optional dependency surfaces."""

import ast
import builtins
import importlib
import re
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _requirements(path):
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _requirement_names(requirements):
    return {
        re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()
        for requirement in requirements
    }


def _setup_constant(name):
    tree = ast.parse((REPO_ROOT / "setup.py").read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} was not defined in setup.py")


def test_core_requirements_do_not_include_hyperopt():
    core_requirements = _requirements(REPO_ROOT / "requirements.txt")

    assert "hyperopt" not in _requirement_names(core_requirements)


def test_example_requirements_cover_self_contained_notebook_dependencies():
    example_requirements = _requirements(REPO_ROOT / "requirements-examples.txt")

    assert {"scikit-learn", "dimod", "nbconvert", "ipykernel"} <= _requirement_names(
        example_requirements
    )


def test_generation_requirements_are_separate_from_core():
    generation_requirements = _requirements(REPO_ROOT / "requirements-generation.txt")

    assert "hyperopt" in _requirement_names(generation_requirements)
    assert any(requirement == "setuptools<81" for requirement in generation_requirements)


def test_setup_extras_match_optional_dependency_groups():
    install_requires = _setup_constant("INSTALL_REQUIRES")
    extras_require = _setup_constant("EXTRAS_REQUIRE")

    assert "hyperopt>=0.2.7" not in install_requires
    assert {"scikit-learn>=1.3.0", "dimod>=0.12"} <= set(
        extras_require["examples"]
    )
    assert {"nbconvert>=7", "ipykernel>=6"} <= set(extras_require["notebooks"])
    assert {"hyperopt>=0.2.7", "setuptools<81"} <= set(
        extras_require["generation"]
    )


def test_wishart_generation_reports_clear_error_without_hyperopt(monkeypatch):
    example_dir = REPO_ROOT / "examples" / "wishart_n_50_alpha_0.5"
    previous_module = sys.modules.pop("wishart_runs", None)
    original_import = builtins.__import__

    def block_hyperopt(name, *args, **kwargs):
        if name == "hyperopt" or name.startswith("hyperopt."):
            raise ImportError("blocked hyperopt")
        return original_import(name, *args, **kwargs)

    monkeypatch.syspath_prepend(str(example_dir))
    monkeypatch.setitem(sys.modules, "dill", types.SimpleNamespace())
    monkeypatch.setattr(builtins, "__import__", block_hyperopt)

    try:
        wishart_runs = importlib.import_module("wishart_runs")

        with pytest.raises(RuntimeError, match="requirements-generation.txt"):
            wishart_runs.run_hyperopt(h=0, hpo_trial=0, instance_num=1)
    finally:
        sys.modules.pop("wishart_runs", None)
        if previous_module is not None:
            sys.modules["wishart_runs"] = previous_module

import ast
import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "wishart_n_50_alpha_0.5"
BLOCKED_GENERATION_MODULES = ("hyperopt", "pysa")


class BlockGenerationImports:
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == module or fullname.startswith(f"{module}.")
            for module in BLOCKED_GENERATION_MODULES
        ):
            raise ModuleNotFoundError(
                f"No module named '{fullname}'",
                name=fullname,
            )
        return None


def _tree(filename):
    return ast.parse((EXAMPLE_DIR / filename).read_text())


def _top_level_import_names(filename):
    names = []
    for node in _tree(filename).body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module)
    return names


def _clear_example_modules(monkeypatch):
    prefixes = (
        "wishart_paths",
        "wishart_runs",
        "wishart_ws",
        *BLOCKED_GENERATION_MODULES,
    )
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _import_with_generation_deps_blocked(module_name, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "src"))
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    _clear_example_modules(monkeypatch)
    monkeypatch.setattr(sys, "meta_path", [BlockGenerationImports(), *sys.meta_path])

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if any(
            missing == module or missing.startswith(f"{module}.")
            for module in BLOCKED_GENERATION_MODULES
        ):
            pytest.fail(f"{module_name} imported generation dependency {missing!r}")
        pytest.skip(f"Missing non-generation dependency required by the example: {missing}")


def test_wishart_ws_has_no_top_level_generation_imports():
    imports = _top_level_import_names("wishart_ws.py")

    assert "wishart_runs" not in imports
    assert "hyperopt" not in imports
    assert "pysa" not in imports


def test_wishart_runs_has_no_top_level_hyperopt_or_pysa_imports():
    imports = _top_level_import_names("wishart_runs.py")

    assert "hyperopt" not in imports
    assert "hyperopt.fmin" not in imports
    assert "pysa.sa" not in imports


def test_wishart_ws_import_does_not_require_generation_dependencies(monkeypatch):
    wishart_ws = _import_with_generation_deps_blocked("wishart_ws", monkeypatch)

    assert hasattr(wishart_ws, "stoch_bench_setup")
    assert "wishart_runs" not in sys.modules


def test_wishart_runs_import_does_not_require_generation_dependencies(monkeypatch):
    wishart_runs = _import_with_generation_deps_blocked("wishart_runs", monkeypatch)

    df_name, obj_name = wishart_runs.logname(1, 10, 2, 1.0, 50.0)
    assert df_name.endswith("inst=1_pcold=1.00_phot=50.0_replicas=2_sweeps=10.pkl")
    assert obj_name.endswith("obj_inst=1_pcold=1.00_phot=50.0_replicas=2_sweeps=10.pkl")

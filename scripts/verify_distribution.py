#!/usr/bin/env python3
"""Build and smoke-test the distribution without importing checkout sources."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path


def verify_distribution() -> None:
    root = Path(__file__).resolve().parents[1]
    modules = sorted(path.stem for path in (root / "src").glob("*.py"))
    style = "_stochastic_benchmark_assets/ws.mplstyle"

    with tempfile.TemporaryDirectory(prefix="stochastic-distribution-") as directory:
        workspace = Path(directory)
        dist = workspace / "dist"
        # The default build creates the sdist, then builds the wheel from it.
        subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(dist)],
            cwd=root,
            check=True,
        )
        wheel, = dist.glob("*.whl")
        sdist, = dist.glob("*.tar.gz")
        expected = {f"{module}.py" for module in modules} | {style}
        with zipfile.ZipFile(wheel) as archive:
            missing = expected - set(archive.namelist())
            if missing:
                raise RuntimeError(f"Wheel is missing library files: {sorted(missing)}")
        with tarfile.open(sdist) as archive:
            # Discard only the generated top-level distribution directory.
            members = {name.split("/", 1)[-1] for name in archive.getnames()}
            missing = {f"src/{name}" for name in expected} - members
            if missing:
                raise RuntimeError(f"Source distribution is missing library files: {sorted(missing)}")

        environment = workspace / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        executable = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        interpreter = environment / executable
        subprocess.run(
            [str(interpreter), "-I", "-m", "pip", "install", str(wheel)],
            cwd=workspace,
            check=True,
        )
        subprocess.run(
            [str(interpreter), "-I", "-c", INSTALLED_SMOKE, *modules],
            cwd=workspace,
            check=True,
        )


INSTALLED_SMOKE = """
import importlib
import importlib.metadata
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

environment = Path(sys.prefix).resolve()
for name in sys.argv[1:]:
    module = importlib.import_module(name)
    location = Path(module.__file__).resolve()
    assert location.is_relative_to(environment), (name, location)

import matplotlib.pyplot as plt
import plotting

style = Path(plotting.ws_style).resolve()
assert style.is_relative_to(environment), style
assert style.is_file(), style
assert plt.rcParams["xtick.direction"] == "in"
assert plt.rcParams["ytick.direction"] == "in"
figure, axis = plt.subplots()
axis.plot([1, 2, 3], [0.25, 0.5, 0.75])
output = Path("installed-plot.png")
figure.savefig(output)
plt.close(figure)
assert output.stat().st_size > 0
print("Installed distribution smoke test passed:",
      importlib.metadata.version("stochastic-benchmark"))
"""


if __name__ == "__main__":
    verify_distribution()

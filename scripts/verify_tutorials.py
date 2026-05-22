#!/usr/bin/env python3
"""Execute self-contained tutorial notebooks listed in the examples manifest."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RUN_CATEGORY = "self_contained"
VALID_CATEGORIES = {RUN_CATEGORY, "external", "slow"}


class ManifestError(ValueError):
    """Raised when the tutorial manifest is invalid."""


@dataclass(frozen=True)
class Tutorial:
    path: Path
    category: str
    reason: str = ""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _entry_path(entry: dict, index: int) -> Path:
    rel_path = entry.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        raise ManifestError(f"manifest entry {index} must include a notebook path")

    path = Path(rel_path)
    if path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"manifest entry {index} path must stay inside the repository")
    return path


def _entry_category(entry: dict, index: int) -> tuple[str, str]:
    category = entry.get("category")
    reason = entry.get("reason", "")

    if category not in VALID_CATEGORIES:
        raise ManifestError(
            f"manifest entry {index} category must be one of {sorted(VALID_CATEGORIES)}"
        )
    if category != RUN_CATEGORY and not reason:
        raise ManifestError(f"manifest entry {index} must include a skip reason")
    return category, reason


def _parse_manifest_entry(entry: dict, index: int, root: Path) -> Tutorial:
    if not isinstance(entry, dict):
        raise ManifestError(f"manifest entry {index} must be an object")

    rel_path = _entry_path(entry, index)
    category, reason = _entry_category(entry, index)
    notebook_path = root / rel_path
    if not notebook_path.is_file():
        raise ManifestError(f"notebook does not exist: {rel_path}")

    return Tutorial(path=notebook_path, category=category, reason=reason)


def load_manifest(manifest_path: Path, root: Path) -> list[Tutorial]:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{manifest_path} is not valid JSON: {exc}") from exc

    entries = raw.get("notebooks")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("manifest must contain a non-empty 'notebooks' list")

    return [
        _parse_manifest_entry(entry, index, root)
        for index, entry in enumerate(entries, start=1)
    ]


def build_nbconvert_command(
    notebook_path: Path,
    output_dir: Path,
    timeout: int,
    python_executable: str = sys.executable,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output",
        notebook_path.name,
        "--output-dir",
        str(output_dir),
        f"--ExecutePreprocessor.timeout={timeout}",
    ]


def _env_with_src_path(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing}" if existing else src_path
    return env


def run_tutorials(
    tutorials: list[Tutorial],
    root: Path,
    output_dir: Path,
    timeout: int,
    dry_run: bool = False,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = _env_with_src_path(root)

    for tutorial in tutorials:
        rel_path = tutorial.path.relative_to(root)
        if tutorial.category != RUN_CATEGORY:
            print(f"SKIP {rel_path} ({tutorial.category}): {tutorial.reason}", flush=True)
            continue

        tutorial_output_dir = output_dir / rel_path.parent
        tutorial_output_dir.mkdir(parents=True, exist_ok=True)
        command = build_nbconvert_command(tutorial.path, tutorial_output_dir, timeout)

        print(f"EXECUTE {rel_path}", flush=True)
        if dry_run:
            print(f"DRY-RUN {shlex.join(command)}", flush=True)
            continue

        result = subprocess.run(command, cwd=root, env=env, check=False)
        if result.returncode != 0:
            print(f"FAILED {rel_path} with exit code {result.returncode}", file=sys.stderr)
            return result.returncode

    print(f"Executed notebooks written to {output_dir}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "examples" / "tutorials.json",
        help="Path to the tutorial notebook manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "executed-notebooks",
        help="Directory where executed notebooks are written.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-notebook execution timeout in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list notebook commands without executing them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    root = repository_root()
    args = parse_args(argv)
    try:
        tutorials = load_manifest(args.manifest, root)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return run_tutorials(
        tutorials=tutorials,
        root=root,
        output_dir=args.output_dir,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())

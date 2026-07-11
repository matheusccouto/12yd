"""Tests for the v5 deployment manifest.

v5 drops lightgbm, plotly, huggingface_hub, sklearn from runtime deps.
v5 runtime deps: tabpfn-client, streamlit, httpx, numpy, pandas, packaging.

The manifest is pyproject.toml at the repo root, resolved to uv.lock.
Streamlit Community Cloud reads uv.lock (priority #1) and runs `uv sync --frozen`,
which installs [project.dependencies] — NOT [dependency-groups]. Runtime deps
must therefore live directly under [project.dependencies], never in groups.
requirements.txt was removed; ADR-0002 mandates pyproject.toml as the sole manifest.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from importlib import import_module
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
UV_LOCK_PATH = REPO_ROOT / "uv.lock"

DROPPED_DEPS = {"plotly", "lightgbm", "huggingface-hub", "scikit-learn", "sklearn"}
EXPECTED_RUNTIME_DEPS = {"httpx", "numpy", "pandas", "packaging", "streamlit", "tabpfn-client"}


# ---------------------------------------------------------------------------
# Runtime deps are importable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("package", ["httpx", "numpy", "pandas"])
def test_runtime_dep_is_importable(package: str) -> None:
    import_module(package)


def test_streamlit_is_importable() -> None:
    streamlit = import_module("streamlit")
    assert streamlit.__version__


def test_tabpfn_client_is_importable() -> None:
    tabpfn = import_module("tabpfn_client")
    assert tabpfn.__name__ == "tabpfn_client"


# ---------------------------------------------------------------------------
# pyproject.toml is the deployment manifest (ADR-0002)
# ---------------------------------------------------------------------------


def test_requirements_txt_does_not_exist() -> None:
    assert not REQUIREMENTS_PATH.exists(), (
        "requirements.txt must not exist — ADR-0002 designates pyproject.toml as "
        "the sole dependency manifest. requirements.txt would shadow uv.lock on "
        "Streamlit Community Cloud (priority order: uv.lock < requirements.txt) "
        "and reinstall the (now deleted) hand-maintained mirror instead of the "
        "local twelveyards package."
    )


def test_uv_lock_exists() -> None:
    assert UV_LOCK_PATH.exists(), (
        "uv.lock is required — Streamlit Community Cloud reads it (priority #1) "
        "and runs `uv sync --frozen` to install the local twelveyards package."
    )


def test_uv_lock_is_tracked_in_git() -> None:
    result = subprocess.run(
        ["git", "ls-files", "uv.lock"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "uv.lock", (
        "uv.lock must be committed to git — Streamlit Community Cloud clones the "
        "repo, so a gitignored uv.lock is invisible to it. Without uv.lock in the "
        "repo, Cloud falls back to pyproject.toml (poetry mode) or requirements.txt, "
        "neither of which builds and installs the local twelveyards package. "
        f"stdout={result.stdout!r}"
    )


def _parse_pyproject_dependencies() -> dict[str, str]:
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    deps: list[str] = data["project"]["dependencies"]
    parsed: dict[str, str] = {}
    for dep in deps:
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", dep.strip())
        assert match, f"unparseable dep: {dep!r}"
        name, specifier = match.group(1), match.group(2).strip()
        parsed[name.lower().replace("_", "-")] = specifier
    return parsed


def test_pyproject_contains_runtime_deps() -> None:
    deps = _parse_pyproject_dependencies()
    missing = EXPECTED_RUNTIME_DEPS - set(deps)
    assert not missing, (
        f"pyproject.toml [project.dependencies] missing runtime deps: "
        f"{sorted(missing)}. Streamlit Community Cloud runs `uv sync --frozen`, "
        f"which installs ONLY [project.dependencies] (and the default dev group), "
        f"NOT [dependency-groups]. Runtime deps must live here, not in groups."
    )


def test_pyproject_no_app_or_pipeline_groups() -> None:
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    groups = data.get("dependency-groups", {})
    leaked = {g for g in ("app", "pipeline") if g in groups}
    assert not leaked, (
        f"[dependency-groups] must not define {sorted(leaked)} — those groups are "
        f"NOT installed by `uv sync --frozen` on Streamlit Cloud. Their contents "
        f"must be flattened into [project.dependencies]."
    )


def test_pyproject_no_dropped_deps() -> None:
    deps = _parse_pyproject_dependencies()
    leaked = DROPPED_DEPS & set(deps)
    assert not leaked, f"v5 dropped deps {sorted(leaked)} must not reappear"


def test_pyproject_no_pinned_versions() -> None:
    deps = _parse_pyproject_dependencies()
    for name, spec in deps.items():
        assert "==" not in spec, (
            f"pyproject [project.dependencies] has a pinned (==) requirement: "
            f"{name}{spec!r} — use range specifiers (>=) instead"
        )


def test_uv_lock_is_in_sync() -> None:
    """uv.lock must be up to date with pyproject.toml.

    `uv sync --frozen` on Streamlit Cloud refuses to mutate uv.lock; a stale
    lock would silently skip newly-added runtime deps → ModuleNotFoundError
    at app boot (the exact bug class this test guards against).
    """
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "uv.lock is stale. Run `uv lock` locally and commit the result. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# app.py boot path is deployable (regression for the ModuleNotFoundError bug)
# ---------------------------------------------------------------------------


def test_app_imports_resolve_under_frozen_sync(tmp_path: Path) -> None:
    """Replay Streamlit Cloud's exact install path: fresh venv + `uv sync --frozen`.

    Guards against the regression where a runtime dep sits in a
    [dependency-groups] entry rather than [project.dependencies] — `uv sync`
    silently skips the group and the app boot fails at import time.
    """
    snapshot = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT,
        snapshot,
        ignore=shutil.ignore_patterns(
            ".venv",
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
        ),
    )
    subprocess.run(
        [sys.executable, "-m", "venv", str(snapshot / ".venv")],
        check=True,
    )
    subprocess.run(
        ["uv", "sync", "--frozen"],
        cwd=snapshot,
        check=True,
        capture_output=True,
        text=True,
    )
    check = subprocess.run(
        [
            str(snapshot / ".venv" / "bin" / "python"),
            "-c",
            "import pandas, streamlit; "
            "from twelveyards.artifacts import Artifacts; "
            "from twelveyards.dashboard import "
            "KickerPrediction, distinct_teams, predictions_for_match; "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, (
        "Boot path broken — a runtime dep is missing from pyproject "
        f"[project.dependencies]. stdout={check.stdout!r} "
        f"stderr={check.stderr!r}"
    )

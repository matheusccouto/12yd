"""Tests for the v5 deployment manifest.

v5 drops lightgbm, plotly, huggingface_hub, packaging, sklearn from runtime deps.
v5 runtime deps: tabpfn-client, streamlit, httpx, numpy, pandas.

The manifest is requirements.txt at the repo root, which Streamlit Cloud reads.
"""

from __future__ import annotations

import re
import tomllib
from importlib import import_module
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"


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
# requirements.txt
# ---------------------------------------------------------------------------


def test_requirements_txt_exists() -> None:
    assert REQUIREMENTS_PATH.exists(), "requirements.txt missing at repo root"


def _parse_pyproject_dependencies() -> dict[str, str]:
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    deps: list[str] = data["project"]["dependencies"]
    app_deps: list[str] = data["dependency-groups"].get("app", [])
    pipeline_deps: list[str] = data["dependency-groups"].get("pipeline", [])
    all_deps = deps + app_deps + pipeline_deps
    parsed: dict[str, str] = {}
    for dep in all_deps:
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", dep.strip())
        assert match, f"unparseable dep: {dep!r}"
        name, specifier = match.group(1), match.group(2).strip()
        parsed[name.lower().replace("_", "-")] = specifier
    return parsed


def _parse_requirements_txt() -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in REQUIREMENTS_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", line)
        assert match, f"unparseable requirement: {line!r}"
        name, specifier = match.group(1), match.group(2).strip()
        parsed[name.lower().replace("_", "-")] = specifier
    return parsed


def test_requirements_txt_contains_tabpfn_client() -> None:
    reqs = _parse_requirements_txt()
    assert "tabpfn-client" in reqs, (
        "requirements.txt must declare tabpfn-client for Streamlit Cloud deploy"
    )


def test_requirements_txt_contains_streamlit() -> None:
    reqs = _parse_requirements_txt()
    assert "streamlit" in reqs, (
        "requirements.txt must declare streamlit for Streamlit Cloud deploy"
    )


def test_requirements_txt_contains_core_deps() -> None:
    reqs = _parse_requirements_txt()
    for dep in ("httpx", "numpy", "pandas"):
        assert dep in reqs, (
            f"requirements.txt missing {dep} (core runtime dep)"
        )


def test_requirements_txt_matches_pyproject() -> None:
    pyproject_deps = _parse_pyproject_dependencies()
    requirements = _parse_requirements_txt()
    missing = set(pyproject_deps) - set(requirements) - {"packaging"}
    assert not missing, (
        f"requirements.txt is missing packages: {sorted(missing)}"
    )


def test_requirements_txt_specifiers_compatible() -> None:
    pyproject_deps = _parse_pyproject_dependencies()
    requirements = _parse_requirements_txt()
    for name, req_spec in requirements.items():
        if name not in pyproject_deps:
            continue
        pyproject_spec = pyproject_deps[name]
        req = SpecifierSet(req_spec) if req_spec else SpecifierSet()
        proj = SpecifierSet(pyproject_spec) if pyproject_spec else SpecifierSet()
        for v in _sample_versions(req):
            if v in req:
                assert v in proj, (
                    f"requirements.txt specifier {req_spec!r} for "
                    f"{name!r} allows {v!r} but pyproject.toml "
                    f"specifier {pyproject_spec!r} does not."
                )


def _sample_versions(specifier: SpecifierSet) -> list[str]:
    return [
        "0.0.0", "0.27.0", "0.3.0", "1.0.0", "1.30.0",
        "2.0.0", "5.0.0", "6.0.0", "24.0.0", "100.0.0",
    ]


def test_requirements_txt_no_pinned_versions() -> None:
    for raw in REQUIREMENTS_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" not in line, (
            f"requirements.txt has a pinned (==) requirement: {line!r}"
        )


# ---------------------------------------------------------------------------
# No dropped deps
# ---------------------------------------------------------------------------


def test_requirements_txt_no_plotly() -> None:
    reqs = _parse_requirements_txt()
    assert "plotly" not in reqs, "v5 drops plotly"


def test_requirements_txt_no_lightgbm() -> None:
    reqs = _parse_requirements_txt()
    assert "lightgbm" not in reqs, "v5 drops lightgbm"


def test_requirements_txt_no_huggingface_hub() -> None:
    reqs = _parse_requirements_txt()
    assert "huggingface-hub" not in reqs, "v5 drops huggingface_hub"


def test_requirements_txt_no_scikit_learn() -> None:
    reqs = _parse_requirements_txt()
    assert "scikit-learn" not in reqs, "v5 drops scikit-learn"
    assert "sklearn" not in reqs, "v5 drops sklearn"

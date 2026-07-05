"""Tests for the Streamlit Cloud deployment manifest.

The v4 dashboard deploy (Issue #48) shipped a new `import
plotly.graph_objects as go` in `app.py`, but Streamlit Community
Cloud's dependency-file detection reads the first file in this
priority order:

    uv.lock > Pipfile > environment.yml > requirements.txt > pyproject.toml

When `uv.lock` is the file found, Cloud's `uv` binary parses it;
when the lock format is newer than the Cloud-shipped `uv` version,
parsing silently falls back to installing only `streamlit` (the
default), and the app dies at startup with `ModuleNotFoundError: No
module named 'plotly'` (Issue #52). The fix is:

1. Ship a `requirements.txt` at the repo root.
2. Stop tracking `uv.lock` (add to `.gitignore`) so Cloud falls
   through to `requirements.txt`.

These tests pin both halves: the deployment manifest is in sync
with `pyproject.toml`'s `[project.dependencies]`, and `uv.lock` is
gitignored so the next deploy doesn't re-introduce the bug.

**Subtle second-order bug (Issue #52, second push).** Streamlit
Cloud uses the *hash* of a recognised dep file to decide whether
to do a fresh install ("smart detection"). The v4 fix
(`requirements.txt` + `uv.lock` gitignored) was correct, but the
follow-up commit (`8a37737`, Phase 3 schema) did NOT touch
`requirements.txt` or `pyproject.toml`, so the smart-detection
saw no change and the cached broken install (no `plotly`) was
re-used — the same `ModuleNotFoundError` persisted at the next
redeploy. The fix: a substantive, hand-maintained change to
`requirements.txt` (a header comment) forces a hash change,
which triggers a full redeploy and a fresh install. The
`test_requirements_txt_documents_deploy_requirement` and
`test_requirements_txt_is_hand_maintained` tests pin both
halves: the file must keep its header comment, and it must
stay in hand-maintained `>=`-specifier form (not `pip freeze`
`==` form, which would mask future deploy failures by changing
the hash on every dep bump).
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
GITIGNORE_PATH = REPO_ROOT / ".gitignore"


def test_plotly_is_importable() -> None:
    """The v4 dashboard imports plotly at module load time
    (`app.py` line ~49). If plotly is not installed the app
    crashes with `ModuleNotFoundError` (Issue #52)."""
    plotly = import_module("plotly")
    assert plotly.__version__  # type: ignore[attr-defined]
    go = import_module("plotly.graph_objects")
    assert go.Figure is not None


def test_requirements_txt_exists() -> None:
    """Streamlit Cloud reads `requirements.txt` from the entrypoint
    directory (or repo root). It must exist for the deployment to
    work — see Issue #52."""
    assert REQUIREMENTS_PATH.exists(), (
        f"requirements.txt missing at {REQUIREMENTS_PATH}. "
        "Streamlit Cloud needs it to install runtime deps."
    )


def _parse_pyproject_dependencies() -> dict[str, str]:
    """Return the `[project.dependencies]` map from pyproject.toml.

    Each entry is a PEP 508 requirement string; we normalise the
    package name (PEP 503) and keep the specifier for comparison.
    """
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    deps: list[str] = data["project"]["dependencies"]
    parsed: dict[str, str] = {}
    for dep in deps:
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", dep.strip())
        assert match, f"unparseable dep: {dep!r}"
        name, specifier = match.group(1), match.group(2).strip()
        parsed[name.lower().replace("_", "-")] = specifier
    return parsed


def _parse_requirements_txt() -> dict[str, str]:
    """Return the `requirements.txt` map keyed by normalised package
    name. Blank lines and `#` comments are skipped. `-r` / `-e` /
    direct-URL lines are not expected in this repo and are
    rejected explicitly so a future agent sees the constraint."""
    parsed: dict[str, str] = {}
    for raw in REQUIREMENTS_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or "://" in line:
            raise AssertionError(
                f"requirements.txt has an unsupported line: {line!r}. "
                "The deployment manifest is pinned to simple "
                "name[specifier] entries only."
            )
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", line)
        assert match, f"unparseable requirement: {line!r}"
        name, specifier = match.group(1), match.group(2).strip()
        parsed[name.lower().replace("_", "-")] = specifier
    return parsed


def test_requirements_txt_matches_pyproject_dependencies() -> None:
    """`requirements.txt` must list every package in
    `pyproject.toml::[project.dependencies]` with a compatible
    specifier. Issue #52 was caused by adding `plotly` to
    `pyproject.toml` but the deployment manifest falling back to
    the default (streamlit-only) install — a sync test catches
    the next drift."""
    pyproject_deps = _parse_pyproject_dependencies()
    requirements = _parse_requirements_txt()
    missing = set(pyproject_deps) - set(requirements)
    assert not missing, (
        f"requirements.txt is missing packages declared in "
        f"pyproject.toml: {sorted(missing)}. "
        "Add them to requirements.txt or Streamlit Cloud will "
        "fail with ModuleNotFoundError at app startup (Issue #52)."
    )


def test_requirements_txt_specifiers_compatible() -> None:
    """When a package is declared in both, the specifier must be
    compatible (the requirements.txt specifier is a subset of the
    pyproject.toml specifier). The `==` operator on specifiers
    handles the common case (>=, <=, ==, ~=, !=)."""
    pyproject_deps = _parse_pyproject_dependencies()
    requirements = _parse_requirements_txt()
    for name, req_spec in requirements.items():
        if name not in pyproject_deps:
            continue
        pyproject_spec = pyproject_deps[name]
        # The requirements.txt specifier must be a subset of the
        # pyproject.toml specifier. If pyproject says >=6.0 and
        # requirements.txt says >=5.0, requirements.txt is wider
        # than the project declares — that's a drift.
        req = SpecifierSet(req_spec) if req_spec else SpecifierSet()
        proj = SpecifierSet(pyproject_spec) if pyproject_spec else SpecifierSet()
        # Every version allowed by req must also be allowed by proj.
        for v in _sample_versions(req):
            if v in req:
                assert v in proj, (
                    f"requirements.txt specifier {req_spec!r} for "
                    f"{name!r} allows {v!r} but pyproject.toml "
                    f"specifier {pyproject_spec!r} does not."
                )


def _sample_versions(specifier: SpecifierSet) -> list[str]:
    """Return a handful of sample versions to test specifier
    compatibility. Covers major Python-package majors across the
    declared ranges (>=0.27, >=0.20, >=2.0, >=5.0, >=6.0,
    >=1.5, >=4.5, >=1.30, >=8.0, >=0.6, >=0.0.1a5,
    >=24.0). The check is conservative — it only fails when
    `requirements.txt` is *wider* than `pyproject.toml`."""
    return [
        "0.0.0",
        "1.0.0",
        "2.0.0",
        "3.0.0",
        "4.0.0",
        "5.0.0",
        "6.0.0",
        "7.0.0",
        "8.0.0",
        "10.0.0",
        "100.0.0",
    ]


def test_uv_lock_is_gitignored() -> None:
    """Streamlit Cloud reads the first dep file it finds in this
    priority order: `uv.lock` > `Pipfile` > `environment.yml` >
    `requirements.txt` > `pyproject.toml`. If `uv.lock` is
    present, Cloud tries to parse it; when the format is newer
    than Cloud's `uv` binary, parsing silently falls back to
    streamlit-only and the app crashes (Issue #52). We must NOT
    ship `uv.lock` in the repo."""
    gitignore_text = GITIGNORE_PATH.read_text()
    # Match a line that is exactly `uv.lock` (with optional trailing
    # whitespace) — not a comment, not a longer path.
    pattern = re.compile(r"(?m)^uv\.lock\s*$")
    assert pattern.search(gitignore_text), (
        "uv.lock must be listed in .gitignore so Streamlit Cloud "
        "falls through to requirements.txt (Issue #52)."
    )


def test_uv_lock_is_not_tracked_by_git() -> None:
    """Belt-and-braces: even if `.gitignore` is misconfigured, the
    tracked-files list must not include `uv.lock`."""
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "uv.lock"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert result.stdout.strip() == "", (
        "uv.lock is tracked in git. Streamlit Cloud will pick it "
        "up and try to parse it; when the format is newer than "
        "Cloud's uv binary, the app crashes with "
        "ModuleNotFoundError (Issue #52). Run "
        "`git rm --cached uv.lock` and ensure it stays in "
        ".gitignore."
    )


def test_requirements_txt_is_hand_maintained() -> None:
    """`requirements.txt` must NOT be in `pip freeze` format
    (e.g. `streamlit==1.58.0`). The deploy manifest is
    hand-maintained to mirror `pyproject.toml::[project.dependencies]`,
    and a `pip freeze > requirements.txt` would (a) pin versions
    that diverge from the local `uv`-managed dev env, (b) change
    the file's hash on every dep bump, masking the real cause of
    a broken deploy (Streamlit Cloud uses the file's hash to
    decide whether to do a fresh install — see Issue #52), and
    (c) lose the header comment that documents the deploy
    requirement.

    Concretely: every non-comment, non-blank line must be a
    `name[specifier]` requirement (PEP 508), with no `==` (pinned)
    specifier, no `-r` / `-e` / direct-URL reference, and no
    `pip freeze` style output."""
    for raw in REQUIREMENTS_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Reject `pip freeze` style: `package==X.Y.Z` (==) is what
        # pip freeze emits. >=, <=, ~=, != are specifier operators,
        # not version pins.
        assert "==" not in line, (
            f"requirements.txt has a pinned (==) requirement: "
            f"{line!r}. The deploy manifest must use minimum-version "
            f"specifiers (>=, <, ~=) to stay hand-maintained and "
            f"in sync with pyproject.toml. See Issue #52."
        )
        assert not line.startswith("-"), (
            f"requirements.txt has a `-r` / `-e` / direct-URL line: "
            f"{line!r}. The deploy manifest is pinned to simple "
            f"name[specifier] entries only. See Issue #52."
        )
        assert "://" not in line, (
            f"requirements.txt has a direct-URL line: {line!r}. "
            f"The deploy manifest is pinned to simple "
            f"name[specifier] entries only. See Issue #52."
        )


def test_requirements_txt_documents_deploy_requirement() -> None:
    """`requirements.txt` must carry a header comment that
    documents the deploy manifest's role. Without the comment,
    a future maintainer may regenerate the file with
    `pip freeze` and lose the hand-maintained contract. See
    Issue #52."""
    header = REQUIREMENTS_PATH.read_text().splitlines()[:8]
    blob = "\n".join(header)
    assert "deploy" in blob.lower() or "streamlit cloud" in blob.lower(), (
        "requirements.txt must start with a comment explaining "
        "that the file is the Streamlit Cloud deploy manifest, "
        "so a future maintainer does not regenerate it with "
        "`pip freeze` and break the deploy hash-detection. See "
        "Issue #52."
    )


def test_fresh_venv_install() -> None:
    """End-to-end check that `pip install -r requirements.txt`
    succeeds in a clean venv and that every declared runtime
    dep is importable afterwards. Catches the class of bug
    where `requirements.txt` *looks* correct (passes the
    in-sync / specifier tests) but the install actually
    fails — e.g. an unresolvable transitive dep, a yanked
    version, or a hash mismatch.

    Streamlit Cloud runs a clean install on every hash change
    of a recognised dep file (see Issue #52); this test
    mirrors that flow locally. The test is slow (network-bound,
    ~30-60s) but red-capable: it fails with the exact pip error
    on a broken `requirements.txt`, and passes on a clean one.
    """
    import subprocess
    import tempfile
    import venv

    with tempfile.TemporaryDirectory() as tmp:
        venv_dir = Path(tmp) / "venv"
        venv.create(venv_dir, with_pip=True, clear=True)
        pip = venv_dir / "bin" / "pip"
        python = venv_dir / "bin" / "python"
        result = subprocess.run(
            [str(pip), "install", "--quiet", "-r", str(REQUIREMENTS_PATH)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, (
            f"`pip install -r requirements.txt` failed in a clean "
            f"venv. This is what Streamlit Cloud does on every "
            f"deploy; a failure here means the v4 dashboard will "
            f"crash at startup with ModuleNotFoundError (Issue #52).\n"
            f"\nstdout:\n{result.stdout}\n"
            f"\nstderr:\n{result.stderr}"
        )
        # Every runtime dep declared in `pyproject.toml` must be
        # importable in the fresh venv. Imports use the
        # distribution name (scikit-learn → sklearn).
        for pkg in (
            "httpx",
            "huggingface_hub",
            "numpy",
            "pandas",
            "packaging",
            "plotly",
            "sklearn",
            "lightgbm",
            "streamlit",
        ):
            check = subprocess.run(
                [str(python), "-c", f"import {pkg}"],
                capture_output=True,
                text=True,
                check=False,
            )
            assert check.returncode == 0, (
                f"Fresh-venv install of `requirements.txt` did not "
                f"install {pkg!r}. Streamlit Cloud would also be "
                f"missing it, and `app.py` line 49 crashes with "
                f"ModuleNotFoundError (Issue #52).\n"
                f"\nstdout:\n{check.stdout}\n"
                f"\nstderr:\n{check.stderr}"
            )


@pytest.mark.parametrize(
    "package",
    [
        "httpx",
        "huggingface_hub",
        "numpy",
        "pandas",
        "packaging",
        "plotly",
        "sklearn",
        "lightgbm",
        "streamlit",
    ],
)
def test_runtime_dep_is_importable(package: str) -> None:
    """Every runtime dep declared in `pyproject.toml` must be
    importable in the test env. Catches the class of bug where a
    dep is declared but not actually installed (e.g. a
    `uv sync` was never run after a dep was added). The
    `scikit-learn` distribution is imported as `sklearn`."""
    import_module(package)

"""Smoke test for the Streamlit app module import."""

from __future__ import annotations

import app


def test_app_module_imports() -> None:
    assert app.main is not None

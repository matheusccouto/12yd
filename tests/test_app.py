"""Tests for the v5 Streamlit app — two-team dropdowns, local file reads."""

from __future__ import annotations

import app


def test_app_module_imports() -> None:
    assert app.main is not None


def test_badge_color_known_team() -> None:
    assert app._badge_color("Brazil") == "yellow"


def test_badge_color_unknown_team() -> None:
    assert app._badge_color("Unknown FC") == "gray"

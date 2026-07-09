"""Tests for the scraper config module."""

from __future__ import annotations

from datetime import date

from twelveyards import config


def test_predict_window_is_a_date() -> None:
    assert isinstance(config.PREDICT_WINDOW_START, date)


def test_history_floor_is_before_predict_window() -> None:
    """The History Floor must be ≤ the Prediction Window start so that
    every target shootout kick has ≥ a few years of pre-window history."""
    assert config.HISTORY_FLOOR < config.PREDICT_WINDOW_START


def test_history_floor_matches_prd() -> None:
    """PRD: current History Floor is 2016-01-01."""
    assert config.HISTORY_FLOOR == date(2016, 1, 1)


def test_lookback_window_is_an_int() -> None:
    assert isinstance(config.LOOKBACK_WINDOW_YEARS, int)
    assert config.LOOKBACK_WINDOW_YEARS >= 1


def test_user_agent_is_present() -> None:
    """docs/fotmob.md: CloudFront returns 403 without a desktop UA."""
    assert config.USER_AGENT
    assert "Mozilla" in config.USER_AGENT


def test_today_utc_returns_a_date() -> None:
    assert isinstance(config.today_utc(), date)

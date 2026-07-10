"""Tests for the scraper config module."""

from __future__ import annotations

from datetime import date

from twelveyards import config


def test_scrape_floor_is_a_date() -> None:
    assert isinstance(config.SCRAPE_FLOOR, date)


def test_train_floor_is_a_date() -> None:
    assert isinstance(config.TRAIN_FLOOR, date)


def test_scrape_floor_is_before_train_floor() -> None:
    assert config.SCRAPE_FLOOR < config.TRAIN_FLOOR


def test_scrape_floor_matches_prd() -> None:
    assert date(2016, 1, 1) == config.SCRAPE_FLOOR


def test_train_floor_matches_prd() -> None:
    assert date(2021, 1, 1) == config.TRAIN_FLOOR


def test_lookback_window_is_an_int() -> None:
    assert isinstance(config.LOOKBACK_WINDOW_YEARS, int)
    assert config.LOOKBACK_WINDOW_YEARS >= 1


def test_user_agent_is_present() -> None:
    """docs/fotmob.md: CloudFront returns 403 without a desktop UA."""
    assert config.USER_AGENT
    assert "Mozilla" in config.USER_AGENT


def test_today_utc_returns_a_date() -> None:
    assert isinstance(config.today_utc(), date)

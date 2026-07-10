"""Tests for the TabPFN classifier wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from twelveyards import tabpfn as tabpfn_module


class _FakeTabPFNClassifier:
    """Returns a fixed 3-class probability distribution for any input."""

    def __init__(self, **_kwargs: object) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.full((n, 3), 1.0 / 3.0)


@pytest.fixture(autouse=True)
def _stub_tabpfn_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tabpfn_module, "TabPFNClassifier", _FakeTabPFNClassifier,
    )
    monkeypatch.setattr(tabpfn_module, "_tabpfn_init", lambda: None)


def test_init_does_not_raise() -> None:
    tabpfn_module.init()


def test_fit_and_predict_proba_returns_expected_shape() -> None:
    model = tabpfn_module.TabPFN()
    X_train = np.random.default_rng(42).random((50, 7))
    y_train = np.random.default_rng(42).integers(0, 3, size=50)
    model.fit(X_train, y_train)
    X_test = np.random.default_rng(43).random((20, 7))
    probs = model.predict_proba(X_test)
    assert probs.shape == (20, 3)


def test_predict_proba_rows_sum_to_one() -> None:
    model = tabpfn_module.TabPFN()
    X = np.random.default_rng(42).random((10, 7))
    y = np.random.default_rng(42).integers(0, 3, size=10)
    model.fit(X, y)
    probs = model.predict_proba(X)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_predict_proba_deterministic_with_fixed_random_state() -> None:
    rng = np.random.default_rng(42)
    X = rng.random((10, 7))
    y = rng.integers(0, 3, size=10)

    model_a = tabpfn_module.TabPFN(random_state=0)
    model_a.fit(X, y)
    probs_a = model_a.predict_proba(X)

    model_b = tabpfn_module.TabPFN(random_state=0)
    model_b.fit(X, y)
    probs_b = model_b.predict_proba(X)

    np.testing.assert_array_almost_equal(probs_a, probs_b)


def test_predict_proba_before_fit_raises() -> None:
    model = tabpfn_module.TabPFN()
    X = np.random.default_rng(42).random((5, 7))
    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict_proba(X)


def test_single_row_predict_proba() -> None:
    model = tabpfn_module.TabPFN()
    X = np.random.default_rng(42).random((5, 7))
    y = np.random.default_rng(42).integers(0, 3, size=5)
    model.fit(X, y)
    X_single = np.random.default_rng(43).random((1, 7))
    probs = model.predict_proba(X_single)
    assert probs.shape == (1, 3)
    np.testing.assert_allclose(probs[0].sum(), 1.0, atol=1e-6)


def test_default_parameters_are_propagated() -> None:
    model = tabpfn_module.TabPFN()
    assert model._n_estimators == 8
    assert model._thinking_mode is False
    assert model._random_state == 0


def test_custom_parameters_are_kept() -> None:
    model = tabpfn_module.TabPFN(
        n_estimators=16,
        thinking_mode=False,
        random_state=99,
        categorical_features_indices=[0, 1, 5],
    )
    assert model._n_estimators == 16
    assert model._random_state == 99
    assert model._categorical_features_indices == [0, 1, 5]

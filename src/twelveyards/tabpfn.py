"""TabPFN classifier wrapper for penalty shootout prediction.

PRD-v5: TabPFNClassifier in cheapest mode (thinking_mode=False,
n_estimators=8). Always batch all roster rows in one predict_proba
call. Authentication via TABPFN_TOKEN env var.
"""

from __future__ import annotations

import numpy as np
from tabpfn_client import TabPFNClassifier
from tabpfn_client import init as _tabpfn_init


def init() -> None:
    """Initialise the TabPFN client (reads TABPFN_TOKEN env var)."""
    _tabpfn_init()


class TabPFN:
    """Thin wrapper over TabPFNClassifier for penalty side prediction.

    Usage:
        init()
        model = TabPFN()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)  # shape (n_test, 3)
    """

    def __init__(
        self,
        *,
        n_estimators: int = 8,
        thinking_mode: bool = False,
        random_state: int = 0,
        categorical_features_indices: list[int] | None = None,
    ) -> None:
        self._n_estimators = n_estimators
        self._thinking_mode = thinking_mode
        self._random_state = random_state
        self._categorical_features_indices = categorical_features_indices
        self._classifier: TabPFNClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit TabPFN on the training data (free — no tokens consumed)."""
        self._classifier = TabPFNClassifier(
            n_estimators=self._n_estimators,
            thinking_mode=self._thinking_mode,
            random_state=self._random_state,
            categorical_features_indices=self._categorical_features_indices,
            ignore_pretraining_limits=True,
        )
        self._classifier.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Run batched prediction (costs tokens — always batch all rows)."""
        if self._classifier is None:
            msg = "TabPFN must be fit before predict_proba"
            raise RuntimeError(msg)
        return np.asarray(self._classifier.predict_proba(X))

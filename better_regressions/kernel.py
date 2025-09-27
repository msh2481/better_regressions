from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.kernel_approximation import Nystroem
from sklearn.utils.validation import check_is_fitted


class SupervisedNystroem(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        forest_kind: Literal[
            "rf_classifier",
            "rf_regressor",
            "et_classifier",
            "et_regressor",
        ] = "rf_classifier",
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int | float = 1,
        n_jobs: int | None = None,
        random_state: int | None = None,
        n_components: int | None = None,
    ) -> None:
        self.forest_kind = forest_kind
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_components = n_components

    def fit(self, X: ArrayLike, y: ArrayLike) -> SupervisedNystroem:
        forest_cls = self._resolve_forest()
        self.forest_ = forest_cls(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.forest_.fit(X, y)
        n_components = self.n_components if self.n_components is not None else np.shape(X)[0]
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if n_components > np.shape(X)[0]:
            raise ValueError("n_components cannot exceed number of samples")
        random_state = self.random_state
        self._nystroem = Nystroem(
            kernel=self._leaf_kernel,
            n_components=n_components,
            random_state=random_state,
        )
        self._nystroem.fit(X)
        self.n_features_in_ = np.shape(X)[1]
        self.n_components_ = n_components
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "_nystroem")
        return self._nystroem.transform(X)

    def fit_transform(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _resolve_forest(self):
        if self.forest_kind == "rf_classifier":
            return RandomForestClassifier
        if self.forest_kind == "rf_regressor":
            return RandomForestRegressor
        if self.forest_kind == "et_classifier":
            return ExtraTreesClassifier
        if self.forest_kind == "et_regressor":
            return ExtraTreesRegressor
        raise ValueError(f"Unknown forest_kind: {self.forest_kind}")

    def _leaf_kernel(self, X: ArrayLike, Y: ArrayLike | None = None) -> np.ndarray:
        check_is_fitted(self, "forest_")
        leaves_X = self.forest_.apply(X)
        leaves_Y = leaves_X if Y is None else self.forest_.apply(Y)
        matches = leaves_X[:, None, :] == leaves_Y[None, :, :]
        return matches.mean(axis=2)

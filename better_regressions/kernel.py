from __future__ import annotations

from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


class SupervisedNystroem(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        forest_kind: Literal["rf", "et"] = "et",
        regression: bool = True,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int | float = 0.1,
        n_jobs: int | None = None,
        random_state: int | None = None,
        n_components: int | None = None,
    ) -> None:
        self.forest_kind = forest_kind
        self.regression = regression
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
        logger.info(
            f"Fitting forest with {self.n_estimators} estimators, X.shape={X.shape}, y.shape={y.shape}",
        )
        self.forest_.fit(X, y)
        logger.info(f"Forest fitted")
        X_arr = np.asarray(X)
        leaves = self.forest_.apply(X_arr)
        logger.info(f"Forest leaves computed with shape={leaves.shape}")
        n_samples = leaves.shape[0]
        n_components = self.n_components if self.n_components is not None else n_samples
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if n_components > n_samples:
            n_components = n_samples
            logger.warning(f"n_components={n_components} > n_samples={n_samples}, setting n_components to n_samples")
        rng = check_random_state(self.random_state)
        component_indices = np.arange(n_samples) if n_components == n_samples else rng.choice(n_samples, size=n_components, replace=False)
        logger.info(f"Selected {n_components} reference components")
        basis_leaves = leaves[component_indices]
        basis_kernel = self._kernel_from_leaves(basis_leaves, basis_leaves)
        eigvals, eigvecs = np.linalg.eigh(basis_kernel)
        eigvals = np.maximum(eigvals, 1e-12)
        inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        inv_basis = eigvecs @ inv_sqrt @ eigvecs.T
        kernel_nm = self._kernel_from_leaves(leaves, basis_leaves)
        training_embedding = kernel_nm @ inv_basis
        self.component_indices_ = component_indices
        self.basis_kernel_ = basis_kernel
        self.basis_leaves_ = basis_leaves
        self.normalization_ = inv_basis
        self.training_embedding_ = training_embedding
        self.components_ = X_arr[component_indices]
        logger.info("Normalization matrix computed")
        self.n_features_in_ = self.forest_.n_features_in_
        self.n_components_ = n_components
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "normalization_")
        X_arr = np.asarray(X)
        X_vector = X_arr.ndim == 1
        if X_vector:
            X_arr = X_arr.reshape(1, -1)
        leaves = self.forest_.apply(X_arr)
        kernel = self._kernel_from_leaves(leaves, self.basis_leaves_)
        embedding = kernel @ self.normalization_
        return embedding[0] if X_vector else embedding

    def fit_transform(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        self.fit(X, y)
        return self.training_embedding_

    def _resolve_forest(self):
        if self.forest_kind == "rf":
            return RandomForestRegressor if self.regression else RandomForestClassifier
        if self.forest_kind == "et":
            return ExtraTreesRegressor if self.regression else ExtraTreesClassifier
        raise ValueError(f"Unknown forest_kind: {self.forest_kind}")

    def _leaf_kernel(self, X: ArrayLike, Y: ArrayLike | None = None) -> np.ndarray:
        check_is_fitted(self, "forest_")
        logger.info(f"Computing leaf kernel for X={X} and Y={Y}")
        X_arr = np.asarray(X)
        X_vector = X_arr.ndim == 1
        if X_vector:
            X_arr = X_arr.reshape(1, -1)
        if Y is None:
            Y_arr = X_arr
            Y_vector = X_vector
        else:
            Y_arr = np.asarray(Y)
            Y_vector = Y_arr.ndim == 1
            if Y_vector:
                Y_arr = Y_arr.reshape(1, -1)
        leaves_X = self.forest_.apply(X_arr)
        leaves_Y = leaves_X if Y is None else self.forest_.apply(Y_arr)
        mean_matches = (leaves_X[:, None, :] == leaves_Y[None, :, :]).mean(axis=2)
        if X_vector and Y_vector:
            return float(mean_matches[0, 0])
        if X_vector:
            return mean_matches[0]
        if Y_vector:
            return mean_matches[:, 0]
        return mean_matches

    def _kernel_from_leaves(self, leaves_X: np.ndarray, leaves_Y: np.ndarray) -> np.ndarray:
        logger.info(f"Computing kernel from leaves with shapes {leaves_X.shape} and {leaves_Y.shape}")
        return (leaves_X[:, None, :] == leaves_Y[None, :, :]).mean(axis=2)

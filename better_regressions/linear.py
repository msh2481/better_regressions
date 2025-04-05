"""Linear regression models with enhanced functionality."""

import numpy as np
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float
from loguru import logger
from numpy import ndarray as ND
from scipy import linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge

from better_regressions.utils import format_array


@typed
class Linear(RegressorMixin, BaseEstimator):
    """Linear regression with configurable regularization and bias handling.

    Args:
        alpha: If float, Ridge's alpha parameter. If "ard", use ARDRegression
        better_bias: If True, include ones column as feature and don't fit intercept
        method: Regression method ("standard", "pcr", "lora")
        n_components: Number of components to use for PCR/LoRA methods
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        alpha: int | float | Literal["ard", "bayes"] = "bayes",
        better_bias: bool = True,
        method: Literal["standard", "pcr", "lora"] = "standard",
        n_components: int | None = None,
        random_state: int | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.better_bias = better_bias
        self.method = method
        self.n_components = n_components
        self.random_state = random_state

    @typed
    def __repr__(self, var_name: str = "model") -> str:
        if not hasattr(self, "coef_"):
            return f"{var_name} = Linear(alpha={repr(self.alpha)}, better_bias={self.better_bias}, method='{self.method}', n_components={self.n_components})"

        model_init = f"{var_name} = Linear(alpha={repr(self.alpha)}, better_bias={self.better_bias}, method='{self.method}', n_components={self.n_components})"
        set_coef = f"{var_name}.coef_ = {format_array(self.coef_)}"
        set_intercept = f"{var_name}.intercept_ = {format_array(self.intercept_)}"

        return "\n".join([model_init, set_coef, set_intercept])

    @typed
    def _fit_standard(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> tuple[Float[ND, "n_features"], float]:
        """Standard ridge regression fit."""
        X_fit = X.copy()

        if self.alpha == "ard":
            model = ARDRegression(fit_intercept=not self.better_bias)
        elif self.alpha == "bayes":
            model = BayesianRidge(fit_intercept=not self.better_bias)
        else:
            model = Ridge(alpha=self.alpha, fit_intercept=not self.better_bias)

        if self.better_bias:
            # Add column of ones to apply regularization to bias
            X_fit = np.hstack([np.ones((X.shape[0], 1)), X_fit])

        model.fit(X_fit, y)

        if self.better_bias:
            coef = model.coef_[1:]
            intercept = model.coef_[0]
        else:
            coef = model.coef_
            intercept = model.intercept_

        if isinstance(self.alpha, str) and self.alpha.lower() == "ard":
            self.lambda_ = model.lambda_

        return coef, intercept

    @typed
    def _fit_pcr(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> tuple[Float[ND, "n_features"], float]:
        """Principal Component Regression (PCR) fit."""
        n_samples, n_features = X.shape

        # Determine number of components to use
        k = min(self.n_components, n_samples, n_features)
        if k is None:
            k = min(n_samples, n_features, 10)  # Default to 10 components or fewer if data is small

        pca = PCA(n_components=k, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        if self.alpha == "ard":
            model = ARDRegression(fit_intercept=not self.better_bias)
        elif self.alpha == "bayes":
            model = BayesianRidge(fit_intercept=not self.better_bias)
        else:
            model = Ridge(alpha=self.alpha, fit_intercept=not self.better_bias)

        if self.better_bias:
            # Add column of ones to apply regularization to bias
            X_pca = np.hstack([np.ones((X.shape[0], 1)), X_pca])

        model.fit(X_pca, y)

        if self.better_bias:
            coef = model.coef_[1:]
            intercept = model.coef_[0]
        else:
            coef = model.coef_
            intercept = model.intercept_

        coef = pca.components_.T @ coef
        return coef, intercept

    @typed
    def _fit_lora(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> tuple[Float[ND, "n_features"], float]:
        """Low Rank Approximation (LoRA) fit."""
        n_samples, n_features = X.shape

        # Determine number of components to use
        k = min(self.n_components, n_samples, n_features)
        if k is None:
            k = min(n_samples, n_features, 10)  # Default to 10 components or fewer if data is small

        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        y_mean = y.mean()
        y_centered = y - y_mean

        # Compute correlation matrix X^T X
        corr_matrix = X_centered.T @ X_centered

        # Compute its low-rank approximation using SVD
        U, s, Vh = linalg.svd(corr_matrix, full_matrices=False)

        # Keep only top-k singular values/vectors
        U_k = U[:, :k]
        s_k = s[:k]
        V_k = Vh[:k, :]

        # Low-rank approximation of correlation matrix
        corr_matrix_lora = (U_k * s_k) @ V_k
        # Ensure that diagonal matches exactly
        corr_matrix_lora += np.diag(np.diag(corr_matrix)) - np.diag(np.diag(corr_matrix_lora))

        # Add regularization
        if isinstance(self.alpha, str):
            raise ValueError("LoRA does not support Bayesian methods")
        else:
            alpha_value = self.alpha

        reg_matrix = corr_matrix_lora + alpha_value * np.eye(n_features)

        # Solve the system: coef = (lora(X^T X) + alpha*I)^-1 X^T y
        coef = linalg.solve(reg_matrix, X_centered.T @ y_centered)

        if not self.better_bias:
            intercept = y_mean - X_mean @ coef
        else:
            intercept = 0.0

        return coef, intercept

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> "Linear":
        """Fit the linear model using the specified method."""
        if self.method == "standard":
            self.coef_, self.intercept_ = self._fit_standard(X, y)
        elif self.method == "pcr":
            self.coef_, self.intercept_ = self._fit_pcr(X, y)
        elif self.method == "lora":
            self.coef_, self.intercept_ = self._fit_lora(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        """Predict using the linear model."""
        return X @ self.coef_ + self.intercept_

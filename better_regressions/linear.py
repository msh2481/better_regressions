"""Linear regression models with enhanced functionality."""

import numpy as np
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float
from numpy import ndarray as ND
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge

from better_regressions.utils import format_array


@typed
class Linear(RegressorMixin, BaseEstimator):
    """Linear regression with configurable regularization and bias handling.
    Args:
        alpha: If float, Ridge's alpha parameter. If "ard", use ARDRegression
        better_bias: If True, include ones column as feature and don't fit intercept
    """

    def __init__(self, alpha: int | float | Literal["ard", "bayes"] = "bayes", better_bias: bool = True):
        super().__init__()
        self.alpha = alpha
        self.better_bias = better_bias

    @typed
    def __repr__(self, var_name: str = "model") -> str:
        if not hasattr(self, "coef_"):
            return f"{var_name} = Linear(alpha={repr(self.alpha)}, better_bias={self.better_bias})"

        model_init = f"{var_name} = Linear(alpha={repr(self.alpha)}, better_bias={self.better_bias})"
        set_coef = f"{var_name}.coef_ = {format_array(self.coef_)}"
        set_intercept = f"{var_name}.intercept_ = {format_array(self.intercept_)}"

        return "\n".join([model_init, set_coef, set_intercept])

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> "Linear":
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
            self.coef_ = model.coef_[1:]
            self.intercept_ = model.coef_[0]
        else:
            self.coef_ = model.coef_
            self.intercept_ = model.intercept_
        if isinstance(self.alpha, str) and self.alpha.lower() == "ard":
            self.lambda_ = model.lambda_

        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        return X @ self.coef_ + self.intercept_

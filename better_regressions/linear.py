"""Linear regression models with enhanced functionality."""

import numpy as np
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float
from loguru import logger
from numpy import ndarray as ND
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.datasets import make_regression
from sklearn.linear_model import ARDRegression, BayesianRidge, LogisticRegression, Ridge

from better_regressions.classifier import AutoClassifier
from better_regressions.utils import format_array


def _repr_logistic_regression(estimator: LogisticRegression, var_name: str) -> str:
    """Generate reconstruction code for a LogisticRegression instance."""
    lines = []
    lines.append(f"{var_name} = LogisticRegression()")
    lines.append(f"{var_name}.coef_ = {format_array(estimator.coef_)}")
    lines.append(f"{var_name}.intercept_ = {format_array(estimator.intercept_)}")
    # Reconstruct classes_ based on coef_ shape, assuming default integer `classes_`
    lines.append(f"{var_name}.classes_ = {format_array(estimator.classes_)}")
    return "\n".join(lines)


@typed
class Linear(RegressorMixin, BaseEstimator):
    """Linear regression with configurable regularization and bias handling.

    Args:
        alpha: If float, Ridge's alpha parameter. If "ard", use ARDRegression
        better_bias: If True, include ones column as feature and don't fit intercept
    """

    def __init__(
        self,
        alpha: int | float | Literal["ard", "bayes"] = "bayes",
        better_bias: bool = True,
    ):
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
        self.coef_ = coef
        self.intercept_ = intercept
        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        """Predict using the linear model."""
        return X @ self.coef_ + self.intercept_


class Soft(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        splits: list[float],  # quantiles to split the data into classes
        estimator: BaseEstimator,
        depth: int | None | Literal["auto"] = "auto",
    ):
        super().__init__()
        self.splits = splits
        # check that splits (quantiles) are in [0, 1]
        assert all(0 <= q <= 1 for q in splits)
        self.estimator = estimator
        self.depth = depth

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> "Soft":
        self.classifier_ = AutoClassifier(depth=self.depth)
        self.estimators_ = []
        classes = np.zeros(len(y), dtype=np.int32)
        y_argsort = y.argsort()
        for split_quantile in self.splits:
            indices = y_argsort[: int(split_quantile * len(y))]
            classes[indices] += 1
        self.classifier_.fit(X, classes)
        pred = self.classifier_.predict_proba(X)
        for cls in range(len(self.splits) + 1):
            weights = pred[:, cls]
            # take each index int(4 * weight[i]) times
            indices = np.repeat(np.arange(len(X)), (4 * weights).astype(np.int32))
            X_subset = X[indices]
            y_subset = y[indices]
            estimator = clone(self.estimator)
            estimator.fit(X_subset, y_subset)
            self.estimators_.append(estimator)
        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        pred = self.classifier_.predict_proba(X)
        total = np.zeros(len(X))
        for cls in range(len(self.splits) + 1):
            weights = pred[:, cls]
            total += weights * self.estimators_[cls].predict(X)
        return total

    def __repr__(self, var_name: str = "model") -> str:
        if not hasattr(self, "classifier_"):
            return f"{var_name} = Soft(splits={self.splits}, estimator=None)"
        lines = []

        # Reconstruct the main Soft object first
        lines.append(f"{var_name} = Soft(splits={self.splits}, estimator=None)")

        # Reconstruct AutoClassifier
        clf_name = f"{var_name}_clf"
        lines.append(self.classifier_.__repr__(var_name=clf_name))
        lines.append(f"{var_name}.classifier_ = {clf_name}")
        lines.append("")  # Add a newline for readability

        # Reconstruct estimators
        lines.append(f"{var_name}.estimators_ = []")
        for i, estimator in enumerate(self.estimators_):
            est_name = f"{var_name}_est{i}"
            lines.append(estimator.__repr__(var_name=est_name))
            lines.append(f"{var_name}.estimators_.append({est_name})")
            lines.append("")

        return "\n".join(lines)

"""Linear regression models with enhanced functionality."""

import numpy as np
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float
from loguru import logger
from numpy import ndarray as ND
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ARDRegression, BayesianRidge, LogisticRegression, Ridge

from better_regressions.classifier import AutoClassifier
from better_regressions.scaling import Scaler
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


"""
$ X^T X = n I,quad y^T y = 1 $
$S = "diag"(s_1, dots, s_n)$ -- scaling factors

$ y = X S S^-1 beta + epsilon, epsilon ~ cal(N)(0, sigma^2) $

$ hat(beta) = arg min (1/2 norm(y - X S S^-1 hat(beta))^2 + 1/2 norm(S^-1 hat(beta))^2) $
$ [ beta' = S^-1 hat(beta), quad X' = X S ] $
$ beta' = arg min (1/2 norm(y - X' beta')^2 + 1/2 norm(beta')^2) $ 
$ beta' = (X'^T X' + I)^(-1)X'^T y $
$ hat(beta) = S(S X^T X S + I)^(-1) S X^T y $
$ hat(beta) = (X^T X + S^(-2))^(-1) X^T y $
$ hat(beta) = (X^T X + S^(-2))^(-1) X^T X beta + (X^T X + S^(-2))^(-1) X^T epsilon $
$ EE[hat(beta)] = (I + 1/n S^(-2))^(-1) beta $
$ "Cov"[hat(beta)] = 1/n sigma^2 (I + 1/n S^(-2))^(-2) $

$ lambda = (1 + 1/(n s^2))^(-1) $
$ "Bias"^2 = (1 - lambda)^2 beta^2 $
$ "Var" = lambda^2 sigma^2 / n $
$ (d cal(L)) / (d lambda) = (lambda - 1) beta^2 + lambda sigma^2 / n = 0 $
$ lambda (beta^2 + sigma^2 / n) = beta^2 $
$ lambda = beta^2 / (beta^2 + sigma^2 / n) $
$ s^2 = 1 / (n(lambda^(-1) - 1)) = beta^2 / sigma^2 $
$ s = beta / sigma $
"""


class Shrinker(BaseEstimator, RegressorMixin):
    def __init__(self, alpha: Literal["bayes", "ard"] | float = "bayes", fit_intercept: bool = True):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> "Shrinker":
        n, d = X.shape
        if np.abs(X.T @ X - n * np.eye(d)).max() > 1e-6:
            logger.error("X^T X should be n * I")
        if np.abs((y**2).mean() - d) > 1e-6:
            logger.error(f"mean y**2 should be d={d}, instead got {(y**2).mean()}")

        if self.fit_intercept:
            if np.abs(X.mean(axis=0)).max() > 1e-6:
                logger.error("X should be centered")
            X = np.hstack([X, np.ones((n, 1))])
            d += 1
            if np.abs(X.T @ X - n * np.eye(d)).max() > 1e-6:
                logger.error("X^T X should be n * I with intercept")

        beta_ols = np.linalg.solve(X.T @ X + 1e-9 * np.eye(d), X.T @ y)
        sigma = (y - X @ beta_ols).std() + 1e-10
        scale = np.abs(beta_ols / sigma)[None, :]
        X_scaled = X * scale

        if self.alpha == "bayes":
            solver = BayesianRidge(fit_intercept=False)
            solver.fit(X_scaled, y)
            coef = solver.coef_.ravel() * scale.ravel()
        elif self.alpha == "ard":
            solver = ARDRegression(fit_intercept=False)
            solver.fit(X_scaled, y)
            coef = solver.coef_.ravel() * scale.ravel()
        else:
            coef = np.linalg.solve(X_scaled.T @ X_scaled + self.alpha * np.eye(d), X_scaled.T @ y)
            coef = coef.ravel() * scale.ravel()

        if self.fit_intercept:
            self.coef_ = coef[:-1]
            self.intercept_ = coef[-1]
        else:
            self.coef_ = coef
            self.intercept_ = 0
        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        return X @ self.coef_ + self.intercept_


class AdaptiveLinear(RegressorMixin, BaseEstimator):
    """Ridge regression with adaptive shrinkage."""

    @typed
    def __init__(
        self,
        method: Literal["pls", "pca"] = "pls",
        alpha: Literal["bayes", "ard"] | float = 1.0,
        fit_intercept: bool = True,
    ):
        super().__init__()
        if not isinstance(alpha, float) or (abs(alpha - 1) > 1e-6):
            logger.warning(f"alpha=1 is theoretically optimal, but you are using alpha={alpha}")
        self.method = method
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> "AdaptiveLinear":
        n_samples, n_features = X.shape
        n_components = min(n_samples - 1, n_features)

        ortho = PCA(n_components=n_components) if self.method == "pca" else PLSRegression(n_components=n_components)
        ortho.fit(X, y)
        X_ortho = ortho.transform(X)
        if self.fit_intercept:
            X_ortho -= X_ortho.mean(axis=0, keepdims=True)

        solver = Scaler(
            Shrinker(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
            ),
            x_method="standard",
            y_method="standard",
            use_feature_variance=True,
        )
        solver.fit(X_ortho, y)
        # I don't want to properly combine them all analytically, so just one more linear fit
        inputs = np.eye(n_features)
        inputs = np.vstack([inputs, np.zeros((1, n_features))])
        outputs = solver.predict(ortho.transform(inputs))
        # Now just fit linear regression to this input-output mapping
        helper_ridge = Ridge(alpha=1e-18)
        helper_ridge.fit(inputs, outputs)
        self.coef_ = helper_ridge.coef_
        self.intercept_ = helper_ridge.intercept_
        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        return X @ self.coef_ + self.intercept_

    def __repr__(self, var_name: str = "model") -> str:
        if not hasattr(self, "coef_"):
            return f"{var_name} = AdaptiveRidge(n_components={self.n_components})"

        lines = [f"{var_name} = AdaptiveRidge(n_components={self.n_components})", f"{var_name}.coef_ = {format_array(self.coef_)}", f"{var_name}.intercept_ = {format_array(self.intercept_)}"]
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


def test_adaptive():
    from matplotlib import pyplot as plt

    # x = np.linspace(0, 1, 4)[:, None]
    # y = x.ravel() + 2 + 0.9 * np.random.randn(len(x))
    # model = AdaptiveLinear(method="pca", alpha=1.0)
    # model.fit(x, y)
    # print(model.coef_)
    # print(model.intercept_)
    # p = model.predict(x)
    # print(p)
    # plt.plot(x, y)
    # plt.plot(x, p)
    # plt.show()

    n, d = 100, 4
    X = np.random.randn(n, d)
    beta = np.random.randn(d)
    y = X @ beta + np.random.randn(n)
    model = AdaptiveLinear(method="pca", alpha=1.0)
    model.fit(X, y)
    print(model.coef_)
    print(model.intercept_)
    print(beta)


if __name__ == "__main__":
    test_adaptive()

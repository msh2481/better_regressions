import numpy as np
from beartype import beartype as typed
from jaxtyping import Float
from loguru import logger
from numpy import ndarray as ND
from sklearn.base import BaseEstimator, RegressorMixin

from better_regressions.linear import Linear
from better_regressions.scaling import Scaler
from better_regressions.utils import format_array


@typed
class Angle(RegressorMixin, BaseEstimator):
    """Piecewise-linear regression with randomly selected breakpoints.

    Args:
        n_breakpoints: Number of breakpoints to use
        alpha: Regularization parameter for the linear model
        random_state: Random seed for reproducibility
        quantile_range: Range of quantiles to consider for breakpoints
    """

    def __init__(self, n_breakpoints: int = 10, alpha: float = 1e-6, random_state: int | None = None, quantile_range: tuple[float, float] = (0.2, 0.8)):
        super().__init__()
        self.n_breakpoints = n_breakpoints
        self.alpha = alpha
        self.random_state = random_state
        self.quantile_range = quantile_range

    @typed
    def _create_features(self, X: Float[ND, "n_samples"]) -> Float[ND, "n_samples n_angles"]:
        n_samples = len(X)
        n_angles = len(self.breakpoints_) + 2
        features = np.zeros((n_samples, n_angles))
        breakpoints = self.breakpoints_
        # First feature: relu(x_0 - x)
        features[:, 0] = np.maximum(0, breakpoints[0] - X)
        # Middle features: clip(x - x_i, 0, x_{i+1} - x_i)
        for i in range(len(breakpoints) - 1):
            features[:, i + 1] = np.clip(X - breakpoints[i], 0, breakpoints[i + 1] - breakpoints[i])
        # Last feature: relu(x - x_{m-1})
        features[:, -1] = np.maximum(0, X - breakpoints[-1])
        return features

    @typed
    def __repr__(self, var_name: str = "model") -> str:
        if not hasattr(self, "linear_models_"):
            return f"{var_name} = Angle(n_breakpoints={self.n_breakpoints}, alpha={self.alpha}, random_state={self.random_state})"

        model_init = f"{var_name} = Angle(n_breakpoints={self.n_breakpoints}, alpha={self.alpha}, random_state={self.random_state})"
        breakpoints_repr = f"{var_name}.breakpoints_ = {format_array(self.breakpoints_)}"

        return "\n".join([model_init, breakpoints_repr])

    @typed
    def fit(self, X: Float[ND, "n_samples 1"] | Float[ND, "n_samples"], y: Float[ND, "n_samples"]) -> "Angle":
        if X.ndim == 2:
            X = X.ravel()
        # if self.random_state is not None:
        #     np.random.seed(self.random_state)

        n_samples = len(X)

        # Initialize transformed feature matrix
        X_transformed = np.zeros((n_samples, 0))

        x_min, x_max = np.quantile(X, self.quantile_range)
        x_range = x_max - x_min
        # Generate random breakpoints
        propto = np.random.rand(self.n_breakpoints + 1) * 10 + 1
        propto /= propto.sum()
        self.breakpoints_ = np.sort(x_min + np.cumsum(propto[:-1]) * x_range)
        assert len(self.breakpoints_) == self.n_breakpoints
        # Transform feature and add to transformed matrix
        feature_transform = self._create_features(X)
        X_transformed = np.hstack([X_transformed, feature_transform])
        # Fit linear model on transformed features
        self.linear_model_ = Scaler(Linear(alpha=self.alpha))
        self.linear_model_.fit(X_transformed, y)

        return self

    @typed
    def predict(self, X: Float[ND, "n_samples 1"] | Float[ND, "n_samples"]) -> Float[ND, "n_samples"]:
        if X.ndim == 2:
            X = X.ravel()
        X_transformed = self._create_features(X)
        return self.linear_model_.predict(X_transformed)

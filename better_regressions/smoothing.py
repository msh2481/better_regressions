import numpy as np
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float
from numpy import ndarray as ND
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.model_selection import train_test_split
from supersmoother import SuperSmoother

from better_regressions.piecewise import Angle
from better_regressions.utils import format_array


@typed
class Smooth(RegressorMixin, BaseEstimator):
    """Boosting-based regression using smooth functions for features.

    Args:
        method: Smoothing method ("supersmoother" or "angle")
        lr: Learning rate for boosting
        max_epochs: Maximum number of boosting rounds
        n_points: Number of points to store for each feature's smoother
        use_early_stopping: Whether to use early stopping on validation split
        val_size: Validation split size when using early stopping
        patience: Number of rounds without improvement before early stopping
        n_breakpoints: Number of breakpoints to use when method is "angle"
    """

    def __init__(
        self,
        method: Literal["supersmoother", "angle"] = "angle",
        lr: float = 0.5,
        max_epochs: int = 100,
        n_points: int = 50,
        use_early_stopping: bool = False,
        val_size: float = 0.2,
        patience: int = 2,
        n_breakpoints: int = 1,
    ):
        super().__init__()
        self.method = method
        self.lr = lr
        self.max_epochs = max_epochs
        self.n_points = n_points
        self.use_early_stopping = use_early_stopping
        self.val_size = val_size
        self.patience = patience
        self.n_breakpoints = n_breakpoints

    @typed
    def _get_smoother(self) -> object:
        """Get smoother based on method name."""
        if self.method == "supersmoother":
            return SuperSmoother(primary_spans=(0.2, 0.3, 0.5))
        elif self.method == "angle":
            return Angle(n_breakpoints=self.n_breakpoints)
        else:
            raise ValueError(f"Unsupported smoothing method: {self.method}")

    @typed
    def __repr__(self, var_name: str = "model") -> str:
        if not hasattr(self, "feature_points_") or not hasattr(self, "feature_values_"):
            repr_str = f"{var_name} = Smooth(method={repr(self.method)}, lr={self.lr}, max_epochs={self.max_epochs}, n_points={self.n_points}, use_early_stopping={self.use_early_stopping}"
            if self.method == "angle":
                repr_str += f", n_breakpoints={self.n_breakpoints}"
            repr_str += ")"
            return repr_str

        model_init = f"{var_name} = Smooth(method={repr(self.method)}, lr={self.lr}, max_epochs={self.max_epochs}, n_points={self.n_points}, use_early_stopping={self.use_early_stopping}"
        if self.method == "angle":
            model_init += f", n_breakpoints={self.n_breakpoints}"
        model_init += ")"

        points_repr = f"{var_name}.feature_points_ = {format_array(self.feature_points_)}"
        values_repr = f"{var_name}.feature_values_ = {format_array(self.feature_values_)}"

        return "\n".join([model_init, points_repr, values_repr])

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> "Smooth":
        n_samples, n_features = X.shape

        # Initialize storage for feature points and values
        self.feature_points_ = np.zeros((n_features, self.n_points))
        self.feature_values_ = np.zeros((n_features, self.n_points))

        # Generate uniform grid points for each feature
        for j in range(n_features):
            feature_min = np.min(X[:, j])
            feature_max = np.max(X[:, j])
            # extend by 20% on each side
            d = feature_max - feature_min
            feature_min -= 0.2 * d
            feature_max += 0.2 * d
            self.feature_points_[j] = np.linspace(feature_min, feature_max, self.n_points, endpoint=True)

        # Handle early stopping setup
        if self.use_early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=42)
            best_val_loss = np.inf
            patience_counter = 0
        else:
            X_train, y_train = X, y

        # Initialize residuals
        residuals = y_train.copy()

        # Boosting iterations
        for epoch in range(self.max_epochs):
            epoch_improvement = False

            # Fit smoother for each feature
            for j in range(n_features):
                X_j = X_train[:, j]
                smoother = self._get_smoother()
                smoother.fit(X_j, residuals)
                point_predictions = smoother.predict(self.feature_points_[j])
                predictions = smoother.predict(X_j)
                self.feature_values_[j] += self.lr * point_predictions
                residuals -= self.lr * predictions

            # Check early stopping condition
            if self.use_early_stopping:
                val_preds = self.predict(X_val)
                val_loss = np.mean((val_preds - y_val) ** 2)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    epoch_improvement = True
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    break

        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        for j in range(X.shape[1]):
            X_j = X[:, j]
            feature_predictions = np.interp(X_j, self.feature_points_[j], self.feature_values_[j], left=self.feature_values_[j][0], right=self.feature_values_[j][-1])
            predictions += feature_predictions
        return predictions

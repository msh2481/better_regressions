"""Scaling transformations for regression inputs and targets."""

import numpy as np
from beartype import beartype as typed
from jaxtyping import Float
from numpy import ndarray as ND
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from better_regressions.repr_utils import format_array


@typed
class Scaler(BaseEstimator, RegressorMixin):
    """Wraps a regression estimator with scaling for inputs and targets.

    Args:
        estimator: The regression estimator to wrap
        x_method: Scaling method for input features
        y_method: Scaling method for target values
        use_feature_variance: If True, normalize y based on sqrt(sum(var(X_scaled))) before y_method
    """

    def __init__(self, estimator, x_method: str = "standard", y_method: str = "standard", use_feature_variance: bool = True):
        self.estimator = estimator
        self.x_method = x_method
        self.y_method = y_method
        self.use_feature_variance = use_feature_variance
        self._validate_methods()
        self.wrapped_estimator = clone(estimator)

    def _validate_methods(self):
        """Validate scaling method names."""
        valid_methods = ["none", "standard", "quantile-uniform", "quantile-normal"]
        if self.x_method not in valid_methods:
            raise ValueError(f"Invalid x_method: {self.x_method}. Choose from {valid_methods}")
        if self.y_method not in valid_methods:
            raise ValueError(f"Invalid y_method: {self.y_method}. Choose from {valid_methods}")

    def _get_transformer(self, method: str):
        """Get transformer instance based on method name."""
        if method == "none":
            return StandardScaler(with_mean=False, with_std=False)
        elif method == "standard":
            return StandardScaler(with_mean=False)
        elif method == "quantile-uniform":
            return QuantileTransformer(output_distribution="uniform")
        elif method == "quantile-normal":
            return QuantileTransformer(output_distribution="normal")

    @typed
    def __repr__(self, var_name: str = "model") -> str:
        """Generate code to recreate this model."""
        lines = []

        # Create base Scaler instance
        estimator_repr = repr(self.estimator)
        assert hasattr(self.estimator, "__repr__") and callable(self.estimator.__repr__)
        est_var = f"{var_name}_est"
        estimator_repr = self.estimator.__repr__(var_name=est_var)
        lines.append(estimator_repr)

        init_line = f"{var_name} = Scaler(estimator={est_var}, x_method='{self.x_method}', y_method='{self.y_method}', use_feature_variance={self.use_feature_variance})"
        lines.append(init_line)

        # If fitted, add transformer states and normalization factor
        if hasattr(self, "x_transformer_") and hasattr(self, "y_transformer_"):
            # TODO: Add code to properly recreate the fitted transformers
            lines.append(f"# TODO: Add code to properly recreate the fitted transformers")
        if self.use_feature_variance and hasattr(self, "y_norm_factor_"):
            lines.append(f"{var_name}.y_norm_factor_ = {self.y_norm_factor_:.9g}")

        return "\n".join(lines)

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> "Scaler":
        """Fit transformers and the wrapped estimator.

        Args:
            X: Input features
            y: Target values

        Returns:
            self: Fitted scaler
        """
        self.x_transformer_ = self._get_transformer(self.x_method)
        self.y_transformer_ = self._get_transformer(self.y_method)
        X_scaled = self.x_transformer_.fit_transform(X)
        sum_variances = np.sum(np.var(X_scaled, axis=0))
        if self.use_feature_variance:
            self.y_norm_factor_ = np.sqrt(sum_variances + 1e-18)
        else:
            self.y_norm_factor_ = 1.0

        y_2d = y.reshape(-1, 1)
        y_to_transform = y_2d / self.y_norm_factor_
        y_scaled = self.y_transformer_.fit_transform(y_to_transform).ravel()
        y_scaled *= self.y_norm_factor_
        self.wrapped_estimator.fit(X_scaled, y_scaled)
        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        X_scaled = self.x_transformer_.transform(X)
        y_scaled_pred = self.wrapped_estimator.predict(X_scaled) / self.y_norm_factor_
        y_scaled_pred = y_scaled_pred.reshape(-1, 1)
        y_pred = self.y_transformer_.inverse_transform(y_scaled_pred) * self.y_norm_factor_
        return y_pred.ravel()

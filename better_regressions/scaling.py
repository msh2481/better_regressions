"""Scaling transformations for regression inputs and targets."""

import numpy as np
from beartype import beartype as typed
from jaxtyping import Float
from numpy import ndarray as ND
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from better_regressions.utils import format_array


class SecondMomentScaler(BaseEstimator, RegressorMixin):
    """Scales data by dividing by the square root of the second moment (mean of squares)"""

    def __init__(self):
        pass

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y=None) -> "SecondMomentScaler":
        self.scale_ = np.sqrt(np.mean(X**2, axis=0) + 1e-18)
        return self

    @typed
    def transform(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples n_features"]:
        return X / self.scale_

    @typed
    def fit_transform(self, X: Float[ND, "n_samples n_features"], y=None) -> Float[ND, "n_samples n_features"]:
        return self.fit(X).transform(X)

    @typed
    def inverse_transform(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples n_features"]:
        return X * self.scale_


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
        self.estimator_ = clone(estimator)

    def _validate_methods(self):
        """Validate scaling method names."""
        valid_methods = ["none", "standard", "quantile-uniform", "quantile-normal", "power"]
        if self.x_method not in valid_methods:
            raise ValueError(f"Invalid x_method: {self.x_method}. Choose from {valid_methods}")
        if self.y_method not in valid_methods:
            raise ValueError(f"Invalid y_method: {self.y_method}. Choose from {valid_methods}")

    def _get_transformer(self, method: str):
        """Get transformer instance based on method name."""
        if method == "none":
            return StandardScaler(with_mean=False, with_std=False)
        elif method == "standard":
            return SecondMomentScaler()
        elif method == "quantile-uniform":
            return QuantileTransformer(output_distribution="uniform", n_quantiles=20)
        elif method == "quantile-normal":
            return QuantileTransformer(output_distribution="normal", n_quantiles=20)
        elif method == "power":
            return PowerTransformer()

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
        self.x_transformer_ = self._get_transformer(self.x_method)
        self.y_transformer_ = self._get_transformer(self.y_method)
        X_scaled = self.x_transformer_.fit_transform(X)
        sum_second_moment = np.sum(np.mean(X_scaled**2, axis=0))
        if self.use_feature_variance:
            self.y_norm_factor_ = np.sqrt(sum_second_moment + 1e-18)
        else:
            self.y_norm_factor_ = 1.0

        y_2d = y.reshape(-1, 1)
        y_to_transform = y_2d / self.y_norm_factor_
        y_scaled = self.y_transformer_.fit_transform(y_to_transform).ravel()

        # Store min/max bounds if using PowerTransformer to handle out-of-bounds later
        if isinstance(self.y_transformer_, PowerTransformer):
            self.y_min_ = np.min(y_scaled)
            self.y_max_ = np.max(y_scaled)

        y_scaled *= self.y_norm_factor_
        self.estimator_.fit(X_scaled, y_scaled)
        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        assert not np.any(np.isnan(X)), "X contains NaNs"
        X_scaled = self.x_transformer_.transform(X)
        assert not np.any(np.isnan(X_scaled)), "X_scaled contains NaNs"
        y_scaled_pred = self.estimator_.predict(X_scaled) / self.y_norm_factor_
        assert not np.any(np.isnan(y_scaled_pred)), "y_scaled_pred contains NaNs"
        y_scaled_pred = y_scaled_pred.reshape(-1, 1)

        # Clip values before inverse_transform to avoid NaNs when using PowerTransformer
        if isinstance(self.y_transformer_, PowerTransformer) and hasattr(self, "y_min_"):
            y_scaled_pred = np.clip(y_scaled_pred, self.y_min_, self.y_max_)

        y_pred = self.y_transformer_.inverse_transform(y_scaled_pred) * self.y_norm_factor_
        assert not np.any(np.isnan(y_pred)), "y_pred contains NaNs"
        return y_pred.ravel()


@typed
class AutoScaler(BaseEstimator, RegressorMixin):
    """Automatically selects the best scaling method for input and target.

    Tries various combinations of scaling methods (standard, power, quantile-normal)
    and selects the best one based on validation performance.

    Args:
        estimator: The regression estimator to wrap
        use_feature_variance: If True, normalize y based on sqrt(sum(var(X_scaled)))
        val_size: Fraction of data to use for validation when selecting scaling
        random_state: Random state for train/val split
    """

    def __init__(self, estimator, val_size: float = 0.3, random_state: int = 42):
        self.estimator = estimator
        self.val_size = val_size
        self.random_state = random_state
        self.estimator_ = clone(estimator)

    @typed
    def __repr__(self, var_name: str = "model") -> str:
        """Generate code to recreate this model."""
        lines = []

        # Create base AutoScaler instance
        estimator_repr = repr(self.estimator)
        assert hasattr(self.estimator, "__repr__") and callable(self.estimator.__repr__)
        est_var = f"{var_name}_est"
        estimator_repr = self.estimator.__repr__(var_name=est_var)
        lines.append(estimator_repr)

        init_line = f"{var_name} = AutoScaler(estimator={est_var}, val_size={self.val_size}, random_state={self.random_state})"
        lines.append(init_line)

        # If fitted, add selected scaler info
        if hasattr(self, "best_scaler_"):
            lines.append(f"# Selected x_method='{self.best_x_method_}', y_method='{self.best_y_method_}' with score: {self.best_score_:.6f}")

        return "\n".join(lines)

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> "AutoScaler":
        """Fits multiple scalers and selects the best one."""
        # Define scaling combinations to try
        scaling_methods = [("standard", "standard"), ("power", "power"), ("quantile-normal", "quantile-normal")]

        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state)

        # Try each scaling method
        best_score = float("inf")
        best_scaler = None
        best_x_method = None
        best_y_method = None

        for x_method, y_method in scaling_methods:
            # Create and fit scaler
            scaler = Scaler(clone(self.estimator), x_method=x_method, y_method=y_method)

            try:
                scaler.fit(X_train, y_train)
                # Evaluate on validation set
                y_val_pred = scaler.predict(X_val)
                score = mean_squared_error(y_val, y_val_pred)
                # Update best if better
                if score < best_score:
                    best_score = score
                    best_scaler = scaler
                    best_x_method = x_method
                    best_y_method = y_method
            except Exception as e:
                print(f"Error with x_method={x_method}, y_method={y_method}: {str(e)}")
                continue

        if best_scaler is None:
            raise ValueError("No valid scaling method found. All combinations failed.")

        # Store best configuration
        self.best_x_method_ = best_x_method
        self.best_y_method_ = best_y_method
        self.best_score_ = best_score

        # Refit the best scaler on the full dataset
        self.best_scaler_ = Scaler(clone(self.estimator), x_method=best_x_method, y_method=best_y_method)
        self.best_scaler_.fit(X, y)

        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        """Predicts using the best selected scaler."""
        if not hasattr(self, "best_scaler_"):
            raise RuntimeError("AutoScaler instance is not fitted yet. Call 'fit' first.")
        return self.best_scaler_.predict(X)


class DebugEstimator(BaseEstimator, RegressorMixin):
    """Estimator that just prints stats of data distribution during fit"""

    def __init__(self):
        pass

    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> "DebugEstimator":
        X_means = np.mean(X, axis=0)
        X_stds = np.std(X, axis=0)
        print(f"X means: {X_means.mean():.3g} +-{X_means.std():.3g}")
        print(f"X stds: {X_stds.mean():.3g} +-{X_stds.std():.3g}")
        print(f"y mean: {np.mean(y)}")
        print(f"y std: {np.std(y)}")
        print()
        return self

    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        return X

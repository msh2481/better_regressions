"""Scaling transformations for regression inputs and targets."""

import numpy as np
from beartype import beartype as typed
from jaxtyping import Float
from numpy import ndarray as ND
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from better_regressions.utils import format_array


class Stabilize(BaseEstimator):
    def __init__(self, k: float = 1.0):
        self.k = k

    @typed
    def _process_col(self, arr: Float[ND, "n_samples"]) -> tuple[float, float, float, float]:
        assert arr.ndim == 1
        unique = np.unique(arr)
        median_ = np.median(arr)
        if len(unique) < 100:
            min_ = np.min(unique)
            max_ = np.max(unique)
        else:
            min_, max_ = np.percentile(arr, [2, 98])
            rng = max_ - min_ + 1e-9
            min_ = median_ - self.k * rng
            max_ = median_ + self.k * rng
        scale_ = 0.5 * (max_ - min_) + 1e-9
        return min_, max_, median_, scale_

    @typed
    def fit(self, arr: Float[ND, "n_samples"] | Float[ND, "n_samples n_features"], y=None):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.ravel()
            self.min_, self.max_, self.median_, self.scale_ = self._process_col(arr)
        else:
            self.min_ = np.zeros(arr.shape[1])
            self.max_ = np.zeros(arr.shape[1])
            self.median_ = np.zeros(arr.shape[1])
            self.scale_ = np.zeros(arr.shape[1])
            for i in range(arr.shape[1]):
                self.min_[i], self.max_[i], self.median_[i], self.scale_[i] = self._process_col(arr[:, i])
        return self

    @typed
    def transform(self, arr: Float[ND, "n_samples"] | Float[ND, "n_samples n_features"]):
        arr = np.asarray(arr)
        arr = np.clip(arr, self.min_, self.max_)
        return (arr - self.median_) / self.scale_

    def fit_transform(self, arr: Float[ND, "n_samples"] | Float[ND, "n_samples n_features"]):
        return self.fit(arr).transform(arr)

    @typed
    def inverse_transform(self, arr: Float[ND, "n_samples"] | Float[ND, "n_samples n_features"]):
        arr = np.asarray(arr)
        return arr * self.scale_ + self.median_


class SecondMomentScaler(BaseEstimator, RegressorMixin):
    """Scales data by dividing by the square root of the second moment (mean of squares)"""

    def __init__(self):
        pass

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y=None) -> "SecondMomentScaler":
        self.scale_ = np.sqrt(np.mean(X**2, axis=0) + 1e-30)
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

    def __init__(self, estimator, x_method: str = "stabilize", y_method: str = "stabilize", use_feature_variance: bool = True):
        self.estimator = estimator
        self.x_method = x_method
        self.y_method = y_method
        self.use_feature_variance = use_feature_variance
        self.estimator_ = clone(estimator)

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
        elif method == "stabilize":
            return Stabilize()
        else:
            raise ValueError(f"Invalid method: {method}")

    @typed
    def __repr__(self, var_name: str = "model") -> str:
        """Generate code to recreate this model."""
        lines = []

        est_var = f"{var_name}_est"
        estimator_repr = self.estimator_.__repr__(var_name=est_var)
        lines.append(estimator_repr)

        init_line = f"{var_name} = Scaler(estimator={est_var}, x_method='{self.x_method}', y_method='{self.y_method}', use_feature_variance={self.use_feature_variance})"
        lines.append(init_line)
        lines.append(f"{var_name}.estimator_ = {est_var}")

        # If fitted, add transformer states and normalization factor
        if hasattr(self, "x_transformer_") and hasattr(self, "y_transformer_"):
            # Add code to properly recreate the fitted transformers
            if isinstance(self.x_transformer_, SecondMomentScaler):
                lines.append(f"{var_name}.x_transformer_ = SecondMomentScaler()")
                lines.append(f"{var_name}.x_transformer_.scale_ = {format_array(self.x_transformer_.scale_)}")
            elif isinstance(self.x_transformer_, StandardScaler):
                lines.append(f"{var_name}.x_transformer_ = StandardScaler(with_mean=False, with_std=False)")
            elif isinstance(self.x_transformer_, QuantileTransformer):
                output_dist = "'uniform'" if self.x_method == "quantile-uniform" else "'normal'"
                lines.append(f"{var_name}.x_transformer_ = QuantileTransformer(output_distribution={output_dist}, n_quantiles=20)")
                if hasattr(self.x_transformer_, "quantiles_"):
                    lines.append(f"{var_name}.x_transformer_.quantiles_ = {format_array(self.x_transformer_.quantiles_)}")
                if hasattr(self.x_transformer_, "references_"):
                    lines.append(f"{var_name}.x_transformer_.references_ = {format_array(self.x_transformer_.references_)}")
            elif isinstance(self.x_transformer_, PowerTransformer):
                lines.append(f"{var_name}.x_transformer_ = PowerTransformer()")
                if hasattr(self.x_transformer_, "lambdas_"):
                    lines.append(f"{var_name}.x_transformer_.lambdas_ = {format_array(self.x_transformer_.lambdas_)}")
                if hasattr(self.x_transformer_, "_scaler"):
                    lines.append(f"{var_name}.x_transformer_._scaler.scale_ = {format_array(self.x_transformer_._scaler.scale_)}")
                    lines.append(f"{var_name}.x_transformer_._scaler.mean_ = {format_array(self.x_transformer_._scaler.mean_)}")
            elif isinstance(self.x_transformer_, Stabilize):
                lines.append(f"{var_name}.x_transformer_ = Stabilize()")
                lines.append(f"{var_name}.x_transformer_.median_ = {format_array(self.x_transformer_.median_)}")
                lines.append(f"{var_name}.x_transformer_.min_ = {format_array(self.x_transformer_.min_)}")
                lines.append(f"{var_name}.x_transformer_.max_ = {format_array(self.x_transformer_.max_)}")
                lines.append(f"{var_name}.x_transformer_.scale_ = {format_array(self.x_transformer_.scale_)}")

            # Y transformer
            if isinstance(self.y_transformer_, SecondMomentScaler):
                lines.append(f"{var_name}.y_transformer_ = SecondMomentScaler()")
                lines.append(f"{var_name}.y_transformer_.scale_ = {format_array(self.y_transformer_.scale_)}")
            elif isinstance(self.y_transformer_, StandardScaler):
                lines.append(f"{var_name}.y_transformer_ = StandardScaler(with_mean=False, with_std=False)")
            elif isinstance(self.y_transformer_, QuantileTransformer):
                output_dist = "'uniform'" if self.y_method == "quantile-uniform" else "'normal'"
                lines.append(f"{var_name}.y_transformer_ = QuantileTransformer(output_distribution={output_dist}, n_quantiles=20)")
                if hasattr(self.y_transformer_, "quantiles_"):
                    lines.append(f"{var_name}.y_transformer_.quantiles_ = {format_array(self.y_transformer_.quantiles_)}")
                if hasattr(self.y_transformer_, "references_"):
                    lines.append(f"{var_name}.y_transformer_.references_ = {format_array(self.y_transformer_.references_)}")
            elif isinstance(self.y_transformer_, PowerTransformer):
                lines.append(f"{var_name}.y_transformer_ = PowerTransformer()")
                if hasattr(self.y_transformer_, "lambdas_"):
                    lines.append(f"{var_name}.y_transformer_.lambdas_ = {format_array(self.y_transformer_.lambdas_)}")
                if hasattr(self.y_transformer_, "_scaler"):
                    lines.append(f"{var_name}.y_transformer_._scaler.scale_ = {format_array(self.y_transformer_._scaler.scale_)}")
                    lines.append(f"{var_name}.y_transformer_._scaler.mean_ = {format_array(self.y_transformer_._scaler.mean_)}")
            elif isinstance(self.y_transformer_, Stabilize):
                lines.append(f"{var_name}.y_transformer_ = Stabilize()")
                lines.append(f"{var_name}.y_transformer_.median_ = {format_array(self.y_transformer_.median_)}")
                lines.append(f"{var_name}.y_transformer_.min_ = {format_array(self.y_transformer_.min_)}")
                lines.append(f"{var_name}.y_transformer_.max_ = {format_array(self.y_transformer_.max_)}")
                lines.append(f"{var_name}.y_transformer_.scale_ = {format_array(self.y_transformer_.scale_)}")

            # Add y_min_ and y_max_ if using PowerTransformer
            if isinstance(self.y_transformer_, PowerTransformer) and hasattr(self, "y_min_") and hasattr(self, "y_max_"):
                lines.append(f"{var_name}.y_min_ = {self.y_min_:.9g}")
                lines.append(f"{var_name}.y_max_ = {self.y_max_:.9g}")

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
        lines.append(f"{var_name}_est = {repr(self.estimator)}")
        init_line = f"{var_name} = AutoScaler(estimator={var_name}_est, val_size={self.val_size}, random_state={self.random_state})"
        lines.append(init_line)

        # If fitted, add selected scaler info
        if hasattr(self, "best_scaler_"):
            lines.append(f"{var_name}.best_x_method_ = '{self.best_x_method_}'")
            lines.append(f"{var_name}.best_y_method_ = '{self.best_y_method_}'")
            lines.append(f"{var_name}.best_score_ = {self.best_score_:.6f}")

            # Use the best_scaler's __repr__ method
            scaler_repr = self.best_scaler_.__repr__(var_name=f"{var_name}_best_scaler")
            lines.append(scaler_repr)
            lines.append(f"{var_name}.best_scaler_ = {var_name}_best_scaler")

        return "\n".join(lines)

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]) -> "AutoScaler":
        """Fits multiple scalers and selects the best one."""
        # Define scaling combinations to try
        scaling_methods = [("stabilize", "stabilize"), ("standard", "standard"), ("power", "power"), ("quantile-normal", "quantile-normal")]

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

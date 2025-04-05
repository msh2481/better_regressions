"""Tests for smooth regression models."""

import numpy as np
from beartype import beartype as typed

from better_regressions.scaling import Scaler
from better_regressions.smoothing import Smooth
from jaxtyping import Float
from numpy import ndarray as ND
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error


def test_smooth_sine():
    """Test Smooth class on a sine wave function."""
    np.random.seed(42)

    # Create synthetic data with sine wave pattern
    n_samples = 200
    X = np.random.uniform(-3, 3, (n_samples, 1))
    y = np.sin(X.ravel()) + np.random.normal(0, 0.2, n_samples)

    # Train model
    model = Smooth(method="supersmoother", lr=0.2, max_epochs=20, n_points=30)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate error
    mse = mean_squared_error(y, y_pred)
    print(f"Smooth regression MSE on sine data: {mse:.4f}")

    # Test points for more detailed evaluation
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = np.sin(X_test.ravel())
    y_pred_test = model.predict(X_test)
    test_mse = mean_squared_error(y_true, y_pred_test)
    print(f"Test MSE on clean sine data: {test_mse:.4f}")

    assert mse < 0.1, "MSE should be relatively low for sine wave pattern"
    assert test_mse < 0.1, "Test MSE should be relatively low for clean sine data"


def test_smooth_nonlinear_multivariate():
    """Test Smooth class on nonlinear multivariate data."""
    np.random.seed(42)

    # Create synthetic nonlinear multivariate data
    n_samples = 300
    n_features = 3
    X = np.random.uniform(-2, 2, (n_samples, n_features))

    # Nonlinear function: y = sin(x0) + x1^2 - exp(x2/2)
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 - np.exp(X[:, 2] / 2)
    y += np.random.normal(0, 0.3, n_samples)  # Add noise

    # Train model with early stopping
    model = Smooth(method="supersmoother", lr=0.1, max_epochs=30, n_points=20, use_early_stopping=True, val_size=0.2, patience=3)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate error
    mse = mean_squared_error(y, y_pred)
    print(f"Smooth regression MSE on nonlinear multivariate data: {mse:.4f}")

    assert mse < 0.5, "MSE should be reasonable for nonlinear multivariate data"


def test_smooth_vs_linear():
    """Compare Smooth regression with Linear regression on nonlinear data."""
    np.random.seed(42)

    # Create synthetic data with quadratic pattern
    n_samples = 200
    X = np.random.uniform(-3, 3, (n_samples, 1))
    y = X.ravel() ** 2 + np.random.normal(0, 0.5, n_samples)

    # Train Smooth model
    smooth_model = Smooth(method="supersmoother", lr=0.2, max_epochs=20)
    smooth_model.fit(X, y)
    y_pred_smooth = smooth_model.predict(X)
    smooth_mse = mean_squared_error(y, y_pred_smooth)

    # Train Linear model (using Scaler for fair comparison)
    from better_regressions.linear import Linear

    linear_model = Scaler(Linear())
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)
    linear_mse = mean_squared_error(y, y_pred_linear)

    print(f"Smooth regression MSE: {smooth_mse:.4f}")
    print(f"Linear regression MSE: {linear_mse:.4f}")

    # Smooth should perform better on this nonlinear data
    assert smooth_mse < linear_mse, "Smooth regression should outperform linear on quadratic data"


def test_smooth_angle_method():
    """Test Smooth class with the 'angle' method on piecewise data."""
    np.random.seed(42)

    # Create synthetic piecewise linear data
    n_samples = 300
    X = np.sort(np.random.uniform(-5, 5, n_samples)).reshape(-1, 1)
    x_val = X.ravel()

    # Create a piecewise function with multiple segments
    y = np.zeros(n_samples)

    # Create sharp transitions at specific points
    y[x_val < -3] = -3 - 0.7 * (x_val[x_val < -3] + 3)
    y[(x_val >= -3) & (x_val < -1)] = -3 + 1.5 * (x_val[(x_val >= -3) & (x_val < -1)] + 3)
    y[(x_val >= -1) & (x_val < 1)] = 0
    y[(x_val >= 1) & (x_val < 3)] = 2 * (x_val[(x_val >= 1) & (x_val < 3)] - 1)
    y[x_val >= 3] = 4 + 0.5 * (x_val[x_val >= 3] - 3)

    # Add noise
    y += np.random.normal(0, 0.2, n_samples)

    # Train models with different methods
    supersmoother_model = Smooth(method="supersmoother", max_epochs=30, lr=0.1)
    supersmoother_model.fit(X, y)

    angle_model = Smooth(method="angle", max_epochs=30, lr=0.1, n_breakpoints=10)
    angle_model.fit(X, y)

    # Make predictions
    y_pred_supersmoother = supersmoother_model.predict(X)
    y_pred_angle = angle_model.predict(X)

    # Calculate errors
    supersmoother_mse = mean_squared_error(y, y_pred_supersmoother)
    angle_mse = mean_squared_error(y, y_pred_angle)

    print(f"Smooth with SuperSmoother MSE: {supersmoother_mse:.4f}")
    print(f"Smooth with Angle MSE: {angle_mse:.4f}")

    # Both models should perform reasonably well
    assert supersmoother_mse < 0.5, "SuperSmoother MSE should be reasonable"
    assert angle_mse < 0.5, "Angle MSE should be reasonable"


def test_compare_smooth_methods():
    """Compare different smoothing methods on a sine wave with sharp transition."""
    np.random.seed(42)

    # Generate data with sine wave and sharp transition
    n_samples = 200
    X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)
    x_val = X.ravel()

    # Sine wave with sharp transition at x=0
    y = np.sin(x_val)
    y[x_val > 0] += 1.0  # Add step at x=0

    # Add noise
    y += np.random.normal(0, 0.1, n_samples)

    # Train different models
    models = {
        "SuperSmoother (10 epochs)": Smooth(method="supersmoother", max_epochs=10, lr=0.1),
        "SuperSmoother (30 epochs)": Smooth(method="supersmoother", max_epochs=30, lr=0.1),
        "Angle (10 epochs)": Smooth(method="angle", max_epochs=10, lr=0.1, n_breakpoints=5),
        "Angle (30 epochs)": Smooth(method="angle", max_epochs=30, lr=0.1, n_breakpoints=5),
    }

    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        print(f"{name} MSE: {mse:.4f}")

        # All models should perform reasonably well
        assert mse < 0.5, f"{name} MSE should be reasonable"


if __name__ == "__main__":
    test_smooth_sine()
    test_smooth_nonlinear_multivariate()
    test_smooth_vs_linear()
    test_smooth_angle_method()
    test_compare_smooth_methods()

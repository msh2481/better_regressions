"""Tests for piecewise linear regression models."""

import numpy as np
from beartype import beartype as typed

from better_regressions.linear import Linear
from better_regressions.piecewise import Angle
from better_regressions.scaling import Scaler
from jaxtyping import Float
from numpy import ndarray as ND
from sklearn.metrics import mean_squared_error


def test_angle_basic():
    """Test Angle class on basic piecewise linear data."""
    np.random.seed(42)

    # Create synthetic piecewise linear data
    n_samples = 200
    X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)

    # Create a piecewise linear function with 2 breakpoints
    breakpoints = np.array([-1.0, 1.0])
    y = np.zeros(n_samples)

    # y = 1 - x for x < -1
    mask = X.ravel() < breakpoints[0]
    y[mask] = 1 - X.ravel()[mask]

    # y = 0.5*x for -1 <= x < 1
    mask = (X.ravel() >= breakpoints[0]) & (X.ravel() < breakpoints[1])
    y[mask] = 0.5 * X.ravel()[mask]

    # y = 2 + 0.3*(x-1) for x >= 1
    mask = X.ravel() >= breakpoints[1]
    y[mask] = 2 + 0.3 * (X.ravel()[mask] - breakpoints[1])

    # Add noise
    y += np.random.normal(0, 0.05, n_samples)

    # Train model
    model = Angle(n_breakpoints=5, random_state=42)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate error
    mse = mean_squared_error(y, y_pred)
    print(f"Angle regression MSE on piecewise linear data: {mse:.4f}")

    # The model should fit reasonably well, but not perfectly due to random breakpoint selection
    assert mse < 0.25, "MSE should be reasonably low for piecewise linear data"


def test_angle_vs_linear():
    """Compare Angle with Linear on piecewise linear data."""
    np.random.seed(42)

    # Create synthetic piecewise linear data
    n_samples = 300
    X = np.sort(np.random.uniform(-5, 5, n_samples)).reshape(-1, 1)

    # Function with multiple breakpoints
    y = np.zeros(n_samples)
    x_val = X.ravel()

    # Define multiple segments
    y = np.sin(X.ravel()) + 0.05 * X.ravel() ** 2
    # Add sharp angles at specific points
    y[x_val > -2] += 0.5
    y[x_val > 0] -= 0.8
    y[x_val > 2] += 1.0

    # Add noise
    y += np.random.normal(0, 0.1, n_samples)

    # Train models
    angle_model = Angle(n_breakpoints=10, random_state=42)
    angle_model.fit(X, y)

    linear_model = Scaler(Linear())
    linear_model.fit(X, y)

    # Make predictions
    y_pred_angle = angle_model.predict(X)
    y_pred_linear = linear_model.predict(X)

    # Calculate errors
    angle_mse = mean_squared_error(y, y_pred_angle)
    linear_mse = mean_squared_error(y, y_pred_linear)

    print(f"Angle regression MSE: {angle_mse:.4f}")
    print(f"Linear regression MSE: {linear_mse:.4f}")

    # Angle should outperform linear on this data
    assert angle_mse < linear_mse, "Angle should outperform Linear on piecewise data"


def test_angle_multivariate():
    """Test Angle model on multivariate data with piecewise relationships."""
    np.random.seed(42)

    # Create synthetic multivariate data
    n_samples = 200
    n_features = 3
    X = np.random.uniform(-2, 2, (n_samples, n_features))

    # Create target with piecewise relationships on each feature
    y = np.zeros(n_samples)

    # Feature 0: step function
    y += (X[:, 0] > 0).astype(float) * 2 - 1

    # Feature 1: piecewise linear
    y += np.maximum(0, X[:, 1]) * 1.5

    # Feature 2: absolute value
    y += np.abs(X[:, 2]) * 0.8

    # Add noise
    y += np.random.normal(0, 0.2, n_samples)

    # Train model
    model = Angle(n_breakpoints=8, random_state=42)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate error
    mse = mean_squared_error(y, y_pred)
    print(f"Angle regression MSE on multivariate piecewise data: {mse:.4f}")

    # The model should fit this data reasonably well
    assert mse < 0.5, "MSE should be reasonable for multivariate piecewise data"


if __name__ == "__main__":
    test_angle_basic()
    test_angle_vs_linear()
    test_angle_multivariate()

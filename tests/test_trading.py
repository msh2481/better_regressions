from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from better_regressions.linear import Linear
from better_regressions.scaling import AutoScaler, Scaler
from better_regressions.smoothing import Smooth
from rich.console import Console
from rich.table import Table
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from tests.data import CURVES


def generate_trading_dataset(n_samples: int = 1000, n_features: int = 10, seed: int = 42, noise_level: float = 0.0, noise_type: Literal["gaussian", "cauchy"] = "gaussian") -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic trading dataset.

    X features are distributed as Beta(5, 5) on [0, 1].
    y is defined as sum_i f_i(X_i), where f_i is given by i-th curve from data.py.
    y is shifted so that only 20% of points are profitable.

    Args:
        n_samples: Number of data points to generate
        n_features: Number of features (limited by number of curves available)
        seed: Random seed for reproducibility
        noise_level: Standard deviation of noise to add to y
        noise_type: Type of noise distribution ("gaussian" or "cauchy")

    Returns:
        X: Feature matrix with shape (n_samples, n_features)
        y: Target vector with shape (n_samples,)
    """
    rng = np.random.RandomState(seed)

    # Generate Beta(5, 5) distributed features on [0, 1]
    X = rng.beta(5, 5, size=(n_samples, n_features))

    # Initialize target
    y = np.zeros(n_samples)

    # For each feature, add the corresponding curve value using interpolation
    for i in range(min(n_features, len(CURVES))):
        # Create x points for interpolation (normalized curve indices)
        x_points = np.linspace(0, 1, len(CURVES[i]))
        # Interpolate using feature values
        y += np.interp(X[:, i], x_points, CURVES[i])

    # Add noise if specified
    if noise_level > 0:
        if noise_type == "gaussian":
            y += rng.randn(size=n_samples) * noise_level
        elif noise_type == "cauchy":
            y += rng.standard_cauchy(size=n_samples) * noise_level

    # Shift y so that only 20% of points are profitable
    percentile_80 = np.percentile(y, 80)
    y -= percentile_80

    return X, y


def test_visualize_trading_dataset():
    """Visualize the generated trading dataset."""
    # Generate dataset
    X, y = generate_trading_dataset(n_samples=1000, n_features=10, noise_level=5.0)

    # Visualize X0 distribution
    plt.figure(figsize=(6, 4))
    plt.hist(X[:, 0], bins=30, alpha=0.7)
    plt.title("X0 Distribution (Beta(5, 5))")
    plt.xlabel("X0")
    plt.ylabel("Frequency")
    plt.savefig("x0_distribution.png")
    plt.close()

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot y vs X0
    axs[0].scatter(X[:, 0], y, alpha=0.5, s=10)
    axs[0].axhline(y=0, color="r", linestyle="--")
    axs[0].set_title("y vs X0")
    axs[0].set_xlabel("X0")
    axs[0].set_ylabel("y")

    # Plot y vs X1
    axs[1].scatter(X[:, 1], y, alpha=0.5, s=10)
    axs[1].axhline(y=0, color="r", linestyle="--")
    axs[1].set_title("y vs X1")
    axs[1].set_xlabel("X1")
    axs[1].set_ylabel("y")

    plt.tight_layout()
    plt.savefig("trading_dataset_visualization.png")
    plt.close()


def calculate_metrics(model: Callable, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float]:
    """Calculate MSE and profit-based metrics for a model.

    Args:
        model: Estimator with fit/predict methods
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets

    Returns:
        mse: Mean squared error
        profit: Mean profit (max(prediction, 0) * y)
        sharpe: Sharpe ratio (mean profit / std profit)
    """
    # Train model
    model.fit(X_train, y_train)

    # Predict on test set
    preds = model.predict(X_test)

    # Calculate MSE
    mse = mean_squared_error(y_test, preds)

    # Calculate profit: max(prediction, 0) * actual_y
    profits = np.maximum(preds, 0) * y_test

    # Calculate mean profit and Sharpe ratio
    mean_profit = np.mean(profits)
    sharpe = mean_profit / (np.std(profits) + 1e-10)  # Add small epsilon to avoid division by zero

    return mse, mean_profit, sharpe


def test_trading_regressors():
    """Test different regressors on the trading dataset."""
    # Generate dataset with noise
    X, y = generate_trading_dataset(n_samples=2000, n_features=10, noise_level=10.0, noise_type="gaussian")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Models to test
    models = {"Scaler(Linear())": Scaler(Linear()), "AutoScaler(Linear())": AutoScaler(Linear()), "AutoScaler(Smooth('angle'))": AutoScaler(Smooth(method="angle"))}

    # Results storage
    results = {name: {"mse": [], "profit": [], "sharpe": []} for name in models}

    # Cross-validation with multiple folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Test each model
        for name, model in models.items():
            mse, profit, sharpe = calculate_metrics(model, X_fold_train, y_fold_train, X_fold_val, y_fold_val)

            results[name]["mse"].append(mse)
            results[name]["profit"].append(profit)
            results[name]["sharpe"].append(sharpe)

    # Create rich table to display results
    table = Table(title="Trading Regressor Comparison")

    # Add columns
    table.add_column("Model", style="cyan")
    table.add_column("MSE", style="green")
    table.add_column("Mean Profit", style="yellow")
    table.add_column("Sharpe Ratio", style="magenta")

    # Add rows with mean values across folds
    for name in models:
        mse = np.mean(results[name]["mse"])
        profit = np.mean(results[name]["profit"])
        sharpe = np.mean(results[name]["sharpe"])

        table.add_row(name, f"{mse:.4f} ± {np.std(results[name]['mse']):.4f}", f"{profit:.4f} ± {np.std(results[name]['profit']):.4f}", f"{sharpe:.4f} ± {np.std(results[name]['sharpe']):.4f}")

    # Display table
    console = Console()
    console.print(table)


if __name__ == "__main__":
    # Run visualizations
    test_visualize_trading_dataset()

    # Run regressor comparisons
    test_trading_regressors()

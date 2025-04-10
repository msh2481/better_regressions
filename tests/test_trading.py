from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from better_regressions.linear import Linear
from better_regressions.scaling import AutoScaler, Scaler
from better_regressions.smoothing import Smooth
from rich.console import Console
from rich.table import Table
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from tests.data import CURVES


def generate_trading_dataset(n_samples: int = 2000, n_features: int = 10, seed: int = 42, noise_level: float = 0.5, noise_type: Literal["gaussian", "cauchy"] = "cauchy") -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic trading dataset.

    X features are distributed as Beta(5, 5) on [0, 1].
    y is defined as sum_i f_i(X_i), where f_i is given by i-th curve from data.py.

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
    n_features = min(n_features, len(CURVES))
    rng = np.random.RandomState(seed)

    X = rng.beta(20, 20, size=(n_samples, n_features))
    scales0 = np.exp(rng.randn(n_features))
    scales1 = np.exp(rng.randn(n_features))

    # Initialize target
    y = np.zeros(n_samples)

    # For each feature, add the corresponding curve value using interpolation
    for i in range(n_features):
        # Create x points for interpolation (normalized curve indices)
        x_points = np.linspace(0, 1, len(CURVES[i]))
        scales = scales0[i] + (scales1[i] - scales0[i]) * np.linspace(-1, 1, n_samples) ** 3
        # Interpolate using feature values
        y += np.interp(X[:, i], x_points, CURVES[i]) * scales

    # Add noise if specified
    if noise_level > 0:
        if noise_type == "gaussian":
            X += rng.randn(n_samples, n_features) * (noise_level * X.std(axis=0, keepdims=True))
            y += rng.randn(n_samples) * noise_level * y.std()
        elif noise_type == "cauchy":
            C = 20
            X += np.clip(rng.standard_cauchy(size=(n_samples, n_features)), -C, C) * (noise_level * X.std(axis=0, keepdims=True))
            y += np.clip(rng.standard_cauchy(size=n_samples), -C, C) * noise_level * y.std()

    baseline = np.percentile(y, 70)
    y -= baseline

    return X, y


def test_visualize_trading_dataset():
    """Visualize the generated trading dataset."""
    # Generate dataset
    X, y = generate_trading_dataset(n_features=10)

    # Visualize X0 distribution
    plt.figure(figsize=(6, 4))
    plt.hist(X[:, 0], bins=30, alpha=0.7)
    plt.title("X0 Distribution")
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
    profits = np.clip(preds, 0, 0.1 * y_train.std()) * y_test

    # Calculate mean profit and Sharpe ratio
    mean_profit = np.mean(profits)
    sharpe = mean_profit / (np.std(profits) + 1e-10)  # Add small epsilon to avoid division by zero

    return mse, mean_profit, sharpe


def test_trading_regressors():
    """Test different regressors on the trading dataset."""
    # Generate dataset with noise
    X, y = generate_trading_dataset(n_features=10)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Models to test
    models = {
        "Scaler(Linear())": Scaler(Linear()),
        "AutoScaler(Linear())": AutoScaler(Linear()),
        "AutoScaler(Smooth('angle'))": AutoScaler(Smooth(method="angle")),
        "Scaler(GP(Matern() + WhiteKernel()))": Scaler(GaussianProcessRegressor(kernel=Matern() + WhiteKernel())),
        "Scaler(GP(RBF() + WhiteKernel()))": Scaler(GaussianProcessRegressor(kernel=RBF() + WhiteKernel())),
    }

    # Results storage
    results = {name: {"mse": [], "profit": [], "sharpe": []} for name in models}

    # Cross-validation with multiple folds
    kf = KFold(n_splits=5, shuffle=True)  # , random_state=42)

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
    test_visualize_trading_dataset()
    test_trading_regressors()

"""Tests for linear regression models."""

import numpy as np
import pandas as pd
from beartype import beartype as typed
from better_regressions import AutoScaler, Linear, Scaler, Smooth

from better_regressions.linear import Soft
from jaxtyping import Float
from numpy import ndarray as ND
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def test_linear_better_bias_equivalence():
    """Test if Linear with tiny alpha produces similar results regardless of better_bias."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    model_true = Linear(alpha=1e-18, better_bias=True)
    model_true.fit(X, y)
    pred_true = model_true.predict(X)
    model_false = Linear(alpha=1e-18, better_bias=False)
    model_false.fit(X, y)
    pred_false = model_false.predict(X)
    assert np.allclose(pred_true, pred_false, rtol=1e-5, atol=1e-5), "Predictions should be very similar with tiny alpha regardless of better_bias"
    mse_diff = mean_squared_error(pred_true, pred_false)
    print(f"MSE between better_bias=True and better_bias=False predictions: {mse_diff:.8f}")
    intercept_true = model_true.intercept_
    coef_true = model_true.coef_
    intercept_false = model_false.intercept_
    coef_false = model_false.coef_
    manual_pred_true = X @ coef_true + intercept_true
    manual_pred_false = X @ coef_false + intercept_false
    assert np.allclose(manual_pred_true, manual_pred_false, rtol=1e-5, atol=1e-5), "Manual predictions should be similar with tiny alpha regardless of better_bias"


@typed
def test_nonlinear_datasets():
    """Test that nonlinear datasets show improvement with smoothers vs linear models."""
    rng = np.random.RandomState(42)

    # Test one of our nonlinear datasets
    X = rng.uniform(-3, 3, size=(500, 2))
    y = np.sin(X[:, 0]) + 0.5 * np.cos(2 * X[:, 1]) + rng.normal(0, 0.1, 500)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Linear model as baseline
    linear_model = Scaler(Linear(alpha=1e-6), x_method="standard", y_method="standard", use_feature_variance=True)
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    linear_mse = mean_squared_error(y_test, linear_pred)

    # Smooth model (angle method)
    smooth_model = Scaler(Smooth(method="angle", n_points=100, max_epochs=100), x_method="standard", y_method="standard", use_feature_variance=True)
    smooth_model.fit(X_train, y_train)
    smooth_pred = smooth_model.predict(X_test)
    smooth_mse = mean_squared_error(y_test, smooth_pred)

    # The smoother should perform better on this nonlinear data
    print(f"Linear MSE: {linear_mse:.6f}")
    print(f"Smooth MSE: {smooth_mse:.6f}")
    improvement = 100 * (1 - smooth_mse / linear_mse)
    print(f"Improvement: {improvement:.2f}%")

    # Assert the smoother is better by at least 10%
    assert smooth_mse < 0.9 * linear_mse, "Smoother should outperform linear model on nonlinear data"


@typed
def generate_regression(n_samples: int, n_features: int, noise: float, outliers: float = 0.0, noninformative: float = 0.0) -> tuple[Float[ND, "n_samples n_features"], Float[ND, "n_samples"]]:
    """Generate a regression dataset."""
    # Random state seed to make results reproducible
    rng = np.random.RandomState(np.random.randint(0, 10000))

    # Generate features with different scales
    X = rng.randn(n_samples, n_features)
    scale = np.exp(rng.randn(n_features))
    w = rng.randn(n_features)
    y = X @ w
    X *= scale[None, :]

    # Add noise to target
    y += np.std(y) * rng.randn(n_samples) * noise

    # Add outliers if requested
    if outliers > 0.0:
        n_outliers = int(n_samples * outliers)
        outliers_idx = rng.choice(n_samples, size=n_outliers, replace=False)
        y[outliers_idx] = np.std(y) * 10 * rng.randn(n_outliers)

    # Replace some features with noise
    if noninformative > 0.0:
        n_noninformative = int(n_features * noninformative)
        noninformative_idx = rng.choice(n_features, size=n_noninformative, replace=False)
        X[:, noninformative_idx] = rng.randn(n_samples, n_noninformative) * scale[noninformative_idx][None, :]

    return X, y


@typed
def benchmark_nonlinear_datasets(n_runs: int = 3, test_size: float = 0.2):
    """Run a benchmark on manually created non-linear datasets with various smoother configurations.

    Compares different smoothers (angle with various configurations, supersmoother with
    different parameters) against a linear baseline on datasets with varying non-linearities.
    """
    print(f"\n=== Non-Linear Datasets Benchmark ({n_runs} runs) ===\n")

    rng = np.random.RandomState(42)

    # Create manual datasets with different non-linearities
    datasets = []

    # 1. Almost linear dataset with small quadratic term
    def create_almost_linear(n_samples=1000, noise=0.1):
        X = rng.uniform(-3, 3, size=(n_samples, 2))
        y = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.2 * X[:, 0] ** 2 + rng.normal(0, noise, n_samples)
        return X, y

    datasets.append(("AlmostLinear", create_almost_linear))

    # 2. Quadratic dataset
    def create_quadratic(n_samples=1000, noise=0.1):
        X = rng.uniform(-3, 3, size=(n_samples, 3))
        y = X[:, 0] ** 2 + X[:, 1] ** 2 - X[:, 2] + rng.normal(0, noise, n_samples)
        return X, y

    datasets.append(("Quadratic", create_quadratic))

    # 3. Sinusoidal dataset (non-monotonic)
    def create_sinusoidal(n_samples=1000, noise=0.2):
        X = rng.uniform(-3, 3, size=(n_samples, 2))
        y = np.sin(X[:, 0]) + 0.5 * np.cos(2 * X[:, 1]) + rng.normal(0, noise, n_samples)
        return X, y

    datasets.append(("Sinusoidal", create_sinusoidal))

    # 4. Exponential dataset (monotonic, highly non-linear)
    def create_exponential(n_samples=1000, noise=0.1):
        X = rng.uniform(-2, 2, size=(n_samples, 2))
        y = np.exp(X[:, 0]) + X[:, 1] + rng.normal(0, noise * np.exp(1), n_samples)
        return X, y

    datasets.append(("Exponential", create_exponential))

    # 5. Step function (discontinuous)
    def create_step(n_samples=1000, noise=0.1):
        X = rng.uniform(-3, 3, size=(n_samples, 2))
        y = 1.0 * (X[:, 0] > 0) + 2.0 * (X[:, 1] > 1) + rng.normal(0, noise, n_samples)
        return X, y

    datasets.append(("Step", create_step))

    # 6. Logarithmic (moderate non-linearity)
    def create_logarithmic(n_samples=1000, noise=0.5):
        X = rng.uniform(0.1, 5, size=(n_samples, 3))
        y = np.log(X[:, 0]) + 2 * np.log(X[:, 1]) - X[:, 2] + rng.normal(0, noise, n_samples)
        return X, y

    datasets.append(("Logarithmic", create_logarithmic))

    # 7. High-dimensional with interactions
    def create_interactions(n_samples=1000, noise=0.2):
        X = rng.uniform(-2, 2, size=(n_samples, 5))
        y = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] ** 2 - X[:, 4] + rng.normal(0, noise, n_samples)
        return X, y

    datasets.append(("Interactions", create_interactions))

    # Define models to test
    models = [
        ("Linear", lambda: Scaler(Linear(alpha=1e-6, better_bias=True))),
        ("Angle-lr0.5-500", lambda: Scaler(Smooth(method="angle", max_epochs=300, lr=0.5))),
        ("Angle-lr0.5-500-2", lambda: Scaler(Smooth(method="angle", n_breakpoints=2, max_epochs=300, lr=0.5))),
        ("Angle-lr0.5-500-3", lambda: Scaler(Smooth(method="angle", n_breakpoints=2, max_epochs=300, lr=0.5))),
        ("SuperSmoother-lr0.2", lambda: Scaler(Smooth(method="supersmoother", max_epochs=10, lr=0.2))),
        ("SuperSmoother-lr0.5", lambda: Scaler(Smooth(method="supersmoother", max_epochs=4, lr=0.5))),
        ("SuperSmoother-lr1.0", lambda: Scaler(Smooth(method="supersmoother", max_epochs=1, lr=1.0))),
    ]

    # Store results
    results = []

    # Run benchmarks
    for run in range(n_runs):
        print(f"Starting run {run+1}/{n_runs}...")

        for ds_name, data_fn in datasets:
            X, y = data_fn()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=run)

            for model_name, model_fn in models:
                model = model_fn()

                # Fit and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_pred)

                # Store results
                results.append({"dataset": ds_name, "model": model_name, "run": run, "test_mse": test_mse})

                print(f"  {ds_name}, {model_name}: MSE = {test_mse:.6f}")

    # Convert to dataframe and analyze
    df = pd.DataFrame(results)

    # Normalize MSE within datasets
    for dataset in df["dataset"].unique():
        dataset_mses = df[df["dataset"] == dataset]["test_mse"]
        baseline_mse = df[(df["dataset"] == dataset) & (df["model"] == "Linear")]["test_mse"].mean()
        df.loc[df["dataset"] == dataset, "rel_mse"] = df.loc[df["dataset"] == dataset, "test_mse"] / baseline_mse

    # Print summary by dataset
    print("\n=== MODEL PERFORMANCE BY DATASET ===")
    for dataset in df["dataset"].unique():
        print(f"\n--- {dataset} ---")
        dataset_df = df[df["dataset"] == dataset]

        # Group by model and get statistics
        model_stats = dataset_df.groupby("model")["test_mse"].agg(["mean", "std", "min", "count"])
        model_stats["sem"] = model_stats["std"] / np.sqrt(model_stats["count"])
        model_stats = model_stats.sort_values("mean")

        # Print model performance
        print(f"Performance (MSE, lower is better):")
        for model, stats in model_stats.iterrows():
            print(f"  {model}: {stats['mean']:.6f} ± {stats['sem']:.6f}")

        # Find best model
        best_model = model_stats.index[0]
        improvement = 100 * (1 - model_stats.loc[best_model, "mean"] / model_stats.loc["Linear", "mean"])
        print(f"Best model: {best_model} ({improvement:.2f}% improvement over Linear)")

    # Overall model comparison
    print("\n=== OVERALL MODEL PERFORMANCE ===")
    model_stats = df.groupby("model")["rel_mse"].agg(["mean", "std", "count"])
    model_stats["sem"] = model_stats["std"] / np.sqrt(model_stats["count"])
    model_stats = model_stats.sort_values("mean")

    print("Average relative MSE across all datasets (lower is better):")
    for model, stats in model_stats.iterrows():
        print(f"  {model}: {stats['mean']:.4f} ± {stats['sem']:.4f}")


@typed
def benchmark_transformed_data(n_runs: int = 5, test_size: float = 0.5):
    """Benchmark Linear regression with various scalers on transformed Gaussian data.

    This benchmark:
    1. Generates standard Gaussian regression data
    2. Applies non-linear transformations to X features
    3. Applies different non-linear transformations to the target y
    4. Tests Linear regression with different scalers
    """
    print(f"\n=== Transformed Data Benchmark ({n_runs} runs) ===\n")

    # Define transformations for X features
    x_transformations = [
        ("exp", lambda x: np.exp(x)),
        ("log", lambda x: np.sign(x) * np.log1p(np.abs(x))),
        ("sqrt", lambda x: np.sign(x) * np.sqrt(np.abs(x))),
        ("square", lambda x: np.sign(x) * x**2),
        ("cube", lambda x: x**3),
    ]

    # Define transformations for y target
    y_transformations = [
        ("exp", lambda y: np.exp(np.clip(y, -10, 10))),
        ("log", lambda y: np.sign(y) * np.log1p(np.abs(y))),
        ("sqrt", lambda y: np.sign(y) * np.sqrt(np.abs(y))),
        ("square", lambda y: np.sign(y) * y**2),
        ("sigmoid", lambda y: 1 / (1 + np.exp(-y))),
    ]

    # Models (Linear with various scalers)
    models = [
        ("SoftLinear-Stabilize-0.5", lambda: Scaler(Soft(splits=[0.5], estimator=Linear(alpha="bayes")), x_method="stabilize", y_method="stabilize")),
        ("SoftLinear-Stabilize-0.25-0.75", lambda: Scaler(Soft(splits=[0.25, 0.75], estimator=Linear(alpha="bayes")), x_method="stabilize", y_method="stabilize")),
        ("SoftLinear-Stabilize-0.1-0.3-0.7-0.9", lambda: Scaler(Soft(splits=[0.1, 0.3, 0.7, 0.9], estimator=Linear(alpha="bayes")), x_method="stabilize", y_method="stabilize")),
        # ...
        ("Linear-Stabilize", lambda: Scaler(Linear(alpha=1e-6), x_method="stabilize", y_method="stabilize")),
        ("Linear-Standard", lambda: Scaler(Linear(alpha=1e-6), x_method="standard", y_method="standard")),
        # ("Linear-Power", lambda: Scaler(Linear(alpha=1e-6), x_method="power", y_method="power")),
        ("Linear-Quantile", lambda: Scaler(Linear(alpha=1e-6), x_method="quantile-normal", y_method="quantile-normal")),
        # ("Linear-Auto", lambda: AutoScaler(Linear(alpha=1e-6))),
        ("Angle-Stabilize", lambda: Scaler(Smooth(method="angle", n_breakpoints=2, max_epochs=100, lr=0.5), x_method="stabilize", y_method="stabilize")),
        ("Angle-Standard", lambda: Scaler(Smooth(method="angle", n_breakpoints=2, max_epochs=100, lr=0.5), x_method="standard", y_method="standard")),
        # ("Angle-Power", lambda: Scaler(Smooth(method="angle", n_breakpoints=2, max_epochs=100, lr=0.5), x_method="power", y_method="power")),
        ("Angle-Quantile", lambda: Scaler(Smooth(method="angle", n_breakpoints=2, max_epochs=100, lr=0.5), x_method="quantile-normal", y_method="quantile-normal")),
        # ("Angle-Auto", lambda: AutoScaler(Smooth(method="angle", n_breakpoints=2, max_epochs=100, lr=0.5))),
    ]

    results = []

    for run in range(n_runs):
        print(f"Starting run {run+1}/{n_runs}...")

        # For each combination of X and Y transformation
        for x_name, x_transform in x_transformations:
            for y_name, y_transform in y_transformations:
                # Skip if X and Y use the same transformation type
                if x_name == y_name:
                    continue

                # Generate base regression data
                X_base, y_base = generate_regression(n_samples=2000, n_features=5, noise=1.0)

                # Apply transformations
                X = np.copy(X_base)
                for j in range(X.shape[1]):
                    X[:, j] = x_transform(X_base[:, j])
                y = y_transform(y_base / np.std(y_base))

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=run)

                for model_name, model_fn in models:
                    model = model_fn()

                    # Fit and predict
                    assert not np.any(np.isnan(X_train)), "X_train contains NaNs"
                    assert not np.any(np.isnan(y_train)), "y_train contains NaNs"
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    assert not np.any(np.isnan(y_pred)), "y_pred contains NaNs"
                    # test_mse = mean_squared_error(y_test, y_pred)
                    test_mse = np.max(np.abs(y_test - y_pred))

                    # Store results
                    results.append({"x_transform": x_name, "y_transform": y_name, "model": model_name, "run": run, "test_mse": test_mse})

                    print(f"  X:{x_name}, Y:{y_name}, {model_name}: MSE = {test_mse:.6f}")

    # Convert to dataframe and analyze
    df = pd.DataFrame(results)

    # Normalize MSE within each transformation pair
    for x_trans in df["x_transform"].unique():
        for y_trans in df["y_transform"].unique():
            mask = (df["x_transform"] == x_trans) & (df["y_transform"] == y_trans)
            if not any(mask):
                continue

            baseline_mse = df[mask & (df["model"] == "Linear-Standard")]["test_mse"].mean()
            df.loc[mask, "rel_mse"] = df.loc[mask, "test_mse"] / baseline_mse

    # Print summary by transformation pair
    print("\n=== MODEL PERFORMANCE BY TRANSFORMATION PAIR ===")
    for x_trans in df["x_transform"].unique():
        for y_trans in df["y_transform"].unique():
            if x_trans == y_trans:
                continue

            print(f"\n--- X: {x_trans}, Y: {y_trans} ---")
            mask = (df["x_transform"] == x_trans) & (df["y_transform"] == y_trans)
            trans_df = df[mask]

            # Group by model and get statistics
            model_stats = trans_df.groupby("model")["test_mse"].agg(["mean", "std", "min", "count"])
            model_stats["sem"] = model_stats["std"] / np.sqrt(model_stats["count"])
            model_stats = model_stats.sort_values("mean")

            # Print model performance
            print(f"Performance (MSE, lower is better):")
            for model, stats in model_stats.iterrows():
                print(f"  {model}: {stats['mean']:.6f} ± {stats['sem']:.6f}")

            # Find best model
            best_model = model_stats.index[0]
            improvement = 100 * (1 - model_stats.loc[best_model, "mean"] / model_stats.loc["Linear-Standard", "mean"])
            print(f"Best model: {best_model} ({improvement:.2f}% improvement over Linear-Standard)")

    # Overall model comparison
    print("\n=== OVERALL MODEL PERFORMANCE ===")
    model_stats = df.groupby("model")["rel_mse"].agg(["mean", "std", "count"])
    model_stats["sem"] = model_stats["std"] / np.sqrt(model_stats["count"])
    model_stats = model_stats.sort_values("mean")

    print("Average relative MSE across all transformations (lower is better):")
    for model, stats in model_stats.iterrows():
        print(f"  {model}: {stats['mean']:.4f} ± {stats['sem']:.4f}")

    # Performance by X transformation
    print("\n=== PERFORMANCE BY X TRANSFORMATION ===")
    x_trans_stats = df.groupby(["x_transform", "model"])["rel_mse"].mean().unstack()
    for x_trans in x_trans_stats.index:
        best_model = x_trans_stats.loc[x_trans].idxmin()
        print(f"X transformation '{x_trans}' best handled by: {best_model}")

    # Performance by Y transformation
    print("\n=== PERFORMANCE BY Y TRANSFORMATION ===")
    y_trans_stats = df.groupby(["y_transform", "model"])["rel_mse"].mean().unstack()
    for y_trans in y_trans_stats.index:
        best_model = y_trans_stats.loc[y_trans].idxmin()
        print(f"Y transformation '{y_trans}' best handled by: {best_model}")


if __name__ == "__main__":
    # test_linear_better_bias_equivalence()
    # test_nonlinear_datasets()
    # print("\n" + "-" * 50 + "\n")
    # benchmark_nonlinear_datasets()
    # print("\n" + "-" * 50 + "\n")
    benchmark_transformed_data()

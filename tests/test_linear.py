"""Tests for linear regression models."""

import numpy as np
import pandas as pd
from beartype import beartype as typed
from better_regressions import Linear, Scaler, Smooth
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
def benchmark_hyperparameters(n_runs: int = 5, test_size: float = 0.8):
    """Run a comprehensive benchmark of hyperparameter combinations.

    Instead of comparing specific combinations, this tests the main effects
    of each hyperparameter across various datasets.
    """
    print(f"\n=== Comprehensive Hyperparameter Benchmark ({n_runs} runs) ===\n")

    # Define datasets with varied characteristics
    datasets = [
        ("Small-LowNoise", lambda: generate_regression(n_samples=500, n_features=5, noise=0.1)),
        ("Small-HighNoise", lambda: generate_regression(n_samples=500, n_features=5, noise=0.5)),
        ("Medium-LowNoise", lambda: generate_regression(n_samples=1000, n_features=10, noise=0.1)),
        ("Medium-HighNoise", lambda: generate_regression(n_samples=1000, n_features=10, noise=0.5)),
        ("Large-LowNoise", lambda: generate_regression(n_samples=3000, n_features=20, noise=0.1)),
        ("Large-HighNoise", lambda: generate_regression(n_samples=3000, n_features=20, noise=0.5)),
        ("Outliers-10pct", lambda: generate_regression(n_samples=1000, n_features=10, noise=0.1, outliers=0.1)),
        ("Outliers-30pct", lambda: generate_regression(n_samples=1000, n_features=10, noise=0.1, outliers=0.3)),
        # ("MoreNoninformative", lambda: generate_regression(n_samples=500, n_features=20, noise=0.1, noninformative=0.8)),
    ]

    # Define hyperparameter values to test
    alphas = [1e-6, "bayes", "smooth-angle", "smooth-supersmoother"]
    better_bias_values = [True, False]
    y_methods = ["none", "standard"]
    use_feature_variance_values = [True, False]

    # Fixed parameters
    x_method = "standard"

    # Store all results in a dataframe
    results = []

    # Run the experiments
    for run in range(n_runs):
        print(f"Starting run {run+1}/{n_runs}...", end="\r")

        for ds_name, data_fn in datasets:
            # Generate dataset for this run
            X, y = data_fn()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=run)

            for alpha in alphas:
                for better_bias in better_bias_values:
                    for y_method in y_methods:
                        for use_feature_variance in use_feature_variance_values:
                            if y_method == "standard" and not use_feature_variance:
                                continue
                            if "smooth" in str(alpha) and (y_method != "standard" or use_feature_variance != True):
                                continue
                            # Create model
                            if alpha == "smooth-angle":
                                base_estimator = Smooth(method="angle", max_epochs=100, lr=0.5, n_points=200)
                            elif alpha == "smooth-supersmoother":
                                base_estimator = Smooth(method="supersmoother", max_epochs=1, lr=1.0, n_points=200)
                            else:
                                base_estimator = Linear(alpha=alpha, better_bias=better_bias)
                            model = Scaler(estimator=base_estimator, x_method=x_method, y_method=y_method, use_feature_variance=use_feature_variance)

                            # Train and evaluate
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            test_mse = mean_squared_error(y_test, y_pred)

                            # Store results
                            results.append(
                                {
                                    "dataset": ds_name,
                                    "run": run,
                                    "alpha": str(alpha),
                                    "better_bias": better_bias,
                                    "y_method": y_method,
                                    "use_feature_variance": use_feature_variance,
                                    "test_mse": test_mse,
                                }
                            )

    # Convert to dataframe
    df = pd.DataFrame(results)

    # Make MSE more comparable across datasets by normalizing
    for dataset in df["dataset"].unique():
        dataset_mses = df[df["dataset"] == dataset]["test_mse"]
        median_mse = dataset_mses.median()
        df.loc[df["dataset"] == dataset, "rel_mse"] = df.loc[df["dataset"] == dataset, "test_mse"] / median_mse

    # Analyze each hyperparameter's effect
    print("\n\n=== HYPERPARAMETER EFFECTS ===")

    # Function to analyze a hyperparameter
    def analyze_param(param_name):
        print(f"\n--- {param_name} ---")

        # Group by this parameter and compute statistics
        grouped = df.groupby(param_name)["rel_mse"].agg(["mean", "std", "median", "count"])
        grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"])

        # Print overall effect
        print(f"Overall effect (relative MSE, lower is better):")
        for value, stats in grouped.iterrows():
            print(f"  {value}: {stats['mean']:.4f} ± {stats['sem']:.4f}")

        if param_name != "y_method":
            return
        # Look at interaction with other parameters
        other_params = [p for p in ["alpha", "better_bias", "y_method", "use_feature_variance"] if p != param_name]
        for other in other_params:
            print(f"\nInteraction with {other}:")
            interaction = df.groupby([param_name, other])["rel_mse"].agg(["mean", "std", "count"])
            interaction["sem"] = interaction["std"] / np.sqrt(interaction["count"])

            # Format the results as a table
            for (param_val, other_val), stats in interaction.iterrows():
                print(f"  {param_val}, {other_val}: {stats['mean']:.4f} ± {stats['sem']:.4f}")

    # Analyze each hyperparameter
    for param in ["alpha", "better_bias", "y_method", "use_feature_variance"]:
        analyze_param(param)

    # Analyze dataset-specific effects
    print("\n\n=== DATASET-SPECIFIC EFFECTS ===")
    for dataset in df["dataset"].unique():
        print(f"\n--- {dataset} ---")
        dataset_df = df[df["dataset"] == dataset]

        # Find best hyperparameter combination
        best_idx = dataset_df["test_mse"].idxmin()
        best_config = dataset_df.loc[best_idx]

        print(f"Best configuration:")
        print(f"  alpha: {best_config['alpha']}")
        print(f"  better_bias: {best_config['better_bias']}")
        print(f"  y_method: {best_config['y_method']}")
        print(f"  use_feature_variance: {best_config['use_feature_variance']}")
        print(f"  test_mse: {best_config['test_mse']:.6f}")


if __name__ == "__main__":
    # test_linear_better_bias_equivalence()
    # test_nonlinear_datasets()
    # print("\n" + "-" * 50 + "\n")
    test_nonlinear_datasets()
    benchmark_nonlinear_datasets()

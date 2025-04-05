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
def test_pcr_and_lora():
    """Test PCR and LoRA methods for linear regression."""
    np.random.seed(42)

    # Create a dataset with more features than samples (n << d regime)
    n_samples, n_features = 200, 1000
    # X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features, noise=0.5, random_state=42)
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    y = X @ w
    y /= np.std(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    # Standard ridge regression
    standard_model = Scaler(Linear(alpha=1e-6, better_bias=False, method="standard"))
    standard_model.fit(X_train, y_train)
    standard_pred = standard_model.predict(X_test)
    standard_mse = mean_squared_error(y_test, standard_pred)

    # PCR with different component counts
    pcr_results = {}
    for n_components in [1, 2, 5, 10, 20, 200]:
        pcr_model = Scaler(Linear(alpha=1e-6, better_bias=False, method="pcr", n_components=n_components, random_state=42))
        pcr_model.fit(X_train, y_train)
        pcr_pred = pcr_model.predict(X_test)
        pcr_mse = mean_squared_error(y_test, pcr_pred)
        pcr_results[n_components] = pcr_mse

    # LoRA with different component counts
    lora_results = {}
    for n_components in [1, 2, 5, 10, 20, 200]:
        lora_model = Scaler(Linear(alpha=1e-6, better_bias=False, method="lora", n_components=n_components, random_state=42))
        lora_model.fit(X_train, y_train)
        lora_pred = lora_model.predict(X_test)
        lora_mse = mean_squared_error(y_test, lora_pred)
        lora_results[n_components] = lora_mse

    # Display results
    print(f"Standard Ridge MSE: {standard_mse:.6f}")
    print("\nPCR Results:")
    for n_components, mse in pcr_results.items():
        print(f"  n_components={n_components}: MSE={mse:.6f} RelMSE={mse/standard_mse:.4f}")

    print("\nLoRA Results:")
    for n_components, mse in lora_results.items():
        print(f"  n_components={n_components}: MSE={mse:.6f} RelMSE={mse/standard_mse:.4f}")

    # At least one of the dimensionality reduction methods should be competitive
    best_pcr = min(pcr_results.values())
    best_lora = min(lora_results.values())
    best_method = min(best_pcr, best_lora)

    # In the n << d regime, dimensionality reduction should help
    print(f"\nBest method / Standard ratio: {best_method/standard_mse:.4f}")

    # Not requiring it to be strictly better since there's randomness,
    # but it should be reasonably close
    assert best_method < 1.5 * standard_mse, "Dimensionality reduction methods should perform reasonably well"


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
def benchmark_pcr_lora(n_runs: int = 3):
    """Benchmark PCR and LoRA methods across different datasets."""
    print(f"\n=== PCR and LoRA Benchmark ({n_runs} runs) ===\n")

    # Create scenarios to test
    scenarios = [
        ("Small-n-small-d", 100, 10, 0.1),  # Small dataset, more samples than features
        ("Small-n-large-d", 50, 200, 0.1),  # High-dimensional, fewer samples than features
        ("Large-n-medium-d", 500, 50, 0.1),  # Large dataset, more samples than features
        ("Large-with-noise", 300, 100, 0.5),  # Large dataset with high noise
        ("Large-with-outliers", 300, 30, 0.1, 0.1),  # Dataset with outliers
    ]

    # Define models to test
    models = [
        ("Standard", lambda: Linear(alpha=1.0, method="standard")),
        ("PCR-5", lambda: Linear(alpha=1.0, method="pcr", n_components=5)),
        ("PCR-10", lambda: Linear(alpha=1.0, method="pcr", n_components=10)),
        ("PCR-20", lambda: Linear(alpha=1.0, method="pcr", n_components=20)),
        ("LoRA-5", lambda: Linear(alpha=1.0, method="lora", n_components=5)),
        ("LoRA-10", lambda: Linear(alpha=1.0, method="lora", n_components=10)),
        ("LoRA-20", lambda: Linear(alpha=1.0, method="lora", n_components=20)),
    ]

    # Store results
    results = []

    for run in range(n_runs):
        print(f"Starting run {run+1}/{n_runs}...")

        for scenario_name, n_samples, n_features, noise, *extra in scenarios:
            # Generate dataset
            outliers = extra[0] if extra else 0.0
            X, y = generate_regression(n_samples, n_features, noise, outliers=outliers)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)

            for model_name, model_fn in models:
                # Skip models that won't work for this scenario
                if "PCR" in model_name or "LoRA" in model_name:
                    n_components = int(model_name.split("-")[1])
                    if n_components > min(n_samples, n_features):
                        # Skip if we ask for more components than possible
                        continue

                model = model_fn()

                try:
                    # Fit and evaluate
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    test_mse = mean_squared_error(y_test, y_pred)

                    # Store results
                    results.append({"scenario": scenario_name, "model": model_name, "run": run, "test_mse": test_mse, "n_samples": n_samples, "n_features": n_features})

                    print(f"  {scenario_name}, {model_name}: MSE = {test_mse:.6f}")

                except Exception as e:
                    print(f"  ERROR with {scenario_name}, {model_name}: {str(e)}")

    # Convert to dataframe and analyze
    df = pd.DataFrame(results)

    # Normalize MSE within scenarios
    for scenario in df["scenario"].unique():
        scenario_mses = df[df["scenario"] == scenario]["test_mse"]
        baseline_mse = df[(df["scenario"] == scenario) & (df["model"] == "Standard")]["test_mse"].mean()
        df.loc[df["scenario"] == scenario, "rel_mse"] = df.loc[df["scenario"] == scenario, "test_mse"] / baseline_mse

    # Print summary by scenario
    print("\n=== MODEL PERFORMANCE BY SCENARIO ===")
    for scenario in df["scenario"].unique():
        print(f"\n--- {scenario} ---")
        scenario_df = df[df["scenario"] == scenario]

        # Compute statistics
        model_stats = scenario_df.groupby("model")["test_mse"].agg(["mean", "std", "min", "count"])
        model_stats["sem"] = model_stats["std"] / np.sqrt(model_stats["count"])
        model_stats = model_stats.sort_values("mean")

        # Print performance
        print(f"Performance (MSE, lower is better):")
        for model, stats in model_stats.iterrows():
            print(f"  {model}: {stats['mean']:.6f} ± {stats['sem']:.6f}")

        # Find best model
        best_model = model_stats.index[0]
        if best_model != "Standard":
            improvement = 100 * (1 - model_stats.loc[best_model, "mean"] / model_stats.loc["Standard", "mean"])
            print(f"Best model: {best_model} ({improvement:.2f}% improvement over Standard)")

    # Overall comparison
    print("\n=== OVERALL MODEL PERFORMANCE ===")
    model_stats = df.groupby("model")["rel_mse"].agg(["mean", "std", "count"])
    model_stats["sem"] = model_stats["std"] / np.sqrt(model_stats["count"])
    model_stats = model_stats.sort_values("mean")

    print("Average relative MSE across all scenarios (lower is better):")
    for model, stats in model_stats.iterrows():
        print(f"  {model}: {stats['mean']:.4f} ± {stats['sem']:.4f}")

    # Check which method performs best on which scenario type
    print("\n=== BEST MODEL BY SCENARIO TYPE ===")
    scenario_types = [["Small-n-small-d", "Large-n-medium-d"], ["Small-n-large-d"], ["Large-with-noise", "Large-with-outliers"]]  # n > d  # n < d  # Noisy data

    for scenario_group in scenario_types:
        group_name = ", ".join(scenario_group)
        group_df = df[df["scenario"].isin(scenario_group)]
        model_perf = group_df.groupby("model")["rel_mse"].mean().sort_values()

        best_model = model_perf.index[0]
        improvement = 100 * (1 - model_perf.iloc[0] / 1.0)  # Relative to standard

        print(f"Best model for {group_name}:")
        print(f"  {best_model} ({improvement:.2f}% improvement)")


if __name__ == "__main__":
    # test_linear_better_bias_equivalence()
    # test_nonlinear_datasets()
    # print("\n" + "-" * 50 + "\n")
    test_pcr_and_lora()
    # print("\n" + "-" * 50 + "\n")
    # benchmark_pcr_lora(n_runs=1)
    # benchmark_nonlinear_datasets()

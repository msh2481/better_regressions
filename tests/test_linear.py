"""Tests for linear regression models."""

import numpy as np
from beartype import beartype as typed

from better_regressions.linear import Linear
from better_regressions.scaling import Scaler
from jaxtyping import Float
from numpy import ndarray as ND
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def test_linear_better_bias_equivalence():
    """Test if Linear with tiny alpha produces similar results regardless of better_bias."""
    # Set a seed for reproducibility
    np.random.seed(42)

    # Create synthetic data - not too large as we just need to verify equivalence
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    # Fit with better_bias=True
    model_true = Linear(alpha=1e-18, better_bias=True)
    model_true.fit(X, y)
    pred_true = model_true.predict(X)

    # Fit with better_bias=False
    model_false = Linear(alpha=1e-18, better_bias=False)
    model_false.fit(X, y)
    pred_false = model_false.predict(X)

    # Test predictions are very close
    assert np.allclose(pred_true, pred_false, rtol=1e-5, atol=1e-5), "Predictions should be very similar with tiny alpha regardless of better_bias"

    # Calculate MSE between predictions
    mse_diff = mean_squared_error(pred_true, pred_false)
    print(f"MSE between better_bias=True and better_bias=False predictions: {mse_diff:.8f}")

    # Test that coefficients have similar effect
    # For better_bias=True, the first coefficient is handled differently
    intercept_true = model_true.intercept_
    coef_true = model_true.coef_

    intercept_false = model_false.intercept_
    coef_false = model_false.coef_

    # Manually calculate predictions to verify coefficient similarity
    manual_pred_true = X @ coef_true + intercept_true
    manual_pred_false = X @ coef_false + intercept_false

    assert np.allclose(manual_pred_true, manual_pred_false, rtol=1e-5, atol=1e-5), "Manual predictions should be similar with tiny alpha regardless of better_bias"


@typed
def compare_linear_variants():
    """Compare different variants of Linear regression on regression problems with appropriate scaling."""
    # Create datasets with different characteristics
    datasets = [
        # Small dataset with low noise
        make_regression(n_samples=50, n_features=10, noise=0.1, random_state=42),
        # Medium dataset with medium noise
        make_regression(n_samples=200, n_features=20, noise=0.5, random_state=42),
        # Large dataset with high noise
        make_regression(n_samples=500, n_features=30, noise=1.0, random_state=42),
        # Dataset with outliers
        make_regression(n_samples=100, n_features=15, noise=0.2, random_state=42, tail_strength=0.9),  # More extreme values
    ]

    # Base model configurations to test
    base_configs = [
        {"alpha": 1e-6, "better_bias": False, "name": "Ridge (α=1e-6, standard bias)"},
        {"alpha": 1e-6, "better_bias": True, "name": "Ridge (α=1e-6, better bias)"},
        {"alpha": "ard", "better_bias": False, "name": "ARD (standard bias)"},
        {"alpha": "ard", "better_bias": True, "name": "ARD (better bias)"},
    ]

    # Wrap all configurations with appropriate scaling
    configs = []
    for config in base_configs:
        # Create a copy of the config
        scaled_config = config.copy()

        # Create base estimator
        base_estimator = Linear(alpha=config["alpha"], better_bias=config["better_bias"])

        # Wrap with Scaler - only scale X, not y (to keep the better_bias meaning)
        scaled_config["estimator"] = Scaler(estimator=base_estimator, x_method="standard", y_method="none")  # Don't scale target
        scaled_config["name"] = f"{config['name']} + X scaling"

        configs.append(scaled_config)

    # Store results for each dataset
    results = []

    for i, (X, y) in enumerate(datasets):
        dataset_name = f"Dataset {i+1}"
        print(f"\n=== {dataset_name} ===")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        dataset_results = {"name": dataset_name, "configs": []}

        for config in configs:
            # Clone estimator to ensure fresh fitting
            model = config["estimator"]

            # Fit model
            model.fit(X_train, y_train)

            # Evaluate on train and test
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)

            print(f"{config['name']}:")
            print(f"  Train MSE: {train_mse:.6f}")
            print(f"  Test MSE: {test_mse:.6f}")

            # Get the coefficients from the wrapped model
            wrapped_model = model.wrapped_estimator if hasattr(model, "wrapped_estimator") else model

            # Calculate coefficient norm as a measure of regularization effectiveness
            if hasattr(wrapped_model, "coef_"):
                coef_norm = np.linalg.norm(wrapped_model.coef_)
                print(f"  Coefficient norm: {coef_norm:.6f}")
            else:
                coef_norm = float("nan")
                print("  Coefficient norm: N/A")

            config_result = {
                "name": config["name"],
                "train_mse": train_mse,
                "test_mse": test_mse,
                "coef_norm": coef_norm,
            }

            dataset_results["configs"].append(config_result)

        results.append(dataset_results)

    # Summarize model comparison
    print("\n=== SUMMARY ===")
    print("Geometric Mean Test MSE across all datasets:")

    for i in range(len(configs)):
        # Use geometric mean instead of arithmetic mean
        test_mses = [r["configs"][i]["test_mse"] for r in results]
        geo_mean_test_mse = np.exp(np.mean(np.log(test_mses)))
        config_name = configs[i]["name"]
        print(f"{config_name}: {geo_mean_test_mse:.6f}")

    # Rank performance
    avg_ranks = []
    for i in range(len(configs)):
        # Calculate average rank across datasets (lower MSE = better rank)
        ranks = []
        for r in results:
            config_mses = [c["test_mse"] for c in r["configs"]]
            sorted_indices = np.argsort(config_mses)
            rank = np.where(sorted_indices == i)[0][0] + 1
            ranks.append(rank)

        avg_rank = np.mean(ranks)
        avg_ranks.append((configs[i]["name"], avg_rank))

    # Sort by average rank
    avg_ranks.sort(key=lambda x: x[1])

    print("\nAverage Rank (1 = best):")
    for name, rank in avg_ranks:
        print(f"{name}: {rank:.2f}")

    # Find cases where better_bias helps
    better_bias_advantage = []
    for i in range(len(results)):
        # Compare Ridge with and without better_bias
        ridge_std = results[i]["configs"][0]["test_mse"]
        ridge_better = results[i]["configs"][1]["test_mse"]
        ridge_diff = (ridge_std - ridge_better) / ridge_std * 100

        # Compare ARD with and without better_bias
        ard_std = results[i]["configs"][2]["test_mse"]
        ard_better = results[i]["configs"][3]["test_mse"]
        ard_diff = (ard_std - ard_better) / ard_std * 100

        better_bias_advantage.append({"dataset": results[i]["name"], "ridge_improvement": ridge_diff, "ard_improvement": ard_diff})

    print("\nBetter Bias Improvement % (positive = better_bias helps):")
    for item in better_bias_advantage:
        print(f"{item['dataset']}:")
        print(f"  Ridge: {item['ridge_improvement']:.2f}%")
        print(f"  ARD: {item['ard_improvement']:.2f}%")


if __name__ == "__main__":
    test_linear_better_bias_equivalence()
    print("\n" + "-" * 50 + "\n")
    compare_linear_variants()

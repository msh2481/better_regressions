"""Tests for linear regression models."""

import numpy as np
from beartype import beartype as typed

from better_regressions.linear import Linear
from better_regressions.scaling import Scaler
from jaxtyping import Float
from numpy import ndarray as ND
from sklearn.datasets import make_regression
from sklearn.linear_model import ARDRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
def compare_linear_variants(n_runs: int = 100):
    """Compare different variants of Linear regression over multiple runs with standard scaling."""
    print(f"\n=== Linear Variant Comparison ({n_runs} runs) ===")
    # Create datasets with different characteristics
    datasets = [
        # Small dataset with low noise
        ("Small Low Noise", lambda r: make_regression(n_samples=50, n_features=10, noise=0.1, random_state=r)),
        # Medium dataset with medium noise
        ("Medium Medium Noise", lambda r: make_regression(n_samples=200, n_features=20, noise=0.5, random_state=r)),
        # Large dataset with high noise
        ("Large High Noise", lambda r: make_regression(n_samples=500, n_features=30, noise=1.0, random_state=r)),
        # Dataset with outliers
        ("Outliers", lambda r: make_regression(n_samples=100, n_features=15, noise=0.2, random_state=r, tail_strength=0.9)),
    ]

    # Base model configurations to test
    base_configs = [
        {"alpha": 1e-6, "better_bias": False, "name": "Ridge (α=1e-6, std bias)"},
        {"alpha": 1e-6, "better_bias": True, "name": "Ridge (α=1e-6, better bias)"},
        {"alpha": "ard", "better_bias": False, "name": "ARD (std bias)"},
        {"alpha": "ard", "better_bias": True, "name": "ARD (better bias)"},
    ]

    # Wrap all configurations with standard scaling for X and Y
    configs = []
    for config in base_configs:
        scaled_config = config.copy()
        base_estimator = Linear(alpha=config["alpha"], better_bias=config["better_bias"])
        scaled_config["estimator"] = Scaler(estimator=base_estimator, x_method="standard", y_method="standard", use_feature_variance=True)
        scaled_config["name"] = f"{config['name']} + StdXY scaling"
        configs.append(scaled_config)

    # Store results across runs
    all_results = {config["name"]: {ds_name: [] for ds_name, _ in datasets} for config in configs}

    for run in range(n_runs):
        print(f"Starting run {run + 1}/{n_runs}...", end="\r")
        run_random_state = np.random.randint(0, 1000000)

        for ds_name, data_fn in datasets:
            X, y = data_fn(run_random_state)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=run_random_state)

            for config in configs:
                model = config["estimator"]
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_pred_test)
                all_results[config["name"]][ds_name].append(test_mse)

    # Summarize results with quantiles
    print("\n=== SUMMARY (Geometric Mean Test MSE over datasets, q5-q50-q95 over runs) ===")
    summary_table = []
    for config in configs:
        config_name = config["name"]
        geo_means_per_run = []
        for run in range(n_runs):
            run_mses = [all_results[config_name][ds_name][run] for ds_name, _ in datasets]
            geo_means_per_run.append(np.exp(np.mean(np.log(run_mses))))

        q5, q50, q95 = np.percentile(geo_means_per_run, [5, 50, 95])
        summary_table.append({"name": config_name, "q5": q5, "q50": q50, "q95": q95})
        print(f"{config_name}: {q50:.6f} ({q5:.6f} - {q95:.6f})")


@typed
def compare_scaling_variants(n_runs: int = 100):
    """Compare ARD with different scaling methods and better_bias over multiple runs."""
    print(f"\n=== ARD Scaling Variant Comparison ({n_runs} runs) ===")
    # Datasets (same as compare_linear_variants)
    datasets = [
        ("Small Low Noise", lambda r: make_regression(n_samples=50, n_features=10, noise=0.1, random_state=r)),
        ("Medium Medium Noise", lambda r: make_regression(n_samples=200, n_features=20, noise=0.5, random_state=r)),
        ("Large High Noise", lambda r: make_regression(n_samples=500, n_features=30, noise=1.0, random_state=r)),
        ("Outliers", lambda r: make_regression(n_samples=100, n_features=15, noise=0.2, random_state=r, tail_strength=0.9)),
    ]

    # Define scaling configurations
    scaling_configs = [
        {"x_method": "none", "y_method": "none", "label": "None"},
        {"x_method": "standard", "y_method": "none", "label": "StdX"},
        {"x_method": "none", "y_method": "standard", "label": "StdY"},
        {"x_method": "standard", "y_method": "standard", "label": "StdXY"},
    ]

    # Generate all 8 model variants
    configs = []
    for use_feat_var in [False, True]:
        feat_var_label = "+ FeatVar" if use_feat_var else ""
        for better_bias in [False, True]:
            bias_label = "better bias" if better_bias else "std bias"
            for sc in scaling_configs:
                base_estimator = Linear(alpha="ard", better_bias=better_bias)
                estimator = Scaler(estimator=base_estimator, x_method=sc["x_method"], y_method=sc["y_method"], use_feature_variance=use_feat_var)
                configs.append({"name": f"ARD ({bias_label}, {sc['label']}{feat_var_label})", "estimator": estimator})

    # Store results across runs
    all_results = {config["name"]: {ds_name: [] for ds_name, _ in datasets} for config in configs}

    for run in range(n_runs):
        print(f"Starting run {run + 1}/{n_runs}...", end="\r")
        run_random_state = np.random.randint(0, 1000000)

        for ds_name, data_fn in datasets:
            X, y = data_fn(run_random_state)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=run_random_state)

            for config in configs:
                model = config["estimator"]
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_pred_test)
                all_results[config["name"]][ds_name].append(test_mse)

    # Summarize results with quantiles
    print("\n=== SUMMARY (Geometric Mean Test MSE over datasets, q5-q50-q95 over runs) ===")
    summary_table = []
    for config in configs:
        config_name = config["name"]
        geo_means_per_run = []
        for run in range(n_runs):
            run_mses = [all_results[config_name][ds_name][run] for ds_name, _ in datasets]
            geo_means_per_run.append(np.exp(np.mean(np.log(run_mses))))

        q5, q50, q95 = np.percentile(geo_means_per_run, [5, 50, 95])
        summary_table.append({"name": config_name, "q5": q5, "q50": q50, "q95": q95})
        print(f"{config_name}: {q50:.6f} ({q5:.6f} - {q95:.6f})")


@typed
def compare_feature_variance_norm(n_runs: int = 100):
    """Compare Ridge regression with and without feature variance normalization."""
    print(f"\n=== Feature Variance Normalization Comparison (Ridge, {n_runs} runs) ===")
    datasets = [
        ("Small Low Noise", lambda r: make_regression(n_samples=50, n_features=10, noise=0.1, random_state=r)),
        ("Medium Medium Noise", lambda r: make_regression(n_samples=200, n_features=20, noise=0.5, random_state=r)),
        ("Large High Noise", lambda r: make_regression(n_samples=500, n_features=30, noise=1.0, random_state=r)),
        ("Outliers", lambda r: make_regression(n_samples=100, n_features=15, noise=0.2, random_state=r, tail_strength=0.9)),
    ]

    configs = []
    alpha = 1e-3  # A moderate alpha for Ridge
    for use_feat_var in [False, True]:
        label = "+ FeatVar" if use_feat_var else "No FeatVar"
        # Using std bias and StdXY scaling as a base
        base_estimator = Linear(alpha="ard", better_bias=False)
        estimator = Scaler(estimator=base_estimator, x_method="standard", y_method="standard", use_feature_variance=use_feat_var)
        configs.append({"name": f"ARD (α={alpha}, StdXY, {label})", "estimator": estimator})
        # Now with only StdX scaling
        estimator = Scaler(estimator=base_estimator, x_method="standard", y_method="none", use_feature_variance=use_feat_var)
        configs.append({"name": f"ARD (α={alpha}, StdX, {label})", "estimator": estimator})
        # Now with only StdY scaling
        estimator = Scaler(estimator=base_estimator, x_method="none", y_method="standard", use_feature_variance=use_feat_var)
        configs.append({"name": f"ARD (α={alpha}, StdY, {label})", "estimator": estimator})
        # Now without scaling
        estimator = Scaler(estimator=base_estimator, x_method="none", y_method="none", use_feature_variance=use_feat_var)
        configs.append({"name": f"ARD (α={alpha}, No scaling, {label})", "estimator": estimator})

    all_results = {config["name"]: {ds_name: [] for ds_name, _ in datasets} for config in configs}

    for run in range(n_runs):
        print(f"Starting run {run + 1}/{n_runs}...", end="\r")
        run_random_state = np.random.randint(0, 1000000)

        for ds_name, data_fn in datasets:
            X, y = data_fn(run_random_state)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=run_random_state)

            for config in configs:
                model = config["estimator"]
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_pred_test)
                all_results[config["name"]][ds_name].append(test_mse)

    print("\n=== SUMMARY (Geometric Mean Test MSE over datasets, q5-q50-q95 over runs) ===")
    summary_table = []
    for config in configs:
        config_name = config["name"]
        geo_means_per_run = []
        for run in range(n_runs):
            run_mses = [all_results[config_name][ds_name][run] for ds_name, _ in datasets]
            geo_means_per_run.append(np.exp(np.mean(np.log(run_mses))))

        q5, q50, q95 = np.percentile(geo_means_per_run, [5, 50, 95])
        summary_table.append({"name": config_name, "q5": q5, "q50": q50, "q95": q95})
        print(f"{config_name}: {q50:.6f} ({q5:.6f} - {q95:.6f})")

    print("\n=== DATASET-SPECIFIC RESULTS (Median Test MSE over runs) ===")
    for ds_name, _ in datasets:
        # Print distribution stats of y for this dataset
        y_means = []
        y_stds = []
        n_samples = 10

        print(f"\nDistribution stats for {ds_name} dataset (over {n_samples} generations):")
        for i in range(n_samples):
            _, data_fn = next((d for d in datasets if d[0] == ds_name), (None, None))
            if data_fn:
                seed = np.random.randint(0, 1000000)
                _, y_sample = data_fn(seed)
                y_means.append(np.mean(y_sample))
                y_stds.append(np.std(y_sample))

        if y_means and y_stds:
            print(f"  Mean of means: {np.mean(y_means):.6f}")
            print(f"  Mean of stds: {np.mean(y_stds):.6f}")

        print(f"\n{ds_name} dataset:")
        ds_results = []
        for config in configs:
            config_name = config["name"]
            results = all_results[config_name][ds_name]
            median_mse = np.median(results)
            q5, q95 = np.percentile(results, [5, 95])
            ds_results.append({"name": config_name, "median": median_mse, "q5": q5, "q95": q95})

        for result in ds_results:
            print(f"{result['name']}: {result['median']:.6f} ({result['q5']:.6f} - {result['q95']:.6f})")


@typed
def analyze_ard_regularization():
    """Analyze why ARD may over-regularize on Dataset 1 and how better_bias affects it."""
    print("\n=== ARD Regularization Analysis on Dataset 1 ===")

    # Recreate Dataset 1 with the same seed
    # np.random.seed(42)
    random_state = np.random.randint(0, 1000000)
    print(f"Random state: {random_state}")
    X, y = make_regression(n_samples=50, n_features=10, noise=1.0, random_state=random_state)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    # Apply standard scaling (without centering) to X
    scaler_x = StandardScaler(with_mean=False, with_std=True)
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    scaler_y = StandardScaler(with_mean=False, with_std=False)  # Identity for y
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Test various Ridge alpha values to find optimal regularization
    alphas = [1e-18, 1e-9, 1e-6, 1e-3, 1e-1, 1, 10, 100]
    ridge_results = []

    print("\nRidge regression with different alpha values (X scaled):")
    for alpha in alphas:
        # Standard bias
        ridge_std = Linear(alpha=alpha, better_bias=False)
        ridge_std.fit(X_train_scaled, y_train_scaled)  # Fit on scaled X, unscaled y
        y_test_pred_std_scaled = ridge_std.predict(X_test_scaled)
        y_test_pred_std = scaler_y.inverse_transform(y_test_pred_std_scaled.reshape(-1, 1)).ravel()  # Inverse transform is identity
        test_mse_std = mean_squared_error(y_test, y_test_pred_std)
        coef_norm_std = np.linalg.norm(ridge_std.coef_)

        # Better bias
        ridge_better = Linear(alpha=alpha, better_bias=True)
        ridge_better.fit(X_train_scaled, y_train_scaled)  # Fit on scaled X, unscaled y
        y_test_pred_better_scaled = ridge_better.predict(X_test_scaled)
        y_test_pred_better = scaler_y.inverse_transform(y_test_pred_better_scaled.reshape(-1, 1)).ravel()  # Inverse transform is identity
        test_mse_better = mean_squared_error(y_test, y_test_pred_better)
        coef_norm_better = np.linalg.norm(ridge_better.coef_)

        ridge_results.append({"alpha": alpha, "standard_bias_mse": test_mse_std, "better_bias_mse": test_mse_better, "standard_bias_norm": coef_norm_std, "better_bias_norm": coef_norm_better})

        print(f"Alpha = {alpha}:")
        print(f"  Standard bias - Test MSE: {test_mse_std:.6f}, Coef norm: {coef_norm_std:.6f}")
        print(f"  Better bias   - Test MSE: {test_mse_better:.6f}, Coef norm: {coef_norm_better:.6f}")

    # Find optimal alpha
    best_ridge = min(ridge_results, key=lambda x: x["standard_bias_mse"])
    print(f"\nBest Ridge alpha: {best_ridge['alpha']} with MSE: {best_ridge['standard_bias_mse']:.6f}")

    # Analyze ARD behavior
    print("\nARD regression analysis (X scaled):")

    # Standard ARD
    ard_std = Linear(alpha="ard", better_bias=False)
    ard_std.fit(X_train_scaled, y_train_scaled)
    y_test_pred_std_scaled = ard_std.predict(X_test_scaled)
    y_test_pred_std = scaler_y.inverse_transform(y_test_pred_std_scaled.reshape(-1, 1)).ravel()
    test_mse_std = mean_squared_error(y_test, y_test_pred_std)
    coef_norm_std = np.linalg.norm(ard_std.coef_)

    # Better bias ARD
    ard_better = Linear(alpha="ard", better_bias=True)
    ard_better.fit(X_train_scaled, y_train_scaled)
    y_test_pred_better_scaled = ard_better.predict(X_test_scaled)
    y_test_pred_better = scaler_y.inverse_transform(y_test_pred_better_scaled.reshape(-1, 1)).ravel()
    test_mse_better = mean_squared_error(y_test, y_test_pred_better)
    coef_norm_better = np.linalg.norm(ard_better.coef_)

    print(f"ARD standard bias - Test MSE: {test_mse_std:.6f}, Coef norm: {coef_norm_std:.6f}")
    print(f"ARD better bias   - Test MSE: {test_mse_better:.6f}, Coef norm: {coef_norm_better:.6f}")

    rnd = np.random.randn()  # Placeholder, analysis part is commented out
    # TODO: Uncomment and fix ARD analysis part if needed
    # # Analyze feature relevance determined by ARD
    # print("\nARD feature relevance (inverse variance of coefficients):")
    # # Need to access the underlying ARDRegression model to get lambda_
    # if hasattr(ard_std, 'wrapped_estimator') and isinstance(ard_std.wrapped_estimator, ARDRegression):
    #     lambdas_std = ard_std.wrapped_estimator.lambda_
    #     print("Standard ARD relevance scores:")
    #     for i, l in enumerate(lambdas_std):
    #         print(f"  Feature {i}: {1/l:.6f}")
    #     avg_alpha_std = np.mean(lambdas_std)
    #     print(f"  Standard bias avg lambda: {avg_alpha_std:.6f}")
    # else:
    #     print("Could not access lambda_ for standard ARD")

    # if hasattr(ard_better, 'wrapped_estimator') and isinstance(ard_better.wrapped_estimator, ARDRegression):
    #     lambdas_better = ard_better.wrapped_estimator.lambda_
    #     print("\nBetter bias ARD relevance scores:")
    #     for i, l in enumerate(lambdas_better):
    #         name = "Bias" if i == 0 else f"Feature {i-1}"
    #         print(f"  {name}: {1/l:.6f}")
    #     avg_alpha_better = np.mean(lambdas_better)
    #     print(f"  Better bias avg lambda: {avg_alpha_better:.6f}")
    # else:
    #     print("Could not access lambda_ for better bias ARD")

    # print(f"\nBest Ridge alpha for comparison: {best_ridge['alpha']}")


if __name__ == "__main__":
    # test_linear_better_bias_equivalence()
    # print("\n" + "-" * 50 + "\n")
    # compare_linear_variants()
    # print("\n" + "-" * 50 + "\n")
    # compare_scaling_variants()
    # print("\n" + "-" * 50 + "\n")
    compare_feature_variance_norm()
    # print("\n" + "-" * 50 + "\n")
    # analyze_ard_regularization()

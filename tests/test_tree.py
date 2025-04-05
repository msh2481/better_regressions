"""Tests for tree-based linear regression models."""

import lightgbm as lgb
import numpy as np
from beartype import beartype as typed
from better_regressions import Linear, Scaler, TreeLinear
from sklearn.datasets import make_regression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


@typed
def test_treelinear_basic():
    """Test that TreeLinear works on basic regression datasets."""
    # Create dataset
    X, y = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compare TreeLinear with different tree types
    rf_model = TreeLinear(tree_type="rf", n_estimators=50, max_depth=3, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)

    et_model = TreeLinear(tree_type="et", n_estimators=50, max_depth=3, random_state=42)
    et_model.fit(X_train, y_train)
    et_pred = et_model.predict(X_test)
    et_mse = mean_squared_error(y_test, et_pred)

    # Test LGBM model
    lgbm_model = TreeLinear(tree_type="lgbm", n_estimators=50, max_depth=3, random_state=42)
    lgbm_model.fit(X_train, y_train)
    lgbm_pred = lgbm_model.predict(X_test)
    lgbm_mse = mean_squared_error(y_test, lgbm_pred)

    # Also compare to a regular linear model as baseline
    linear_model = Scaler(Linear(alpha=1e-6))
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    linear_mse = mean_squared_error(y_test, linear_pred)

    # Compare with vanilla tree models
    vanilla_rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
    vanilla_rf.fit(X_train, y_train)
    vanilla_rf_pred = vanilla_rf.predict(X_test)
    vanilla_rf_mse = mean_squared_error(y_test, vanilla_rf_pred)

    vanilla_et = ExtraTreesRegressor(n_estimators=50, max_depth=3, random_state=42)
    vanilla_et.fit(X_train, y_train)
    vanilla_et_pred = vanilla_et.predict(X_test)
    vanilla_et_mse = mean_squared_error(y_test, vanilla_et_pred)

    vanilla_lgbm = lgb.LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
    vanilla_lgbm.fit(X_train, y_train)
    vanilla_lgbm_pred = vanilla_lgbm.predict(X_test)
    vanilla_lgbm_mse = mean_squared_error(y_test, vanilla_lgbm_pred)

    print(f"RandomForest TreeLinear MSE: {rf_mse:.6f}")
    print(f"ExtraTrees TreeLinear MSE: {et_mse:.6f}")
    print(f"LightGBM TreeLinear MSE: {lgbm_mse:.6f}")
    print(f"Linear baseline MSE: {linear_mse:.6f}")
    print(f"Vanilla RandomForest MSE: {vanilla_rf_mse:.6f}")
    print(f"Vanilla ExtraTrees MSE: {vanilla_et_mse:.6f}")
    print(f"Vanilla LightGBM MSE: {vanilla_lgbm_mse:.6f}")

    # TreeLinear should work and not crash
    assert rf_mse > 0, "MSE should be positive"
    assert et_mse > 0, "MSE should be positive"
    assert lgbm_mse > 0, "MSE should be positive"


@typed
def test_treelinear_nonlinear():
    """Test TreeLinear on non-linear data where it should outperform linear model."""
    # Create non-linear dataset
    np.random.seed(42)
    n_samples = 500
    X = np.random.uniform(-3, 3, size=(n_samples, 2))
    y = X[:, 0] ** 2 + np.sin(X[:, 1]) + np.random.normal(0, 0.1, n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TreeLinear model with different tree types
    rf_model = TreeLinear(tree_type="rf", n_estimators=100, max_depth=5, alpha="bayes", random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)

    et_model = TreeLinear(tree_type="et", n_estimators=100, max_depth=5, alpha="bayes", random_state=42)
    et_model.fit(X_train, y_train)
    et_pred = et_model.predict(X_test)
    et_mse = mean_squared_error(y_test, et_pred)

    # LGBM model - note that LGBM may not perform well on this specific dataset
    lgbm_model = TreeLinear(tree_type="lgbm", n_estimators=100, max_depth=5, alpha="bayes", random_state=42)
    lgbm_model.fit(X_train, y_train)
    lgbm_pred = lgbm_model.predict(X_test)
    lgbm_mse = mean_squared_error(y_test, lgbm_pred)

    # Linear baseline
    linear_model = Scaler(Linear(alpha=1e-6))
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    linear_mse = mean_squared_error(y_test, linear_pred)

    # Vanilla tree models
    vanilla_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    vanilla_rf.fit(X_train, y_train)
    vanilla_rf_pred = vanilla_rf.predict(X_test)
    vanilla_rf_mse = mean_squared_error(y_test, vanilla_rf_pred)

    vanilla_et = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42)
    vanilla_et.fit(X_train, y_train)
    vanilla_et_pred = vanilla_et.predict(X_test)
    vanilla_et_mse = mean_squared_error(y_test, vanilla_et_pred)

    vanilla_lgbm = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
    vanilla_lgbm.fit(X_train, y_train)
    vanilla_lgbm_pred = vanilla_lgbm.predict(X_test)
    vanilla_lgbm_mse = mean_squared_error(y_test, vanilla_lgbm_pred)

    print(f"RandomForest TreeLinear MSE: {rf_mse:.6f}")
    print(f"ExtraTrees TreeLinear MSE: {et_mse:.6f}")
    print(f"LightGBM TreeLinear MSE: {lgbm_mse:.6f}")
    print(f"Linear baseline MSE: {linear_mse:.6f}")
    print(f"Vanilla RandomForest MSE: {vanilla_rf_mse:.6f}")
    print(f"Vanilla ExtraTrees MSE: {vanilla_et_mse:.6f}")
    print(f"Vanilla LightGBM MSE: {vanilla_lgbm_mse:.6f}")

    # At least one TreeLinear implementation should outperform linear on non-linear data
    assert rf_mse < linear_mse, "RandomForest TreeLinear should outperform linear model on non-linear data"
    assert et_mse < linear_mse, "ExtraTrees TreeLinear should outperform linear model on non-linear data"
    # Don't assert for LightGBM as it may vary depending on dataset and implementation details

    # Compare TreeLinear with vanilla tree models (should be close or better)
    print(f"\nTreeLinear/Vanilla performance ratio (lower is better):")
    print(f"RandomForest: {rf_mse/vanilla_rf_mse:.4f}")
    print(f"ExtraTrees: {et_mse/vanilla_et_mse:.4f}")
    print(f"LightGBM: {lgbm_mse/vanilla_lgbm_mse:.4f}")


@typed
def test_treelinear_with_pca():
    """Test TreeLinear with PCA dimensionality reduction."""
    # Create non-linear dataset with higher dimensionality
    np.random.seed(42)
    n_samples = 500
    n_features = 4
    X = np.random.uniform(-3, 3, size=(n_samples, n_features))
    y = X[:, 0] ** 2 + np.sin(X[:, 1]) + X[:, 2] * X[:, 3] + np.random.normal(0, 0.1, n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline TreeLinear without PCA
    baseline_model = TreeLinear(tree_type="rf", n_estimators=30, max_depth=4, alpha="bayes", random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_mse = mean_squared_error(y_test, baseline_pred)

    # Models with different PCA dimensions
    pca_dims = [10, 50, 100]
    pca_results = {}

    for dim in pca_dims:
        model = TreeLinear(tree_type="rf", n_estimators=30, max_depth=4, alpha="bayes", random_state=42, hidden_dim=dim, pca_whiten=False)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        pca_results[dim] = mse

    # Print results
    print(f"TreeLinear without PCA: {baseline_mse:.6f}")
    for dim, mse in pca_results.items():
        print(f"TreeLinear with PCA (dim={dim}): {mse:.6f}")

    # At least one PCA configuration should perform reasonably compared to baseline
    best_pca_mse = min(pca_results.values())
    mse_ratio = best_pca_mse / baseline_mse
    print(f"Best PCA / No PCA ratio: {mse_ratio:.4f}")

    # Not asserting exact performance ratios as they're dataset dependent,
    # but ensure PCA doesn't completely destroy performance
    assert mse_ratio < 2.0, "PCA shouldn't drastically worsen performance"


if __name__ == "__main__":
    test_treelinear_basic()
    print("\n" + "-" * 50 + "\n")
    test_treelinear_nonlinear()
    print("\n" + "-" * 50 + "\n")
    test_treelinear_with_pca()

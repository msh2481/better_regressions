"""Visualization of tree-based models vs TreeLinear on a simple 1D example."""

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype as typed

from better_regressions import TreeLinear
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error


@typed
def generate_sine_data(n_samples: int = 1000, noise: float = 1.0, test_ratio: float = 0.8, random_state: int = 42):
    """Generate a simple 1D sine function dataset with noise."""
    np.random.seed(random_state)
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(X.ravel()) + np.random.normal(0, noise, n_samples)

    # Split into train/test randomly
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * (1 - test_ratio))
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test, X, y


@typed
def visualize_models(max_depth: int = 5, n_estimators: int = 100, num_leaves: int = 31, random_state: int = 42):
    """Visualize and compare different tree models on a 1D sine function."""
    # Generate data
    X_train, X_test, y_train, y_test, X_full, y_full = generate_sine_data(random_state=random_state)

    # Ground truth for plotting
    X_plot = np.linspace(-3.5, 3.5, 500).reshape(-1, 1)
    y_true = np.sin(X_plot.ravel())

    # Initialize models
    vanilla_et = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    vanilla_lgbm = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, verbose=-1)

    tree_linear_et = TreeLinear(tree_type="et", alpha="bayes", n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, hidden_dim=10)

    tree_linear_lgbm = TreeLinear(tree_type="lgbm", alpha="bayes", n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, hidden_dim=10)

    # Train models
    vanilla_et.fit(X_train, y_train)
    vanilla_lgbm.fit(X_train, y_train)
    tree_linear_et.fit(X_train, y_train)
    tree_linear_lgbm.fit(X_train, y_train)

    # Predict
    et_pred = vanilla_et.predict(X_plot)
    lgbm_pred = vanilla_lgbm.predict(X_plot)
    tree_linear_et_pred = tree_linear_et.predict(X_plot)
    tree_linear_lgbm_pred = tree_linear_lgbm.predict(X_plot)

    # Compute MSE on test set
    et_mse = mean_squared_error(y_test, vanilla_et.predict(X_test))
    lgbm_mse = mean_squared_error(y_test, vanilla_lgbm.predict(X_test))
    tree_linear_et_mse = mean_squared_error(y_test, tree_linear_et.predict(X_test))
    tree_linear_lgbm_mse = mean_squared_error(y_test, tree_linear_lgbm.predict(X_test))

    # Extract leaf counts for analysis
    if hasattr(tree_linear_et, "n_leaves_") and hasattr(tree_linear_lgbm, "n_leaves_"):
        print(f"ExtraTrees leaf count: {tree_linear_et.n_leaves_}")
        print(f"LightGBM leaf count: {tree_linear_lgbm.n_leaves_}")

    # Setup plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    # Plot data and predictions
    for i, (name, pred, mse) in enumerate([("ExtraTrees", et_pred, et_mse), ("LightGBM", lgbm_pred, lgbm_mse), ("TreeLinear + ET", tree_linear_et_pred, tree_linear_et_mse), ("TreeLinear + LGBM", tree_linear_lgbm_pred, tree_linear_lgbm_mse)]):
        axs[i].scatter(X_train, y_train, alpha=0.5, label="Train data")
        axs[i].scatter(X_test, y_test, alpha=0.5, label="Test data")
        axs[i].plot(X_plot, y_true, "r-", alpha=0.7, label="True function")
        axs[i].plot(X_plot, pred, "g-", linewidth=2, label=f"{name} prediction")
        axs[i].set_title(f"{name} (MSE: {mse:.5f})")
        axs[i].set_xlabel("X")
        axs[i].set_ylabel("y")
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tree_comparison.png", dpi=300)
    plt.show()

    # # Plot leaf distributions (relevant for understanding overfitting)
    # analyze_leaf_distributions(X_train, tree_linear_et, tree_linear_lgbm)


@typed
def analyze_leaf_distributions(X: np.ndarray, et_model: TreeLinear, lgbm_model: TreeLinear):
    """Analyze the leaf distributions of different tree models."""
    # Get leaf indices for both models
    et_leaves_raw = et_model.tree_model_.apply(X)
    lgbm_leaves_raw = lgbm_model.tree_model_.predict(X, pred_leaf=True)

    # Plot leaf distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Flatten for histograms
    et_leaves_flat = et_leaves_raw.flatten()
    lgbm_leaves_flat = lgbm_leaves_raw.flatten()

    # Histogram of leaf indices
    ax1.hist(et_leaves_flat, bins=50, alpha=0.7, label="ExtraTrees")
    ax1.set_title("ExtraTrees Leaf Distribution")
    ax1.set_xlabel("Leaf Index")
    ax1.set_ylabel("Count")

    ax2.hist(lgbm_leaves_flat, bins=50, alpha=0.7, label="LightGBM")
    ax2.set_title("LightGBM Leaf Distribution")
    ax2.set_xlabel("Leaf Index")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("leaf_distribution.png", dpi=300)
    plt.show()

    # Print statistics
    print(f"ExtraTrees unique leaves per tree (avg): {np.mean([len(np.unique(et_leaves_raw[:, i])) for i in range(et_leaves_raw.shape[1])]):.2f}")
    print(f"LightGBM unique leaves per tree (avg): {np.mean([len(np.unique(lgbm_leaves_raw[:, i])) for i in range(lgbm_leaves_raw.shape[1])]):.2f}")
    print(f"ExtraTrees total unique leaves: {len(np.unique(et_leaves_flat))}")
    print(f"LightGBM total unique leaves: {len(np.unique(lgbm_leaves_flat))}")


if __name__ == "__main__":
    # Try different depths and leaf counts
    for depth, leaves in [(3, 8), (5, 32)]:
        print(f"\n--- Testing with max_depth={depth}, num_leaves={leaves} ---\n")
        visualize_models(max_depth=depth, num_leaves=leaves)

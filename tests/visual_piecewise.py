import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype as typed

from better_regressions import Angle, AutoScaler, Linear, Scaler, Silencer, Smooth

from better_regressions.linear import Soft
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def test_piecewise_linear():
    """Visualize Angle regression on piecewise linear data."""
    np.random.seed(42)

    # Generate piecewise linear data with sharp transitions
    N = 300
    X = np.sort(np.random.uniform(-5, 5, N)).reshape(-1, 1)
    x_val = X.ravel()

    # Create piecewise function with multiple segments
    y = np.zeros(N)
    y[x_val < -3] = -3 - 0.7 * (x_val[x_val < -3] + 3)
    y[(x_val >= -3) & (x_val < -1)] = -3 + 1.5 * (x_val[(x_val >= -3) & (x_val < -1)] + 3)
    y[(x_val >= -1) & (x_val < 1)] = 0
    y[(x_val >= 1) & (x_val < 3)] = 2 * (x_val[(x_val >= 1) & (x_val < 3)] - 1)
    y[x_val >= 3] = 4 + 0.5 * (x_val[x_val >= 3] - 3)

    # Add noise
    y += np.random.normal(0, 0.2, N)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear": Scaler(Linear()),
        # "Angle (5 breakpoints)": Angle(n_breakpoints=5, random_state=42),
        "Angle (10 breakpoints)": Angle(n_breakpoints=10, random_state=42),
        # "Angle (20 breakpoints)": Angle(n_breakpoints=20, random_state=42),
        # "Smooth": Smooth(max_epochs=30, lr=0.1),
        "SoftLinear-Stabilize-0.1-0.3-0.7-0.9": Scaler(Soft(splits=[0.1, 0.3, 0.7, 0.9], estimator=Linear(alpha="bayes")), x_method="stabilize", y_method="stabilize"),
    }

    plt.figure(figsize=(12, 6))

    for model_name, model in models.items():
        with Silencer():
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{model_name}: {mse:.4f}")

        # Generate smooth curve for visualization
        X_demo = np.linspace(-5.5, 5.5, 500).reshape(-1, 1)
        y_demo = model.predict(X_demo)
        plt.plot(X_demo, y_demo, label=f"{model_name} (MSE: {mse:.4f})")

    # Plot training data
    plt.scatter(X_train, y_train, s=10, color="gray", alpha=0.5, label="Training data")

    # Plot true piecewise function (without noise)
    x_true = np.linspace(-5.5, 5.5, 1000)
    y_true = np.zeros(1000)
    y_true[x_true < -3] = -3 - 0.7 * (x_true[x_true < -3] + 3)
    y_true[(x_true >= -3) & (x_true < -1)] = -3 + 1.5 * (x_true[(x_true >= -3) & (x_true < -1)] + 3)
    y_true[(x_true >= -1) & (x_true < 1)] = 0
    y_true[(x_true >= 1) & (x_true < 3)] = 2 * (x_true[(x_true >= 1) & (x_true < 3)] - 1)
    y_true[x_true >= 3] = 4 + 0.5 * (x_true[x_true >= 3] - 3)

    plt.plot(x_true, y_true, "k--", label="True function")

    plt.title("Regression Models on Piecewise Linear Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def test_step_function():
    """Visualize Angle regression on step function data."""
    np.random.seed(42)

    # Generate step function data
    N = 300
    X = np.sort(np.random.uniform(-4, 4, N)).reshape(-1, 1)
    x_val = X.ravel()

    # Create step function
    y = np.zeros(N)
    y[x_val < -2] = -1.5
    y[(x_val >= -2) & (x_val < 0)] = 0
    y[(x_val >= 0) & (x_val < 2)] = 1.5
    y[x_val >= 2] = 0

    # Add noise
    y += np.random.normal(0, 0.15, N)

    models = {
        "Linear": Scaler(Linear()),
        "Angle (5 breakpoints)": Angle(n_breakpoints=5, random_state=42),
        "Angle (15 breakpoints)": Angle(n_breakpoints=15, random_state=42),
        "Smooth": Smooth(max_epochs=50, lr=0.1),
    }

    plt.figure(figsize=(12, 6))

    for model_name, model in models.items():
        with Silencer():
            model.fit(X, y)

        # Generate smooth curve for visualization
        X_demo = np.linspace(-4.5, 4.5, 500).reshape(-1, 1)
        y_demo = model.predict(X_demo)
        plt.plot(X_demo, y_demo, label=model_name)

    # Plot training data
    plt.scatter(X, y, s=10, color="gray", alpha=0.5, label="Training data")

    # Plot true step function (without noise)
    x_true = np.linspace(-4.5, 4.5, 1000)
    y_true = np.zeros(1000)
    y_true[x_true < -2] = -1.5
    y_true[(x_true >= -2) & (x_true < 0)] = 0
    y_true[(x_true >= 0) & (x_true < 2)] = 1.5
    y_true[x_true >= 2] = 0

    plt.plot(x_true, y_true, "k--", label="True function")

    plt.title("Regression Models on Step Function Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


@typed
def test_exp_sqrt_transformation():
    """Visualize the specific case of x-transform=exp, y-transform=sqrt with various models including AutoScaler."""
    np.random.seed(42)

    # Generate regression data (simpler version than benchmark to allow visualization)
    n_samples = 500
    n_features = 1  # Use 1D for visualization

    # Generate base data
    X_base = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    y_base = X_base @ w + 1.0 * np.random.randn(n_samples)

    # Apply transformations
    X = np.exp(X_base)  # exp transform
    y = np.sign(y_base) * np.sqrt(np.abs(y_base / np.std(y_base)))  # sqrt transform

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear-Standard": Scaler(Linear(alpha=1e-6), x_method="standard", y_method="standard"),
        "Linear-Power": Scaler(Linear(alpha=1e-6), x_method="power", y_method="power"),
        "Linear-Quantile": Scaler(Linear(alpha=1e-6), x_method="quantile-normal", y_method="quantile-normal"),
        "Angle-Standard": Scaler(Smooth(method="angle", n_breakpoints=2, max_epochs=100, lr=0.5), x_method="standard", y_method="standard"),
        "Angle-Power": Scaler(Smooth(method="angle", n_breakpoints=2, max_epochs=100, lr=0.5), x_method="power", y_method="power"),
        "Angle-Quantile": Scaler(Smooth(method="angle", n_breakpoints=2, max_epochs=100, lr=0.5), x_method="quantile-normal", y_method="quantile-normal"),
        "AutoScaler-Linear": AutoScaler(Linear(alpha=1e-6)),
        "AutoScaler-Angle": AutoScaler(Smooth(method="angle", n_breakpoints=2, max_epochs=100, lr=0.5)),
    }

    plt.figure(figsize=(12, 8))

    # Sort X for plotting
    X_sorted_indices = np.argsort(X_train.ravel())
    X_train_sorted = X_train[X_sorted_indices]
    y_train_sorted = y_train[X_sorted_indices]

    # Plot training data
    plt.scatter(X_train, y_train, s=10, color="gray", alpha=0.5, label="Training data")

    # Generate points for prediction curve
    X_demo = np.linspace(np.min(X) * 0.9, np.max(X) * 1.1, 1000).reshape(-1, 1)

    results = {}
    auto_selections = {}

    for model_name, model in models.items():
        with Silencer():
            model.fit(X_train, y_train)

        # Store AutoScaler selection info
        if "AutoScaler" in model_name and hasattr(model, "best_x_method_"):
            auto_selections[model_name] = f"{model.best_x_method_}/{model.best_y_method_} (score: {model.best_score_:.6f})"

        # Predict on test data for MSE calculation
        y_pred_test = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)
        results[model_name] = mse

        # Predict on demo points for visualization
        y_demo = model.predict(X_demo)

        # Plot predictions
        plt.plot(X_demo, y_demo, label=f"{model_name} (MSE: {mse:.6f})")

    # Add AutoScaler selections to title
    auto_title = ", ".join([f"{k}: chose {v}" for k, v in auto_selections.items()])
    plt.title(f"Regression Models on exp(X), sqrt(y) Transformed Data\n{auto_title}")
    plt.xlabel("X (after exp transform)")
    plt.ylabel("y (after sqrt transform)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Print MSE results
    print("\nMSE for exp(X), sqrt(y) transformation:")
    for model, mse in sorted(results.items(), key=lambda x: x[1]):
        print(f"{model}: {mse:.6f}")

    # Add a subplot to visualize the raw data relationship
    plt.figure(figsize=(12, 8))
    plt.scatter(X_base, y_base, s=10, color="gray", alpha=0.5, label="Original data (before transforms)")
    plt.title("Original Data (Before Transforms)")
    plt.xlabel("X (before exp transform)")
    plt.ylabel("y (before sqrt transform)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    test_piecewise_linear()
    # test_step_function()
    # test_exp_sqrt_transformation()

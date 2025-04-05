import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype as typed

from better_regressions import Angle, Linear, Scaler, Silencer, Smooth
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
        "Angle (5 breakpoints)": Angle(n_breakpoints=5, random_state=42),
        "Angle (10 breakpoints)": Angle(n_breakpoints=10, random_state=42),
        "Angle (20 breakpoints)": Angle(n_breakpoints=20, random_state=42),
        "Smooth": Smooth(max_epochs=30, lr=0.1),
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


if __name__ == "__main__":
    test_piecewise_linear()
    test_step_function()

import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype as typed

from better_regressions import Angle, Linear, Scaler, Silencer, Smooth
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def test_sine_wave():
    """Test visualization of smooth regressor on sine wave data."""
    np.random.seed(42)

    # Generate sine wave data with noise
    N = 200
    X = np.sort(np.random.uniform(-3, 3, N)).reshape(-1, 1)
    y = np.sin(X.ravel()) + np.random.normal(0, 0.2, N)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        # "Linear": Scaler(Linear()),
        "Smooth (supersmoother, 1 epochs)": Smooth(method="supersmoother", max_epochs=1, lr=1.0),
        "Smooth (supersmoother, 300 epochs)": Smooth(method="supersmoother", max_epochs=40, lr=0.5),
        "Smooth (angle, 300 epochs)": Smooth(method="angle", max_epochs=300, lr=0.2, n_breakpoints=1),
    }

    plt.figure(figsize=(10, 6))

    for model_name, model in models.items():
        with Silencer():
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{model_name}: {mse:.4f}")

        # Generate smooth curve for visualization
        X_demo = np.linspace(-3, 3, 300).reshape(-1, 1)
        y_demo = model.predict(X_demo)
        plt.plot(X_demo, y_demo, label=f"{model_name} (MSE: {mse:.4f})")

    # Plot training data and true function
    plt.scatter(X_train, y_train, s=10, color="gray", alpha=0.5, label="Training data")
    plt.plot(X_demo, np.sin(X_demo.ravel()), "k--", label="True function")

    plt.title("Regression Models on Sine Wave Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def test_multivariate():
    """Test visualization of feature contribution in multivariate smooth regression."""
    np.random.seed(42)

    # Generate multivariate data
    N = 3000
    X = np.random.uniform(-2, 2, (N, 3))

    # Function: y = sin(x0) + x1^2 - exp(x2/2)
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 - np.exp(X[:, 2] / 2)
    y += np.random.normal(0, 0.3, N)

    # Train smooth model
    model = Smooth(method="angle", max_epochs=50, lr=0.5, n_points=50)
    model.fit(X, y)

    # Create visualization of individual feature contributions
    plt.figure(figsize=(15, 5))

    feature_names = ["sin(x)", "x²", "exp(-x/2)"]
    true_funcs = [lambda x: np.sin(x), lambda x: x**2, lambda x: -np.exp(x / 2)]

    for i in range(3):
        plt.subplot(1, 3, i + 1)

        # Generate test points for this feature
        x_range = np.linspace(-2, 2, 100)

        # For each test point, randomize other features to isolate this feature's contribution
        feature_contrib = np.zeros(100)
        for j in range(100):
            # Run multiple predictions with random values for other features
            n_samples = 10**4
            X_batch = np.random.uniform(-2, 2, (n_samples, 3))
            X_batch[:, i] = x_range[j]
            # Average predictions to get stable contribution estimate
            feature_contrib[j] = np.mean(model.predict(X_batch))

        true_contrib = true_funcs[i](x_range)
        feature_contrib += true_contrib.mean() - feature_contrib.mean()
        plt.plot(x_range, feature_contrib, label="Smooth model")
        plt.plot(x_range, true_contrib, "k--", label="True function")

        plt.title(f"Feature: {feature_names[i]}")
        plt.xlabel("x")
        plt.ylabel("Contribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def test_method_comparison():
    """Compare different smoothing methods on piecewise data."""
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

    models = {
        "Linear": Scaler(Linear()),
        "Angle (direct)": Angle(n_breakpoints=10, random_state=42),
        "Smooth (supersmoother)": Smooth(method="supersmoother", max_epochs=30, lr=0.1),
        "Smooth (angle)": Smooth(method="angle", max_epochs=30, lr=0.1, n_breakpoints=10),
    }

    plt.figure(figsize=(12, 6))

    for model_name, model in models.items():
        with Silencer():
            model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        print(f"{model_name}: {mse:.4f}")

        # Generate smooth curve for visualization
        X_demo = np.linspace(-5.5, 5.5, 500).reshape(-1, 1)
        y_demo = model.predict(X_demo)
        plt.plot(X_demo, y_demo, label=f"{model_name} (MSE: {mse:.4f})")

    # Plot training data
    plt.scatter(X, y, s=10, color="gray", alpha=0.5, label="Training data")

    # Plot true piecewise function (without noise)
    x_true = np.linspace(-5.5, 5.5, 1000)
    y_true = np.zeros(1000)
    y_true[x_true < -3] = -3 - 0.7 * (x_true[x_true < -3] + 3)
    y_true[(x_true >= -3) & (x_true < -1)] = -3 + 1.5 * (x_true[(x_true >= -3) & (x_true < -1)] + 3)
    y_true[(x_true >= -1) & (x_true < 1)] = 0
    y_true[(x_true >= 1) & (x_true < 3)] = 2 * (x_true[(x_true >= 1) & (x_true < 3)] - 1)
    y_true[x_true >= 3] = 4 + 0.5 * (x_true[x_true >= 3] - 3)

    plt.plot(x_true, y_true, "k--", label="True function")

    plt.title("Comparison of Smoothing Methods on Piecewise Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def test_step_transition():
    """Compare how different methods handle step transitions."""
    np.random.seed(42)

    # Generate data with sine wave and sharp transition
    N = 200
    X = np.sort(np.random.uniform(-3, 3, N)).reshape(-1, 1)
    x_val = X.ravel()

    # Sine wave with step transitions
    y = np.sin(x_val)
    y[x_val > 0] += 1.0  # Add step at x=0
    y[x_val > 2] -= 0.5  # Add another step at x=2

    # Add noise
    y += np.random.normal(0, 0.1, N)

    models = {
        "Linear": Scaler(Linear()),
        "Smooth (supersmoother, 10 epochs)": Smooth(method="supersmoother", max_epochs=10, lr=0.1),
        "Smooth (supersmoother, 50 epochs)": Smooth(method="supersmoother", max_epochs=50, lr=0.1),
        "Smooth (angle, 10 epochs)": Smooth(method="angle", max_epochs=10, lr=0.1, n_breakpoints=5),
        "Smooth (angle, 50 epochs)": Smooth(method="angle", max_epochs=50, lr=0.1, n_breakpoints=5),
    }

    plt.figure(figsize=(14, 6))

    for model_name, model in models.items():
        with Silencer():
            model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        print(f"{model_name}: {mse:.4f}")

        # Generate smooth curve for visualization
        X_demo = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
        y_demo = model.predict(X_demo)
        plt.plot(X_demo, y_demo, label=f"{model_name} (MSE: {mse:.4f})")

    # Plot training data
    plt.scatter(X, y, s=10, color="gray", alpha=0.5, label="Training data")

    # Plot true function (without noise)
    x_true = np.linspace(-3.5, 3.5, 500)
    y_true = np.sin(x_true)
    y_true[x_true > 0] += 1.0
    y_true[x_true > 2] -= 0.5

    plt.plot(x_true, y_true, "k--", label="True function")

    plt.title("Comparison of Smoothing Methods on Data with Step Transitions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def test_cubic():
    N = 100
    X = np.linspace(-2, 2, N).reshape(-1, 1)
    y = X.ravel() ** 3 + np.random.randn(N) * 2.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear": Scaler(Linear()),
        "Smooth-Angle": Smooth(method="angle", max_epochs=100, lr=0.5, n_points=50),
        "Smooth-SuperSmoother": Smooth(method="supersmoother", max_epochs=1, lr=1.0, n_points=50),
        # "QNormal": Scaler(Linear(), x_method="quantile-normal", y_method="quantile-normal"),
        # "Power": Scaler(Linear(), x_method="power", y_method="power"),
        # "Isotonic": IsotonicRegression(out_of_bounds="clip"),
    }

    for model_name, model in models.items():
        with Silencer():
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{model_name}: {mse}")
        X_demo = np.linspace(-2, 2, 100).reshape(-1, 1)
        y_demo = model.predict(X_demo)
        plt.plot(X_demo, y_demo, label=model_name)
    plt.plot(X_train, y_train, "kx", label="Training data")
    plt.plot(X_demo, X_demo**3, "k--", label="True function")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_cubic()

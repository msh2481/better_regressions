import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

from better_regressions import Scaler
from better_regressions.linear import AdaptiveLinear, Linear


def test_same_noise_level():
    np.random.seed(42)
    n_samples = 500
    n_test = 1000
    d = 4
    n_runs = 50

    noise_levels = [0.01, 0.1, 0.2, 0.5, 1.0, 4.0]
    n_copies = 3

    models = {
        "Linear": Linear(alpha=1e-18, better_bias=False),
        "Linear'": Linear(alpha=1e-18),
        "Linear(ARD)": Linear(alpha="ard", better_bias=False),
        "Linear(ARD')": Linear(alpha="ard"),
        "PLS": PLSRegression(n_components=d),
        # "AdaptiveRidge(pca, 1e-18)": AdaptiveLinear(method="pca", alpha=1e-9),
        # "AdaptiveRidge(pls, 1e-18)": AdaptiveLinear(method="pls", alpha=1e-9),
        "AdaptiveRidge(pca, 1)": AdaptiveLinear(method="pca"),
        "AdaptiveRidge(pls, 1)": AdaptiveLinear(method="pls"),
        "AdaptiveRidge(pca, bayes)": AdaptiveLinear(method="pca", alpha="bayes"),
        "AdaptiveRidge(pls, bayes)": AdaptiveLinear(method="pls", alpha="bayes"),
    }

    print(f"True features: {d}")
    print(f"Noise levels per copy: {noise_levels}")
    print(f"Copies per noise level: {n_copies}")
    print(f"Number of runs per configuration: {n_runs}\n")

    for noise_level in noise_levels:
        print(f"Features: {d + n_copies * d} (added {n_copies} copies with noise={noise_level})")

        mse_results = {name: [] for name in models.keys()}

        for run in range(n_runs):
            true_coef = np.random.randn(d)
            X_true = np.random.randn(n_samples, d)
            y = X_true @ true_coef + 1.0 * np.random.randn(n_samples)

            X_test = np.random.randn(n_test, d)
            y_test = X_test @ true_coef

            new_features = []
            new_test_features = []

            for _ in range(n_copies):
                noisy_copy = X_true + noise_level * np.random.randn(n_samples, d)
                new_features.append(noisy_copy)

                noisy_test = X_test + noise_level * np.random.randn(X_test.shape[0], d)
                new_test_features.append(noisy_test)

            X_train = np.hstack(new_features)
            X_test = np.hstack(new_test_features)

            for name, model in models.items():
                model_copy = clone(model)
                model_copy.fit(X_train, y)
                y_pred = model_copy.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_results[name].append(mse)

        for name in models.keys():
            avg_mse = np.mean(mse_results[name])
            std_mse = np.std(mse_results[name])
            print(f"  {name}: MSE = {avg_mse:.4f} ± {std_mse:.4f}")

        print()


def test_diverse_noise_levels():
    np.random.seed(42)
    N = 1000
    levels = np.arange(1, 11)
    ks = [1, 1, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10]

    def test_model(model_name, model_fn, k):
        results = []
        noise_scales = []
        for level in levels[:k]:
            noise_scales.append(level * np.ones(N))
        noise_scale = np.stack(noise_scales, axis=1)
        for _ in range(1):
            g = np.random.uniform(-1e6, 1e6, N)
            X = g[:, None] + noise_scale * np.random.randn(N, k)
            y = g
            X_train = X[: len(X) // 20]
            y_train = y[: len(y) // 20]
            # y_train += np.random.randn(len(y_train))
            X_test = X[len(X) // 20 :]
            y_test = y[len(y) // 20 :]
            model = model_fn()
            model.fit(X_train, y_train)

            inputs = np.random.randn(10 * k, k) * 1e6
            outputs = model.predict(inputs)
            linear = Linear(alpha=1e-9, better_bias=False)
            linear.fit(inputs, outputs)

            y_pred = model.predict(X_test)
            result = mean_squared_error(y_test, y_pred) ** 0.5
            results.append(result)
        return np.mean(results), np.std(results) / np.sqrt(len(results))

    models = {
        "Linear": lambda: Scaler(Linear(alpha=1e-18, better_bias=False), x_method="standard", y_method="standard"),
        "Linear'": lambda: Scaler(Linear(alpha=1e-18), x_method="standard", y_method="standard"),
        # "Linear(ARD)": lambda: Scaler(Linear(alpha="ard", better_bias=False), x_method="standard", y_method="standard"),
        # "Linear(ARD')": lambda: Scaler(Linear(alpha="ard"), x_method="standard", y_method="standard"),
        "AdaptiveRidge(pca, 1)": lambda: AdaptiveLinear(method="pca"),
        # "AdaptiveRidge(pls, 1)": lambda: AdaptiveLinear(method="pls"),
        "AdaptiveRidge(pca, bayes)": lambda: AdaptiveLinear(method="pca", alpha="bayes"),
        # "AdaptiveRidge(pls, bayes)": lambda: AdaptiveLinear(method="pls", alpha="bayes"),
    }

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="RMSE Results by Model and Number of Features")

    table.add_column("Model", style="cyan", no_wrap=True)
    for k in ks:
        table.add_column(f"k={k}", justify="right")

    for name, model in models.items():
        mse_results = []
        for k in ks:
            mse, std = test_model(name, model, k)
            mse_results.append(f"{mse:.4f} ± {std:.4f}")
        table.add_row(name, *mse_results)

    console.print(table)


def test_sparse():
    N_train = 1000
    N_test = 10000
    N = N_train + N_test

    def test_model(name, model_fn, k):
        results = []
        for _ in range(100):
            X = np.random.randn(N, k)
            w = np.random.randn(k)
            w[3:] = 0
            eps = np.random.randn(N) * 0.1
            y = X @ w + eps
            X_train = X[:N_train]
            y_train = y[:N_train]
            X_test = X[N_train:]
            y_test = y[N_train:]
            model = model_fn()
            model.fit(X_train, y_train)

            # inputs = np.random.randn(10 * k, k) * 1e6
            # outputs = model.predict(inputs)
            # linear = Linear(alpha=1e-9, better_bias=False)
            # linear.fit(inputs, outputs)

            y_pred = model.predict(X_test)
            result = mean_squared_error(y_test, y_pred) ** 0.5
            results.append(result)
        return np.mean(results), np.std(results) / np.sqrt(len(results))

    # ks = [5, 10, 20, 50]
    ks = [8]
    models = {
        "Linear": lambda: Scaler(Linear(alpha=1e-18, better_bias=False), x_method="standard", y_method="standard"),
        # "Linear'": lambda: Scaler(Linear(alpha=1e-18), x_method="standard", y_method="standard"),
        # "AdaptiveRidge(pca, 1)": lambda: AdaptiveLinear(method="pca"),
        "AdaptiveRidge(pls, 1)": lambda: AdaptiveLinear(method="pls"),
        # "AdaptiveRidge(pca, bayes)": lambda: AdaptiveLinear(method="pca", alpha="bayes"),
        # "AdaptiveRidge(pls, bayes)": lambda: AdaptiveLinear(method="pls", alpha="bayes"),
    }

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="RMSE Results by Model and Number of Features (Sparse)")

    table.add_column("Model", style="cyan", no_wrap=True)
    for k in ks:
        table.add_column(f"k={k}", justify="right")

    for name, model in models.items():
        mse_results = []
        for k in ks:
            mse, std = test_model(name, model, k)
            mse_results.append(f"{mse:.4f} ± {std:.4f}")
        table.add_row(name, *mse_results)

    console.print(table)


if __name__ == "__main__":
    # test_same_noise_level()
    # test_diverse_noise_levels()
    test_sparse()

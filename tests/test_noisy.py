import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

from better_regressions.linear import AdaptiveRidge, Linear


def test_noisy_features():
    np.random.seed(42)
    n_samples = 500
    n_test = 1000
    d = 4
    n_runs = 50

    noise_levels = [0.0, 0.01, 0.03, 0.1, 0.2]
    n_copies = 3

    models = {
        "Linear": Linear(alpha=1e-18, better_bias=False),
        "Linear'": Linear(alpha=1e-18),
        "Linear(ARD)": Linear(alpha="ard", better_bias=False),
        "Linear(ARD')": Linear(alpha="ard"),
        "PLS": PLSRegression(n_components=d),
        "AdaptiveRidge0000": AdaptiveRidge(use_pls=False, use_scaling=False, use_corr=False, alpha=1e-18),
        "AdaptiveRidge": AdaptiveRidge(better_bias=False),
        "AdaptiveRidge'": AdaptiveRidge(better_bias=True),
        "AdaptiveRidge'(ARD)": AdaptiveRidge(better_bias=True, alpha="ard"),
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


if __name__ == "__main__":
    test_noisy_features()

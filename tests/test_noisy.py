import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

from better_regressions.linear import AdaptiveRidge, Linear


def test_noisy_features():
    np.random.seed(42)
    n_samples = 500
    n_test = 100
    d = 4

    true_coef = np.random.randn(d)
    X_true = np.random.randn(n_samples, d)
    y = X_true @ true_coef

    X_test = np.random.randn(n_test, d)
    y_test = X_test @ true_coef

    noise_levels = [0.5, 1.0, 2.0, 4.0]
    n_copies = 3

    models = {"Linear": Linear(), "Linear(ARD)": Linear(alpha="ard"), "AdaptiveRidge": AdaptiveRidge(), "PLS": PLSRegression()}

    print(f"True features: {d}")
    print(f"Noise levels per copy: {noise_levels}")
    print(f"Copies per noise level: {n_copies}\n")

    for noise_level in noise_levels:
        new_features = []
        new_test_features = []

        for _ in range(n_copies):
            noisy_copy = X_true + noise_level * np.random.randn(n_samples, d)
            new_features.append(noisy_copy)

            noisy_test = X_test + noise_level * np.random.randn(X_test.shape[0], d)
            new_test_features.append(noisy_test)

        X_train = X_true.copy()
        X_test_full = X_test.copy()
        X_train = np.hstack([X_train] + new_features)
        X_test_full = np.hstack([X_test_full] + new_test_features)

        print(f"Features: {X_train.shape[1]} (added {n_copies} copies with noise={noise_level})")

        for name, model in models.items():
            model_copy = clone(model)
            model_copy.fit(X_train, y)
            y_pred = model_copy.predict(X_test_full)
            mse = mean_squared_error(y_test, y_pred)
            print(f"  {name}: MSE = {mse:.4f}")

        print()


if __name__ == "__main__":
    test_noisy_features()

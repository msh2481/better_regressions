import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from better_regressions.kan import OrdinaryMLP, PointwiseKAN, PowerMLP, ReLUKAN, fit_regression, test_regression


def generate_sine_1d(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, (n_samples, 1))
    y = np.sin(2 * math.pi * x[:, 0]) + np.random.normal(0, noise, n_samples)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32)


def generate_exponential_1d(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    x = np.random.uniform(-2, 2, (n_samples, 1))
    y = np.exp(-(x[:, 0] ** 2)) + np.random.normal(0, noise, n_samples)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32)


def generate_abs_1d(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, (n_samples, 1))
    y = np.abs(x[:, 0]) + np.random.normal(0, noise, n_samples)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32)


def generate_step_1d(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, (n_samples, 1))
    y = (x[:, 0] >= 0).astype(float) + np.random.normal(0, noise, n_samples)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32)


def generate_product_2d(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, (n_samples, 2))
    y = x[:, 0] * x[:, 1] + np.random.normal(0, noise, n_samples)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32)


def generate_sum_sines_2d(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, (n_samples, 2))
    y = np.sin(2 * math.pi * x[:, 0]) + np.sin(2 * math.pi * x[:, 1]) + np.random.normal(0, noise, n_samples)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32)


def generate_gaussian_2d(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    x = np.random.uniform(-2, 2, (n_samples, 2))
    y = np.exp(-(x[:, 0] ** 2 + x[:, 1] ** 2)) + np.random.normal(0, noise, n_samples)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32)


def generate_product_sine_2d(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, (n_samples, 2))
    y = x[:, 0] * x[:, 1] + np.sin(2 * math.pi * x[:, 0]) + np.random.normal(0, noise, n_samples)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32)


def generate_piecewise_2d(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, (n_samples, 2))
    y = np.where(x[:, 0] < 0, 1.0, np.where(x[:, 1] < 0, 2.0, 3.0)) + np.random.normal(0, noise, n_samples)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32)


def generate_product_3d(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, (n_samples, 3))
    y = x[:, 0] * x[:, 1] * x[:, 2] + np.random.normal(0, noise, n_samples)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32)


def test_kan_models():
    functions = [
        ("Sine 1D", generate_sine_1d, 1),
        ("Exponential 1D", generate_exponential_1d, 1),
        ("Absolute 1D", generate_abs_1d, 1),
        ("Step 1D", generate_step_1d, 1),
        ("Product 2D", generate_product_2d, 2),
        ("Sum Sines 2D", generate_sum_sines_2d, 2),
        ("Gaussian 2D", generate_gaussian_2d, 2),
        ("Product+Sine 2D", generate_product_sine_2d, 2),
        ("Piecewise 2D", generate_piecewise_2d, 2),
        ("Product 3D", generate_product_3d, 3),
    ]

    noise_level = 0.05
    n_train = 1000
    n_val = 200
    n_test = 200
    max_epochs = 200
    batch_size = 32

    print(f"\nTesting KAN models with noise={noise_level}, n_train={n_train}")
    print("=" * 80)

    for func_name, func_gen, n_features in functions:
        print(f"\n{func_name} (dim={n_features}):")
        
        X_train, y_train = func_gen(n_train, noise=noise_level, seed=42)
        X_val, y_val = func_gen(n_val, noise=noise_level, seed=43)
        X_test, y_test = func_gen(n_test, noise=0.0, seed=44)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

        criterion = nn.MSELoss()
        results = {}

        model = OrdinaryMLP(dim_list=[n_features, 32, 32, 1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        fit_regression(model, criterion, optimizer, train_loader, val_loader, max_epochs=max_epochs)
        rmse = test_regression(model, test_loader, use_rmse=True)
        results["OrdinaryMLP"] = rmse

        power_mlp_width = [n_features, 32, 32, 1]
        if len(power_mlp_width) == 2:
            power_mlp_width = [n_features, 16, 1]
        model = PowerMLP(power_mlp_width, repu_order=2, res=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        fit_regression(model, criterion, optimizer, train_loader, val_loader, max_epochs=max_epochs)
        rmse = test_regression(model, test_loader, use_rmse=True)
        results["PowerMLP"] = rmse

        model = ReLUKAN(width=[n_features, 16, 16, 1], grid=5, k=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        fit_regression(model, criterion, optimizer, train_loader, val_loader, max_epochs=max_epochs)
        rmse = test_regression(model, test_loader, use_rmse=True)
        results["ReLUKAN"] = rmse

        model = PointwiseKAN(dim_list=[n_features, 16, 16, 1], num_repu_terms=4, repu_order=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        fit_regression(model, criterion, optimizer, train_loader, val_loader, max_epochs=max_epochs)
        rmse = test_regression(model, test_loader, use_rmse=True)
        results["PointwiseKAN"] = rmse

        print(f"  OrdinaryMLP:  RMSE = {results['OrdinaryMLP']:.6f}")
        print(f"  PowerMLP:      RMSE = {results['PowerMLP']:.6f}")
        print(f"  ReLUKAN:       RMSE = {results['ReLUKAN']:.6f}")
        print(f"  PointwiseKAN:  RMSE = {results['PointwiseKAN']:.6f}")

if __name__ == "__main__":
    test_kan_models()
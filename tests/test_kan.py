import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from better_regressions.kan import MLP, KAN, fit_regression, test_regression


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


def visualize_predictions(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    func_name: str,
    model_name: str,
    n_features: int,
    output_dir: str = "kan_results",
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    X_test = X_test.to(device)
    
    with torch.no_grad():
        y_pred = model(X_test)
    
    X_test = X_test.cpu().numpy()
    y_test = y_test.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if n_features == 1:
        idx = np.argsort(X_test[:, 0])
        ax.plot(X_test[idx, 0], y_test[idx, 0], "b-", label="Ground Truth", alpha=0.7, linewidth=2)
        ax.plot(X_test[idx, 0], y_pred[idx, 0], "r--", label="Prediction", alpha=0.7, linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5, s=10)
        min_val = min(y_test[:, 0].min(), y_pred[:, 0].min())
        max_val = max(y_test[:, 0].max(), y_pred[:, 0].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x", linewidth=2)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    
    title = f"{func_name} - {model_name}"
    ax.set_title(title)
    
    filename = f"{func_name.replace(' ', '_').replace('+', 'plus')}_{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


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

    noise_level = 0.1
    n_train = 100
    n_val = 200
    n_test = 1000
    max_epochs = 1000
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

        LR = 1e-2

        models_to_test = [
            ("MLP", MLP(dim_list=[n_features, 32, 1])),
            ("MLP (residual)", MLP(dim_list=[n_features, 32, 1], residual=True)),
            ("KAN", KAN(dim_list=[n_features, 32, 32, 32, 1], k=16)),
            ("KAN (residual)", KAN(dim_list=[n_features, 32, 32, 32, 1], k=16, residual=True)),
        ]
        
        for model_name, model in models_to_test:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            fit_regression(model, criterion, optimizer, train_loader, val_loader, max_epochs=max_epochs)
            train_rmse = test_regression(model, train_loader, use_rmse=True)
            test_rmse = test_regression(model, test_loader, use_rmse=True)
            results[model_name] = (train_rmse, test_rmse)
            visualize_predictions(model, X_test, y_test, func_name, model_name, n_features)

        print(f"  MLP:                     Train RMSE = {results['MLP'][0]:.6f} | Test RMSE = {results['MLP'][1]:.6f}")
        print(f"  MLP (residual):          Train RMSE = {results['MLP (residual)'][0]:.6f} | Test RMSE = {results['MLP (residual)'][1]:.6f}")
        print(f"  KAN:                     Train RMSE = {results['KAN'][0]:.6f} | Test RMSE = {results['KAN'][1]:.6f}")
        print(f"  KAN (residual):          Train RMSE = {results['KAN (residual)'][0]:.6f} | Test RMSE = {results['KAN (residual)'][1]:.6f}")

if __name__ == "__main__":
    test_kan_models()
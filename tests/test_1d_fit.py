import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from better_regressions import kan


def generate_4d_dataset(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    a = np.random.uniform(-2, 2, n_samples)
    b = np.random.uniform(-2, 2, n_samples)
    c = np.random.uniform(-2, 2, n_samples)
    d = np.random.uniform(-1, 1, n_samples)
    
    x = np.column_stack([a, b, c, d])
    
    y = np.column_stack([
        a**3 - a,
        np.tanh(b),
        np.exp(-c**2),
        1.0 / (np.abs(d) + 0.1)
    ])
    
    if noise > 0:
        y += np.random.normal(0, noise, y.shape)
    
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def visualize_predictions(model, x_test, y_test, save_path: str):
    model.eval()
    device = next(model.parameters()).device
    x_test = x_test.to(device)
    
    with torch.no_grad():
        y_pred = model(x_test).cpu().numpy()
    
    x_test_np = x_test.cpu().numpy()
    y_test_np = y_test.numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    labels = ["a³ - a", "tanh(b)", "exp(-c²)", "1/(|d|+0.1)"]
    input_idx = [0, 1, 2, 3]
    
    for idx, (ax, label, inp_idx) in enumerate(zip(axes, labels, input_idx)):
        sorted_indices = np.argsort(x_test_np[:, inp_idx])
        x_sorted = x_test_np[sorted_indices, inp_idx]
        y_true_sorted = y_test_np[sorted_indices, idx]
        y_pred_sorted = y_pred[sorted_indices, idx]
        
        ax.plot(x_sorted, y_true_sorted, "b-", label="Ground truth", alpha=0.7, linewidth=2)
        ax.plot(x_sorted, y_pred_sorted, "r-", label="Prediction", alpha=0.7, linewidth=0.5)
        ax.set_xlabel(f"Input {['a', 'b', 'c', 'd'][inp_idx]}")
        ax.set_ylabel(f"Output: {label}")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def test_pointwise():
    os.makedirs("kan_results", exist_ok=True)
    
    x_train, y_train = generate_4d_dataset(1000, noise=0.01, seed=42)
    x_test, y_test = generate_4d_dataset(500, noise=0.0, seed=123)
    
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # model = kan.PointwiseRELUKAN(input_size=4, k=8)
    model = kan.MLP(dim_list=[4, 10, 4], residual=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    iteration = 0
    max_iterations = 10**9
    
    for epoch in range(1000):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            iteration += 1
            
            if iteration % 1000 == 0:
                visualize_predictions(model, x_test, y_test, "kan_results/1d.png")
                time.sleep(0.1)
            
            if iteration >= max_iterations:
                break
        
        if iteration >= max_iterations:
            break
    
    visualize_predictions(model, x_test, y_test, "kan_results/1d.png")


if __name__ == "__main__":
    test_pointwise()


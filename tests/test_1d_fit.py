import os
from shutil import rmtree
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from better_regressions import kan
from better_regressions.kan import visualize_kan


def generate_4d_dataset(n_samples: int, noise: float = 0.0, seed: int = 42):
    np.random.seed(seed)
    a = np.random.uniform(-2, 2, n_samples)
    b = np.random.uniform(-2, 2, n_samples)
    c = np.random.uniform(-2, 2, n_samples)
    d = np.random.uniform(-1, 1, n_samples)

    x = np.column_stack([a, b, c, d])

    y = np.column_stack([a**3 - a, np.tanh(b), np.exp(-(c**2)), 1.0 / (np.abs(d) + 0.1)])
    y /= y.std(axis=0)

    if noise > 0:
        y += np.random.normal(0, noise, y.shape)

    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def visualize_predictions(model, x_train, y_train, x_test, y_test, save_path: str):
    model.eval()
    device = next(model.parameters()).device
    x_train = x_train.to(device)
    x_test = x_test.to(device)

    criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        train_pred = model(x_train)
        test_pred = model(x_test)

        train_losses = criterion(train_pred, y_train.to(device)).mean(dim=0).cpu().numpy()
        test_losses = criterion(test_pred, y_test.to(device)).mean(dim=0).cpu().numpy()

    x_test_np = x_test.cpu().numpy()
    y_test_np = y_test.numpy()
    test_pred_np = test_pred.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    labels = ["a³ - a", "tanh(b)", "exp(-c²)", "1/(|d|+0.1)"]
    input_idx = [0, 1, 2, 3]

    for idx, (ax, label, inp_idx) in enumerate(zip(axes, labels, input_idx)):
        sorted_indices = np.argsort(x_test_np[:, inp_idx])
        x_sorted = x_test_np[sorted_indices, inp_idx]
        y_true_sorted = y_test_np[sorted_indices, idx]
        y_pred_sorted = test_pred_np[sorted_indices, idx]

        ax.plot(x_sorted, y_true_sorted, "b-", label="Ground truth", alpha=0.7, linewidth=2)
        ax.plot(x_sorted, y_pred_sorted, "r-", label="Prediction", alpha=0.7, linewidth=0.5)
        ax.set_xlabel(f"Input {['a', 'b', 'c', 'd'][inp_idx]}")
        ax.set_ylabel(f"Output: {label}")
        loss_text = f"Train: {train_losses[idx]:.4f}, Test: {test_losses[idx]:.4f}"
        ax.set_title(f"{label}\n{loss_text}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def test_pointwise():
    rmtree("kan_results", ignore_errors=True)
    os.makedirs("kan_results", exist_ok=True)

    n = 1000
    x_train, y_train = generate_4d_dataset(n, noise=0.01, seed=42)
    x_test, y_test = generate_4d_dataset(n, noise=0.0, seed=123)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=n, shuffle=True)

    # model = kan.PointwiseRELUKAN(input_size=4, k=8)
    # model = kan.MLP(dim_list=[4, 10, 4], residual=False)
    model = kan.KAN(dim_list=[4, 32, 4], k=16, copies=8, residual=False)
    # model = nn.Sequential(
    #     kan.RunningRMSNorm(4),
    #     kan.PointwiseRELU(input_size=4, k=16)
    # )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    max_epochs = 10**9
    next_save_epoch = 16

    for epoch in range(max_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) + model.extra_loss()
            loss.backward()
            optimizer.step()

        if epoch == next_save_epoch:
            visualize_predictions(model, x_train, y_train, x_test, y_test, f"kan_results/1d_{epoch}.png")
            if isinstance(model, kan.KAN):
                visualize_kan(model, f"kan_results/kan_epoch_{epoch}")
            next_save_epoch *= 2

        if epoch >= max_epochs:
            break

    visualize_predictions(model, x_train, y_train, x_test, y_test, f"kan_results/1d_{epoch}.png")
    if isinstance(model, kan.KAN):
        visualize_kan(model, f"kan_results/kan_epoch_{epoch}")


if __name__ == "__main__":
    test_pointwise()

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from better_regressions.timeseries import AdaptiveEMA, MLPEMA, ema, fit_regression, test_regression


def test_timeseries():
    torch.manual_seed(42)

    n = 10
    x_value = torch.zeros(1, 1, n)
    x_value[0, 0, n // 2] = 1.0
    x = x_value.detach().clone().requires_grad_(True)
    halflife = torch.full((1,), 1.0, requires_grad=True)

    print("Input x:", x.detach().numpy())
    print("Input halflife:", halflife.detach().numpy())

    result = ema(x, halflife)
    print("\nEMA output:", result.detach().numpy())

    loss = result[0, 0, -1]
    loss.backward()

    print("\nGradients:")
    print("x.grad:", x.grad.numpy() if x.grad is not None else None)
    print("halflife.grad:", halflife.grad.numpy() if halflife.grad is not None else None)

    assert x.grad is not None, "Gradients should flow to x"
    assert halflife.grad is not None, "Gradients should flow to halflife"

    assert not torch.isnan(x.grad).any(), "x.grad should not contain NaNs"
    assert not torch.isnan(halflife.grad).any(), "halflife.grad should not contain NaNs"

    print("\n✓ All gradients flow correctly!")


def test_adaptive_ema():
    torch.manual_seed(42)

    batch_size, num_features, seq_len = 4, 3, 10
    x = torch.randn(batch_size, num_features, seq_len)

    model = AdaptiveEMA(num_features, halflife_bounds=(0.1, 10.0))
    result = model(x)

    print(f"\nAdaptiveEMA output shape: {result.shape}")
    print(f"Output sample: {result[0, 0, :5].detach().numpy()}")

    loss = result.sum()
    loss.backward()

    assert model.log_halflife.grad is not None, "Gradients should flow to log_halflife"
    print("✓ AdaptiveEMA gradients flow correctly!")


def test_fit_adaptive_ema():
    torch.manual_seed(0)

    batch_size, num_features, seq_len = 32, 3, 50

    x = torch.randn(batch_size, num_features, seq_len)

    y = torch.zeros_like(x)
    y[:, 0, :] = x[:, 0, :]

    halflife_index = torch.tensor([3.0])
    y[:, 1:2, :] = ema(x[:, 1:2, :], halflife_index)

    halflife_index = torch.tensor([5.0])
    y[:, 2:3, :] = ema(x[:, 2:3, :], halflife_index)

    train_size = int(0.8 * batch_size)
    x_train, x_val = x[:train_size], x[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = AdaptiveEMA(num_features, halflife_bounds=(0.01, 20.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print("\nBefore training:")
    halflifes = torch.exp(model.log_halflife.detach())
    print(f"Halflifes: {halflifes.numpy()}")

    best_loss, best_epoch = fit_regression(model, optimizer, train_loader, val_loader, max_epochs=1000, use_rmse=True)

    print(f"\nBest validation loss: {best_loss:.6f} at epoch {best_epoch}")

    print("\nAfter training:")
    halflifes = torch.exp(model.log_halflife.detach())
    print(f"Halflifes: {halflifes.numpy()}")
    print(f"\nExpected:")
    print(f"Feature 0 (last x): halflife ~ 0.01")
    print(f"Feature 1 (ema span=3, index space): halflife ~ 3.0")
    print(f"Feature 2 (ema span=5, time space): halflife ~ 5.0")


def test_fit_mlpema():
    torch.manual_seed(42)

    batch_size, num_features, seq_len = 1000, 1, 100
    x = torch.randn(batch_size, num_features, seq_len)

    halflife = torch.tensor([5.0])
    y = ema(x**2, halflife) - ema(x, 2 * halflife) ** 2
    y -= y.mean(dim=(0, 2), keepdim=True)
    y /= y.std(dim=(0, 2), keepdim=True) + 1e-8

    train_size = int(0.8 * batch_size)
    x_train, x_val = x[:train_size], x[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    model = MLPEMA(num_features=1, dim_lists=[[1, 32, 4]], halflife_bounds=(1.0, 100.0), out_features=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

    print("\nFitting MLPEMA to y = ema(x^2, halflife=1000)")
    print(f"Before training - Validation RMSE: {test_regression(model, val_loader, use_rmse=True):.6f}")

    best_loss, best_epoch = fit_regression(model, optimizer, train_loader, val_loader, max_epochs=200, use_rmse=True)

    print(f"\nBest validation loss: {best_loss:.6f} at epoch {best_epoch}")
    print(f"Final validation RMSE: {test_regression(model, val_loader, use_rmse=True):.6f}")


def test_fit_mlpema_hard():
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size, num_features, seq_len = 200, 3, 50

    x_list = []
    y_list = []

    for b in range(batch_size):
        x1_bias = np.random.randn() * 10
        x1_scale = np.exp(np.linspace(0.0, 2.0, seq_len))
        df = pd.DataFrame(
            {
                "x1": x1_bias + np.random.randn(seq_len) * x1_scale,
                "x2": np.random.randn(seq_len),
                "x3": np.random.randn(seq_len),
            }
        )

        df["y1"] = df["x1"].rolling(window=1000, min_periods=1).std().fillna(0.0)

        df["x2x3"] = df["x2"] * df["x3"]
        halflife_val = 10.0
        alpha = 0.5 ** (1 / halflife_val)
        df["ema_x2x3"] = df["x2x3"].ewm(alpha=alpha, adjust=True).mean()
        df["y2"] = df["x1"] + df["ema_x2x3"]

        df["ema_x2"] = df["x2"].ewm(alpha=alpha, adjust=True).mean()
        df["ema_x3"] = df["x3"].ewm(alpha=alpha, adjust=True).mean()
        df["ema_x2x3_2"] = df["ema_x2"] * df["ema_x3"]
        df["y3"] = df["ema_x2x3_2"].ewm(alpha=alpha, adjust=True).mean()

        df["y1"] = df["y1"] * 0
        df["y2"] = df["y2"] * 0
        # df["y3"] = df["y3"] * 0

        x_list.append(df[["x1", "x2", "x3"]].values.T)
        y_list.append(df[["y1", "y2", "y3"]].values.T)

    x = torch.tensor(np.stack(x_list), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32)
    y -= y.mean(dim=(0, 2), keepdim=True)
    y /= y.std(dim=(0, 2), keepdim=True) + 1e-8
    print("Std of y:", y.std(dim=(0, 2), keepdim=True))

    print(f"\nChecking for NaNs in datasets:")
    print(f"x has NaN: {torch.isnan(x).any().item()}")
    print(f"y has NaN: {torch.isnan(y).any().item()}")
    if torch.isnan(y).any():
        nan_mask = torch.isnan(y)
        print(f"y NaN locations: {nan_mask.nonzero()}")
        print(f"y NaN count: {nan_mask.sum().item()}")

    train_size = int(0.8 * batch_size)
    x_train, x_val = x[:train_size], x[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = MLPEMA(
        num_features=3,
        dim_lists=[
            [3, 64, 64],
            [64, 64, 4],
        ],
        halflife_bounds=(50.0, 50.0),
        out_features=3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

    print("\nFitting MLPEMA to hard dataset:")
    print("y1 = running_std(x1)")
    print("y2 = x1 + ema(x2 * x3)")
    print("y3 = ema(ema(x2) * ema(x3))")
    print(f"Before training - Validation RMSE: {test_regression(model, val_loader, use_rmse=True):.6f}")

    best_loss, best_epoch = fit_regression(
        model,
        optimizer,
        train_loader,
        val_loader,
        max_epochs=300,
        use_rmse=True,
        clip_grad_norm=0.5,
    )

    print(f"\nBest validation loss: {best_loss:.6f} at epoch {best_epoch}")
    print(f"Final validation RMSE: {test_regression(model, val_loader, use_rmse=True):.6f}")


if __name__ == "__main__":
    # test_timeseries()
    # test_adaptive_ema()
    # test_fit_adaptive_ema()
    # test_fit_mlpema()
    test_fit_mlpema_hard()

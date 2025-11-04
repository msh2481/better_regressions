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
    dt = torch.full(
        (
            1,
            1,
            n,
        ),
        0.1,
        requires_grad=True,
    )
    halflife = torch.full((1,), 0.1, requires_grad=True)

    print("Input x:", x.detach().numpy())
    print("Input dt:", dt.detach().numpy())
    print("Input halflife:", halflife.detach().numpy())

    result = ema(x, dt, halflife)
    print("\nEMA output:", result.detach().numpy())

    loss = result[0, 0, -1]
    loss.backward()

    print("\nGradients:")
    print("x.grad:", x.grad.numpy() if x.grad is not None else None)
    print("dt.grad:", dt.grad.numpy() if dt.grad is not None else None)
    print("halflife.grad:", halflife.grad.numpy() if halflife.grad is not None else None)

    assert x.grad is not None, "Gradients should flow to x"
    assert dt.grad is not None, "Gradients should flow to dt"
    assert halflife.grad is not None, "Gradients should flow to halflife"

    assert not torch.isnan(x.grad).any(), "x.grad should not contain NaNs"
    assert not torch.isnan(dt.grad).any(), "dt.grad should not contain NaNs"
    assert not torch.isnan(halflife.grad).any(), "halflife.grad should not contain NaNs"

    print("\n✓ All gradients flow correctly!")


def test_adaptive_ema():
    torch.manual_seed(42)

    batch_size, num_features, seq_len = 4, 3, 10
    x = torch.randn(batch_size, num_features, seq_len)
    t = torch.cumsum(torch.ones(batch_size, seq_len) * 0.1, dim=-1)

    model = AdaptiveEMA(num_features, halflife_bounds=(0.1, 10.0))
    result = model(x, t)

    print(f"\nAdaptiveEMA output shape: {result.shape}")
    print(f"Output sample: {result[0, 0, :5].detach().numpy()}")

    loss = result.sum()
    loss.backward()

    assert model.log_halflife.grad is not None, "Gradients should flow to log_halflife"
    assert model.power.grad is not None, "Gradients should flow to power"

    print("✓ AdaptiveEMA gradients flow correctly!")


def test_fit_adaptive_ema():
    torch.manual_seed(0)
    
    batch_size, num_features, seq_len = 32, 3, 50
    
    x = torch.randn(batch_size, num_features, seq_len)
    t = torch.cumsum(torch.exp(torch.randn(batch_size, seq_len)) * 0.1, dim=-1)
    
    y = torch.zeros_like(x)
    y[:, 0, :] = x[:, 0, :]
    
    dt_index = torch.ones(batch_size, 1, seq_len)
    halflife_index = torch.tensor([3.0])
    y[:, 1:2, :] = ema(x[:, 1:2, :], dt_index, halflife_index)
    
    dt_time = torch.cat([torch.ones(batch_size, 1, 1), torch.diff(t.unsqueeze(1).expand(batch_size, 1, seq_len), dim=-1)], dim=-1)
    halflife_time = torch.tensor([5.0])
    y[:, 2:3, :] = ema(x[:, 2:3, :], dt_time, halflife_time)
    
    train_size = int(0.8 * batch_size)
    x_train, x_val = x[:train_size], x[train_size:]
    t_train, t_val = t[:train_size], t[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    train_dataset = TensorDataset(x_train, t_train, y_train)
    val_dataset = TensorDataset(x_val, t_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = AdaptiveEMA(num_features, halflife_bounds=(0.01, 20.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    print("\nBefore training:")
    halflifes = torch.exp(model.log_halflife.detach())
    powers = torch.clamp(model.power.detach(), 1e-3, 1.0 - 1e-3)
    print(f"Halflifes: {halflifes.numpy()}")
    print(f"Powers: {powers.numpy()}")
    
    best_loss, best_epoch = fit_regression(
        model, optimizer, train_loader, val_loader, max_epochs=1000, use_rmse=True
    )
    
    print(f"\nBest validation loss: {best_loss:.6f} at epoch {best_epoch}")
    
    print("\nAfter training:")
    halflifes = torch.exp(model.log_halflife.detach())
    powers = torch.clamp(model.power.detach(), 1e-3, 1.0 - 1e-3)
    print(f"Halflifes: {halflifes.numpy()}")
    print(f"Powers: {powers.numpy()}")
    print(f"\nExpected:")
    print(f"Feature 0 (last x): halflife ~ 0.01")
    print(f"Feature 1 (ema span=3, index space): halflife ~ 3.0, power ~ 0.0 (uses dt^0=1)")
    print(f"Feature 2 (ema span=5, time space): halflife ~ 5.0, power ~ 1.0 (uses dt^1=dt)")


def test_fit_mlpema():
    torch.manual_seed(42)
    
    batch_size, num_features, seq_len = 32, 1, 100
    x = torch.randn(batch_size, num_features, seq_len)
    t = torch.cumsum(torch.exp(torch.randn(batch_size, seq_len)) * 0.1, dim=-1)
    
    dt = torch.cat([torch.ones(batch_size, num_features, 1), torch.diff(t.unsqueeze(1).expand(batch_size, num_features, seq_len), dim=-1)], dim=-1)
    halflife = torch.tensor([1000.0])
    y = ema(x**2, dt, halflife)
    
    train_size = int(0.8 * batch_size)
    x_train, x_val = x[:train_size], x[train_size:]
    t_train, t_val = t[:train_size], t[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    train_dataset = TensorDataset(x_train, t_train, y_train)
    val_dataset = TensorDataset(x_val, t_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = MLPEMA(
        num_features=1,
        dim_lists=[[1, 32, 4]],
        halflife_bounds=(0.1, 100.0),
        out_features=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    print("\nFitting MLPEMA to y = ema(x^2, halflife=1000)")
    print(f"Before training - Validation RMSE: {test_regression(model, val_loader, use_rmse=True):.6f}")
    
    best_loss, best_epoch = fit_regression(
        model, optimizer, train_loader, val_loader, max_epochs=200, use_rmse=True
    )
    
    print(f"\nBest validation loss: {best_loss:.6f} at epoch {best_epoch}")
    print(f"Final validation RMSE: {test_regression(model, val_loader, use_rmse=True):.6f}")


def test_fit_mlpema_hard():
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size, num_features, seq_len = 64, 3, 200
    
    x_list = []
    t_list = []
    y_list = []
    
    for b in range(batch_size):
        df = pd.DataFrame({
            'x1': np.random.randn(seq_len),
            'x2': np.random.randn(seq_len),
            'x3': np.random.randn(seq_len),
        })
        df['t'] = np.cumsum(np.exp(np.random.randn(seq_len)) * 0.1)
        
        df['y1'] = df['x1'].rolling(window=20, min_periods=1).std()
        
        df['x2x3'] = df['x2'] * df['x3']
        dt = df['t'].diff().fillna(1.0)
        halflife_val = 10.0
        alpha = 0.5 ** (1 / halflife_val)
        df['ema_x2x3'] = df['x2x3'].ewm(alpha=alpha, adjust=True).mean()
        df['y2'] = df['x1'] + df['ema_x2x3']
        
        df['ema_x2'] = df['x2'].ewm(alpha=alpha, adjust=True).mean()
        df['ema_x3'] = df['x3'].ewm(alpha=alpha, adjust=True).mean()
        df['ema_x2x3_2'] = df['ema_x2'] * df['ema_x3']
        df['y3'] = df['ema_x2x3_2'].ewm(alpha=alpha, adjust=True).mean()
        
        x_list.append(df[['x1', 'x2', 'x3']].values.T)
        t_list.append(df['t'].values)
        y_list.append(df[['y1', 'y2', 'y3']].values.T)
    
    x = torch.tensor(np.stack(x_list), dtype=torch.float32)
    t = torch.tensor(np.stack(t_list), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32)
    
    train_size = int(0.8 * batch_size)
    x_train, x_val = x[:train_size], x[train_size:]
    t_train, t_val = t[:train_size], t[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    train_dataset = TensorDataset(x_train, t_train, y_train)
    val_dataset = TensorDataset(x_val, t_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = MLPEMA(
        num_features=3,
        dim_lists=[[3, 64, 32], [32, 16, 4]],
        halflife_bounds=(0.1, 100.0),
        out_features=3
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    print("\nFitting MLPEMA to hard dataset:")
    print("y1 = running_std(x1)")
    print("y2 = x1 + ema(x2 * x3)")
    print("y3 = ema(ema(x2) * ema(x3))")
    print(f"Before training - Validation RMSE: {test_regression(model, val_loader, use_rmse=True):.6f}")
    
    best_loss, best_epoch = fit_regression(
        model, optimizer, train_loader, val_loader, max_epochs=300, use_rmse=True
    )
    
    print(f"\nBest validation loss: {best_loss:.6f} at epoch {best_epoch}")
    print(f"Final validation RMSE: {test_regression(model, val_loader, use_rmse=True):.6f}")


if __name__ == "__main__":
    # test_timeseries()
    # test_adaptive_ema()
    # test_fit_adaptive_ema()
    # test_fit_mlpema()
    test_fit_mlpema_hard()

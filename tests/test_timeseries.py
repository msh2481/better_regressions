import torch
from torch.utils.data import DataLoader, TensorDataset

from better_regressions.timeseries import AdaptiveEMA, ema, fit_regression


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
    t = torch.cumsum(torch.ones_like(x) * 0.1, dim=-1)

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
    t = torch.cumsum(torch.randn_like(x).exp() * 0.1, dim=-1)
    
    y = torch.zeros_like(x)
    y[:, 0, :] = x[:, 0, :]
    
    dt_index = torch.ones(batch_size, 1, seq_len)
    halflife_index = torch.tensor([3.0])
    y[:, 1:2, :] = ema(x[:, 1:2, :], dt_index, halflife_index)
    
    dt_time = torch.cat([torch.ones_like(t[:, 2:3, :1]), torch.diff(t[:, 2:3, :], dim=-1)], dim=-1)
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


if __name__ == "__main__":
    # test_timeseries()
    # test_adaptive_ema()
    test_fit_adaptive_ema()

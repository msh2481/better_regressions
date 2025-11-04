import torch

from better_regressions.timeseries import AdaptiveEMA, ema


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
    assert model.logit_power.grad is not None, "Gradients should flow to logit_power"

    print("✓ AdaptiveEMA gradients flow correctly!")


if __name__ == "__main__":
    test_timeseries()
    test_adaptive_ema()

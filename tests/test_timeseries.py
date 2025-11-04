import torch

from better_regressions.timeseries import ema


def test_timeseries():
    torch.manual_seed(42)
    
    n = 10
    x_value = torch.zeros(n)
    x_value[n//2] = 1.0
    x = x_value.detach().clone().requires_grad_(True)
    dt = torch.full((n,), 0.1, requires_grad=True)
    halflife = torch.tensor(0.1, requires_grad=True)
    
    print("Input x:", x.detach().numpy())
    print("Input dt:", dt.detach().numpy())
    print("Input halflife:", halflife.item())
    
    result = ema(x, dt, halflife)
    print("\nEMA output:", result.detach().numpy())
    
    loss = result[-1]
    loss.backward()
    
    print("\nGradients:")
    print("x.grad:", x.grad.numpy() if x.grad is not None else None)
    print("dt.grad:", dt.grad.numpy() if dt.grad is not None else None)
    print("halflife.grad:", halflife.grad.item() if halflife.grad is not None else None)
    
    assert x.grad is not None, "Gradients should flow to x"
    assert dt.grad is not None, "Gradients should flow to dt"
    assert halflife.grad is not None, "Gradients should flow to halflife"
    
    assert not torch.isnan(x.grad).any(), "x.grad should not contain NaNs"
    assert not torch.isnan(dt.grad).any(), "dt.grad should not contain NaNs"
    assert not torch.isnan(halflife.grad), "halflife.grad should not be NaN"
    
    print("\n✓ All gradients flow correctly!")


if __name__ == "__main__":
    test_timeseries()


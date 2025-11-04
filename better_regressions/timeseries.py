import torch


def ema(x: torch.Tensor, dt: torch.Tensor, halflife: torch.Tensor) -> torch.Tensor:
    alpha = 0.5**(1/halflife)
    n = x.shape[0]
    result = torch.zeros_like(x)
    weighted_sum = torch.zeros_like(x[0])
    total_coeff = torch.zeros_like(x[0])
    
    for i in range(n):
        decay = alpha ** dt[i]
        weighted_sum = weighted_sum * decay + x[i]
        total_coeff = total_coeff * decay + 1.0
        result[i] = weighted_sum / (total_coeff + 1e-10)
    
    return result


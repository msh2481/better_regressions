import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseKANLayer(nn.Module):
    def __init__(self, input_size: int, num_repu_terms: int = 4, repu_order: int = 2, eps: float = 0.01):
        super().__init__()
        self.input_size = input_size
        self.num_repu_terms = num_repu_terms
        self.repu_order = repu_order
        self.silu = nn.SiLU()
        self.coefficients = nn.Parameter(torch.randn(input_size, num_repu_terms) * eps)
        self.biases = nn.Parameter(torch.randn(input_size, num_repu_terms))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_out = self.silu(x)
        x_expanded = x.unsqueeze(-1)
        shifted = x_expanded + self.biases
        repu_terms = F.relu(shifted) ** self.repu_order
        linear_comb = torch.sum(self.coefficients.unsqueeze(0) * repu_terms, dim=-1)
        return silu_out + linear_comb


class PointwiseKAN(nn.Module):
    def __init__(self, dim_list: list[int], num_repu_terms: int = 4, repu_order: int = 2, eps: float = 0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dim_list) - 1):
            linear = nn.Linear(dim_list[i], dim_list[i + 1])
            nn.init.kaiming_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            self.layers.append(linear)
            if i < len(dim_list) - 2:
                self.layers.append(PointwiseKANLayer(dim_list[i + 1], num_repu_terms, repu_order, eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

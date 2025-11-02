import torch
import torch.nn as nn


class OrdinaryMLP(nn.Module):
    def __init__(self, dim_list: list[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dim_list) - 1):
            linear = nn.Linear(dim_list[i], dim_list[i + 1])
            nn.init.kaiming_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            self.layers.append(linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.silu(x)
        return x


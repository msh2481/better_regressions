import torch
import torch.nn as nn


class ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int):
        super().__init__()
        self.g, self.k, self.r = g, k, 4 * g * g / ((k + 1) * (k + 1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = torch.arange(-k, g, dtype=torch.float32) / g
        phase_height = phase_low + (k + 1) / g
        self.phase_low = nn.Parameter(
            phase_low.unsqueeze(0).repeat(input_size, 1), requires_grad=True
        )
        self.phase_height = nn.Parameter(
            phase_height.unsqueeze(0).repeat(input_size, 1), requires_grad=True
        )
        self.fc = nn.Linear((g + k) * input_size, output_size)
        nn.init.kaiming_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, f"Expected 2D input (batch, input_size), got {x.shape}"
        assert x.shape[1] == self.input_size, f"Input size mismatch: got {x.shape[1]}, expected {self.input_size}"
        assert self.phase_low.shape == (self.input_size, self.g + self.k), f"phase_low shape: {self.phase_low.shape}, expected ({self.input_size}, {self.g + self.k})"
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_height - x)
        x = x1 * x2 * self.r
        x = x * x
        assert x.shape == (x.shape[0], self.input_size, self.g + self.k), f"After phase ops: {x.shape}"
        x = x.reshape(x.shape[0], (self.g + self.k) * self.input_size)
        x = self.fc(x)
        assert x.shape == (x.shape[0], self.output_size), f"After fc: {x.shape}"
        return x


class ReLUKAN(nn.Module):
    def __init__(self, width: list[int], grid: int = 5, k: int = 3):
        super().__init__()
        self.width = width
        self.grid = grid
        self.k = k
        self.rk_layers = nn.ModuleList()
        for i in range(len(width) - 1):
            self.rk_layers.append(ReLUKANLayer(width[i], grid, k, width[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, f"Initial input must be 2D, got {x.shape}"
        for i, rk_layer in enumerate(self.rk_layers):
            assert x.dim() == 2, f"Before layer {i}: {x.shape}"
            assert x.shape[1] == self.width[i], f"Layer {i} input size mismatch: got {x.shape[1]}, expected {self.width[i]}"
            x = rk_layer(x)
        return x

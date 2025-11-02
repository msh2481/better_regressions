import torch
import torch.nn as nn


class RePU(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** self.n


class ResSiLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.kaiming_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.silu(x)
        out = self.fc(out)
        return out


class ResRePUBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, repu_order: int, res: bool = True):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.repu = RePU(repu_order)
        if res:
            self.res = ResSiLU(input_dim, output_dim)
        else:
            self.res = nn.Identity()

        nn.init.kaiming_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res(x)
        out = self.fc(x)
        out = self.repu(out)
        out += residual
        return out


class PowerMLP(nn.Module):
    def __init__(self, dim_list: list[int], repu_order: int, res: bool = True):
        super().__init__()
        res_block_list = []
        for i in range(len(dim_list) - 2):
            res_block = ResRePUBlock(dim_list[i], dim_list[i + 1], repu_order, res=res)
            res_block_list.append(res_block)
        self.res_layers = nn.ModuleList(res_block_list)
        self.fc = nn.Linear(dim_list[-2], dim_list[-1])
        nn.init.kaiming_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res_layer in self.res_layers:
            x = res_layer(x)
        x = self.fc(x)
        return x

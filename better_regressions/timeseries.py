import math
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from better_regressions.mlp import MLP, RunningRMSNorm


def _ema_impl(x_flat: torch.Tensor, halflife: torch.Tensor, max_size: int = 200) -> torch.Tensor:
    batch_features, seq_len = x_flat.shape
    alpha = 0.5 ** (1 / halflife)
    
    kernel_size = min(max_size + 1, seq_len)
    arange = torch.arange(kernel_size, device=x_flat.device, dtype=x_flat.dtype).flip(0)
    kernel_weights = (alpha.unsqueeze(1) ** arange.unsqueeze(0)).unsqueeze(1)
    kernel = kernel_weights
    
    x_reshaped = x_flat.unsqueeze(0)
    result = F.conv1d(x_reshaped, kernel, padding=kernel_size - 1, groups=batch_features)
    result = result.squeeze(0)[:, :seq_len]
    
    ones_input = torch.ones_like(x_reshaped)
    total_weight = F.conv1d(ones_input, kernel, padding=kernel_size - 1, groups=batch_features)
    total_weight = total_weight.squeeze(0)[:, :seq_len]
    
    result = result / (total_weight + 1e-8)
    return result

def ema(x: Float[Tensor, "... seq"], halflife: Float[Tensor, "..."]) -> Float[Tensor, "... seq"]:
    assert x.ndim >= 2, "x must have at least 2 dimensions"
    original_shape = x.shape
    x_flat = x.flatten(0, -2)
    halflife_expanded = halflife.expand(x_flat.shape[:-1]).flatten()
    result_flat = _ema_impl(x_flat, halflife_expanded)
    return result_flat.reshape(original_shape)


class AdaptiveEMA(nn.Module):
    def __init__(self, num_features: int, halflife_bounds: tuple[float, float]):
        super().__init__()
        self.num_features = num_features
        log_halflife_min = torch.log(torch.tensor(halflife_bounds[0]))
        log_halflife_max = torch.log(torch.tensor(halflife_bounds[1]))
        self.log_halflife = nn.Parameter(torch.rand(num_features) * (log_halflife_max - log_halflife_min) + log_halflife_min)

    def forward(self, x: Float[Tensor, "batch features seq"]) -> Float[Tensor, "batch features seq"]:
        batch_size, num_features, seq_len = x.shape
        halflife = torch.exp(self.log_halflife.view(1, -1).expand(batch_size, -1).flatten())
        return ema(x, halflife)


class FeaturewiseMLP(nn.Module):
    def __init__(self, num_features: int, dim_list: list[int], **mlp_kwargs):
        super().__init__()
        self.num_features = num_features
        self.mlp = MLP(dim_list, **mlp_kwargs)

    def forward(self, x: Float[Tensor, "batch features seq"]) -> Float[Tensor, "batch features seq"]:
        batch_size, num_features, seq_len = x.shape
        x_flat = x.permute(0, 2, 1).reshape(batch_size * seq_len, num_features)
        y_flat = self.mlp(x_flat)
        y = y_flat.reshape(batch_size, seq_len, -1).permute(0, 2, 1)
        return y


class MLPEMABlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        dim_list: list[int],
        halflife_bounds: tuple[float, float],
        out_features: int,
        **mlp_kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.mlp_out_features = dim_list[-1]
        self.out_features = out_features
        assert dim_list[0] == in_features, "First dimension of dim_list must be equal to in_features"
        self.mlp = FeaturewiseMLP(self.in_features, dim_list, **mlp_kwargs)
        self.ema_input = AdaptiveEMA(self.in_features, halflife_bounds)
        self.ema_output = AdaptiveEMA(self.mlp_out_features, halflife_bounds)
        self.ema_var = AdaptiveEMA(self.in_features, halflife_bounds)
        self.norm = RunningRMSNorm(self.in_features + self.mlp_out_features * 2)
        self.combined_linear = nn.Linear(self.in_features + self.mlp_out_features * 2, self.out_features * 2)

    def forward(self, x: Float[Tensor, "batch features seq"]) -> tuple[Float[Tensor, "batch features seq"], Float[Tensor, "batch out_features seq"]]:
        ema_mlp = self.ema_output(self.mlp(x))
        mlp_ema = self.mlp(self.ema_input(x))
        ema_var = self.ema_var(x ** 2) - self.ema_var(x) ** 2
        batch_size, out_features, seq_len = ema_mlp.shape
        combined = torch.cat([ema_mlp * 0, mlp_ema * 0, ema_var], dim=1)
        combined_flat = combined.permute(0, 2, 1).reshape(batch_size * seq_len, self.in_features + self.mlp_out_features * 2)
        combined_norm = self.norm(combined_flat)
        linear_output_flat = self.combined_linear(combined_norm)
        skip_y_flat, output_flat = torch.split(linear_output_flat, self.out_features, dim=1)
        skip_y = skip_y_flat.reshape(batch_size, seq_len, -1).permute(0, 2, 1)
        output = output_flat.reshape(batch_size, seq_len, -1).permute(0, 2, 1)
        return output, skip_y


class MLPEMA(nn.Module):
    def __init__(
        self,
        num_features: int,
        dim_lists: list[list[int]],
        halflife_bounds: tuple[float, float],
        out_features: int,
        **mlp_kwargs
    ):
        super().__init__()
        self.num_features = num_features
        self.out_features = out_features
        self.blocks = nn.ModuleList()
        
        current_features = num_features
        self.skip_linears = nn.ModuleList()
        self.skip_linears.append(nn.Linear(num_features, out_features))
        
        for dim_list in dim_lists:
            block = MLPEMABlock(current_features, dim_list, halflife_bounds, out_features, **mlp_kwargs)
            self.blocks.append(block)
            current_features = dim_list[-1] if dim_list else current_features
        
        self.final_ema = AdaptiveEMA(out_features, halflife_bounds)

    def forward(self, x: Float[Tensor, "batch features seq"]) -> Float[Tensor, "batch features seq"]:
        batch_size, _, seq_len = x.shape
        
        x_flat = x.permute(0, 2, 1).reshape(batch_size * seq_len, self.num_features)
        y_flat = self.skip_linears[0](x_flat)
        y = y_flat.reshape(batch_size, seq_len, -1).permute(0, 2, 1)
        
        for block in self.blocks:
            x, skip_y = block(x)
            y = y + skip_y
        
        y = self.final_ema(y)
        return y

    def extra_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0)
        device = next(self.parameters()).device
        total_loss = total_loss.to(device)
        
        for block in self.blocks:
            if hasattr(block.mlp.mlp, "extra_loss"):
                total_loss = total_loss + block.mlp.mlp.extra_loss()
        
        return total_loss


class LRScheduler(Protocol):
    def step(self) -> None: ...


def _get_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def fit_regression(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    max_epochs: int = 100,
    scheduler: LRScheduler | None = None,
    save_path: str | None = None,
    use_rmse: bool = True,
    clip_grad_norm: float | None = None,
) -> tuple[float, int]:
    device = _get_device(model)
    best_loss = float("inf")
    best_epoch = 0
    pbar = tqdm(range(max_epochs), desc="Training", ncols=100)

    for epoch in pbar:
        model.train()
        for x, targets in train_loader:
            x, targets = x.to(device), targets.to(device)
            outputs = model(x)

            optimizer.zero_grad()
            loss = ((outputs - targets) ** 2).mean()

            if hasattr(model, "extra_loss"):
                extra = model.extra_loss()
                loss = loss + extra

            loss.backward()

            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

        model.eval()
        train_loss = 0.0
        train_batches = 0
        with torch.no_grad():
            for x, targets in train_loader:
                x, targets = x.to(device), targets.to(device)
                outputs = model(x)
                loss = ((outputs - targets) ** 2).mean()
                train_loss += loss.item()
                train_batches += 1

        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, targets in val_loader:
                x, targets = x.to(device), targets.to(device)
                outputs = model(x)
                loss = ((outputs - targets) ** 2).mean()
                val_loss += loss.item()
                val_batches += 1

        if math.isnan(val_loss):
            return float("inf"), epoch

        train_loss = train_loss / train_batches
        val_loss = val_loss / val_batches
        if use_rmse:
            train_loss = math.sqrt(train_loss)
            val_loss = math.sqrt(val_loss)

        pbar.set_description(f"lr: {optimizer.param_groups[0]['lr']:.2e} | train: {train_loss:.2f} | val: {val_loss:.2f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            if save_path:
                torch.save(model.state_dict(), save_path)

        if scheduler:
            scheduler.step()

    return float(best_loss), best_epoch


def test_regression(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    use_rmse: bool = True,
) -> float:
    device = _get_device(model)
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, targets in test_loader:
            x, targets = x.to(device), targets.to(device)
            outputs = model(x)
            loss = ((outputs - targets) ** 2).mean()
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    if use_rmse:
        return float(math.sqrt(avg_loss))
    return float(avg_loss)

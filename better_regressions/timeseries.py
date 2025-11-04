import math
from typing import Protocol

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from better_regressions.mlp import MLP


def ema(x: Float[Tensor, "... seq"], dt: Float[Tensor, "... seq"], halflife: Float[Tensor, "..."]) -> Float[Tensor, "... seq"]:
    assert x.ndim >= 2, "x must have at least 2 dimensions"
    assert x.shape == dt.shape, "x and dt must have the same shape"
    original_shape = x.shape
    x_flat = x.flatten(0, -2)
    dt_flat = dt.flatten(0, -2)
    halflife = halflife.expand(x_flat.shape[:-1]).flatten()

    alpha = 0.5 ** (1 / halflife)

    result = torch.zeros_like(x_flat)
    weighted_sum = torch.zeros_like(x_flat[:, 0])
    total_coeff = torch.zeros_like(x_flat[:, 0])

    for i in range(x_flat.shape[1]):
        decay = alpha ** dt_flat[:, i]
        weighted_sum = weighted_sum * decay + x_flat[:, i]
        total_coeff = total_coeff * decay + 1.0
        result[:, i] = weighted_sum / (total_coeff + 1e-10)

    return result.reshape(original_shape)


class AdaptiveEMA(nn.Module):
    def __init__(self, num_features: int, halflife_bounds: tuple[float, float]):
        super().__init__()
        self.num_features = num_features
        log_halflife_min = torch.log(torch.tensor(halflife_bounds[0]))
        log_halflife_max = torch.log(torch.tensor(halflife_bounds[1]))
        self.log_halflife = nn.Parameter(torch.rand(num_features) * (log_halflife_max - log_halflife_min) + log_halflife_min)
        self.power = nn.Parameter(torch.full((num_features,), 0.5))
        self.register_buffer("running_sum_dt", torch.zeros(num_features))
        self.register_buffer("dt_count", torch.tensor(0.0))

    def forward(self, x: Float[Tensor, "batch features seq"], t: Float[Tensor, "batch features seq"]) -> Float[Tensor, "batch features seq"]:
        self.power.data.clamp_(1e-3, 1.0 - 1e-3)
        batch_size, num_features, seq_len = x.shape
        dt = torch.cat([torch.ones_like(t[:, :, :1]), torch.diff(t, dim=-1)], dim=-1)
        
        if self.training:
            sum_dt = torch.sum(dt.detach(), dim=(0, 2))
            self.running_sum_dt += sum_dt
            self.dt_count += batch_size * seq_len
        
        mean_dt = self.running_sum_dt / (self.dt_count + 1e-10)
        dt_normalized = dt / mean_dt.view(1, num_features, 1)
        
        dt_powered = dt_normalized ** self.power.view(1, num_features, 1)
        halflife = torch.exp(self.log_halflife.expand(1, batch_size, num_features).flatten())
        return ema(x, dt_powered, halflife)


class FeaturewiseMLP(nn.Module):
    def __init__(self, num_features: int, dim_list: list[int], **mlp_kwargs):
        super().__init__()
        self.num_features = num_features
        self.mlp = MLP(dim_list, **mlp_kwargs)

    def forward(self, x: Float[Tensor, "batch features seq"], t: Float[Tensor, "batch features seq"]) -> Float[Tensor, "batch features seq"]:
        batch_size, num_features, seq_len = x.shape
        x_flat = x.permute(0, 2, 1).reshape(batch_size * seq_len, num_features)
        y_flat = self.mlp(x_flat)
        y = y_flat.reshape(batch_size, seq_len, -1).permute(0, 2, 1)
        return y


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
        self.blocks = nn.ModuleList()
        
        current_features = num_features
        for dim_list in dim_lists:
            mlp = FeaturewiseMLP(current_features, dim_list, **mlp_kwargs)
            self.blocks.append(mlp)
            current_features = dim_list[-1] if dim_list else current_features
            ema = AdaptiveEMA(current_features, halflife_bounds)
            self.blocks.append(ema)
        self.final_linear = nn.Linear(current_features, out_features)
        self.final_ema = AdaptiveEMA(out_features, halflife_bounds)

    def forward(self, x: Float[Tensor, "batch features seq"], t: Float[Tensor, "batch features seq"]) -> Float[Tensor, "batch features seq"]:
        for block in self.blocks:
            x = block(x, t)
        
        batch_size, num_features, seq_len = x.shape
        x_flat = x.permute(0, 2, 1).reshape(batch_size * seq_len, num_features)
        y_flat = self.final_linear(x_flat)
        y = y_flat.reshape(batch_size, seq_len, -1).permute(0, 2, 1)
        y = self.final_ema(y, t)
        return y

    def extra_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0)
        device = next(self.parameters()).device
        total_loss = total_loss.to(device)
        
        for block in self.blocks:
            if isinstance(block, FeaturewiseMLP) and hasattr(block.mlp, "extra_loss"):
                total_loss = total_loss + block.mlp.extra_loss()
        
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
        for x, t, targets in train_loader:
            x, t, targets = x.to(device), t.to(device), targets.to(device)
            outputs = model(x, t)

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
            for x, t, targets in train_loader:
                x, t, targets = x.to(device), t.to(device), targets.to(device)
                outputs = model(x, t)
                loss = ((outputs - targets) ** 2).mean()
                train_loss += loss.item()
                train_batches += 1

        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, t, targets in val_loader:
                x, t, targets = x.to(device), t.to(device), targets.to(device)
                outputs = model(x, t)
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
        for x, t, targets in test_loader:
            x, t, targets = x.to(device), t.to(device), targets.to(device)
            outputs = model(x, t)
            loss = ((outputs - targets) ** 2).mean()
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    if use_rmse:
        return float(math.sqrt(avg_loss))
    return float(avg_loss)

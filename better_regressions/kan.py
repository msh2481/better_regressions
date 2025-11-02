import math
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class RunningRMSNorm(nn.Module):
    def __init__(self, num_features: int, span: int = 1000, eps: float = 1e-8):
        super().__init__()
        self.num_features = num_features
        self.span = span
        self.eps = eps
        self.alpha = 1.0 / span
        self.register_buffer("running_mean_sq", torch.zeros(num_features))
        self.register_buffer("count", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean_sq = torch.mean(x.detach() ** 2, dim=0)
            self.running_mean_sq = (1 - self.alpha) * self.running_mean_sq + self.alpha * mean_sq
            self.count += 1
        
        bias_correction = 1 - (1 - self.alpha) ** self.count.item()
        adjusted_mean_sq = self.running_mean_sq / (bias_correction + self.eps)
        rms = torch.sqrt(adjusted_mean_sq + self.eps)
        return x / rms.unsqueeze(0)


class PointwiseKANLayer(nn.Module):
    def __init__(self, input_size: int, num_repu_terms: int = 8, repu_order: int = 2, eps: float = 0.01):
        super().__init__()
        self.input_size = input_size
        self.num_repu_terms = num_repu_terms
        self.repu_order = repu_order
        self.base_activation = nn.ReLU()
        self.norm = RunningRMSNorm(input_size)
        self.coefficients = nn.Parameter(torch.randn(input_size, num_repu_terms) * eps)
        self.biases = nn.Parameter(torch.randn(input_size, num_repu_terms))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_activation(x)
        x = self.norm(x)
        x_expanded = x.unsqueeze(-1)
        shifted = x_expanded + self.biases
        repu_terms = F.relu(shifted) ** self.repu_order
        linear_comb = torch.sum(self.coefficients.unsqueeze(0) * repu_terms, dim=-1)
        return base_out + linear_comb



class MLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        pointwise: bool = False,
        num_repu_terms: int = 4,
        repu_order: int = 2,
        eps: float = 0.01,
        residual: bool = False,
        use_norm: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.residual = residual and in_features == out_features
        if self.residual:
            nn.init.normal_(self.linear.weight, std=1e-6)
        
        if use_norm:
            self.norm = RunningRMSNorm(in_features)
        else:
            self.norm = None
        
        if pointwise:
            self.activation = PointwiseKANLayer(out_features, num_repu_terms, repu_order, eps)
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            out = self.norm(x)
        else:
            out = x
        out = self.linear(out)
        out = self.activation(out)
        if self.residual:
            out = x + out
        return out


class MLP(nn.Module):
    def __init__(
        self,
        dim_list: list[int],
        residual: bool = False,
        pointwise: bool = False,
        num_repu_terms: int = 4,
        repu_order: int = 2,
        eps: float = 0.01,
        use_norm: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        for i in range(len(dim_list) - 2):
            block = MLPBlock(
                dim_list[i],
                dim_list[i + 1],
                pointwise=pointwise,
                num_repu_terms=num_repu_terms,
                repu_order=repu_order,
                eps=eps,
                residual=residual,
                use_norm=use_norm,
            )
            self.blocks.append(block)
        self.final_linear = nn.Linear(dim_list[-2], dim_list[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.final_linear(x)
        return x


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
    criterion: nn.Module,
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
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

        model.eval()
        train_loss = 0.0
        train_batches = 0
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                train_batches += 1

        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
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
    criterion: nn.Module | None = None,
    use_rmse: bool = True,
) -> float:
    device = _get_device(model)
    if criterion is None:
        criterion = nn.MSELoss()

    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    if use_rmse:
        return float(math.sqrt(avg_loss))
    return float(avg_loss)


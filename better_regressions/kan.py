import math
import os
from typing import Protocol

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
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


class PointwiseRELUKAN(nn.Module):
    def __init__(self, input_size: int, k: int = 8):
        super().__init__()
        self.input_size = input_size
        self.k = k
        
        quantiles = torch.linspace(1 / k, 1 - 1 / k, k)
        normal_dist = torch.distributions.Normal(0.0, 1.0)
        midpoint_values = normal_dist.icdf(quantiles)
        
        self.midpoints = nn.Parameter(midpoint_values.repeat((input_size, 1)).unsqueeze(0), requires_grad=False)
        self.radius = nn.Parameter(torch.tensor(8 / k).repeat(1, input_size, 1), requires_grad=False)
        print("Shape:", self.midpoints.shape, self.radius.shape)
        self.w = nn.Parameter(torch.randn(1, input_size, k) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2) # [batch, input, k]
        a = torch.nn.functional.relu(x - (self.midpoints - self.radius))
        b = torch.nn.functional.relu((self.midpoints + self.radius) - x)
        h = self.radius ** 2
        ab2 = (a * b / h)**2
        return (ab2 * self.w).sum(dim=-1)

class PointwiseRELU(nn.Module):
    def __init__(self, input_size: int, k: int = 8):
        super().__init__()
        self.input_size = input_size
        self.k = k
        
        quantiles = torch.linspace(1 / k, 1 - 1 / k, k)
        normal_dist = torch.distributions.Normal(0.0, 1.0)
        midpoint_values = normal_dist.icdf(quantiles)
        self.midpoints = nn.Parameter(midpoint_values.repeat((input_size, 1)).unsqueeze(0), requires_grad=False)
        print("Shape:", self.midpoints.shape)
        self.w = nn.Parameter(torch.randn(1, input_size, k) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2) # [batch, input, k]
        a = torch.abs(x - self.midpoints)
        return (a * self.w).sum(dim=-1)
    

class Copy(nn.Module):
    def __init__(self, input_dim: int, k: int):
        super().__init__()
        self.input_dim = input_dim
        self.k = k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim)
        # output shape: (batch, input_dim * k)
        return x.repeat_interleave(self.k, dim=-1)


def l1_to_max(x: torch.Tensor, norm_dim: int, eps: float = 1e-8) -> torch.Tensor:
    return (x.abs().mean(dim=norm_dim) / (x.abs().max(dim=norm_dim).values + eps)).mean()

class MLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
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
        
        self.activation = nn.GELU()
    
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
    
    def extra_loss(self, sparsity_weight: float = 0.0) -> torch.Tensor:
        if sparsity_weight <= 0:
            device = self.linear.weight.device
            return torch.tensor(0.0, device=device)
        
        weights = self.linear.weight
        return sparsity_weight * l1_to_max(weights, norm_dim=1)


class KANBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        copies: int = 4,
        k: int = 4,
        residual: bool = False,
    ):
        super().__init__()
        self.copy = Copy(in_features, copies)
        self.linear = nn.Linear(in_features * copies, out_features)
        self.residual = residual and in_features == out_features
        if self.residual:
            nn.init.normal_(self.linear.weight, std=1e-6)
        self.norm = RunningRMSNorm(in_features)
        self.activation = PointwiseRELU(in_features * copies, k)
        self._last_linear_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.copy(out)
        out = self.activation(out)
        out = self.linear(out)
        if self.training:
            self._last_linear_output = out
        else:
            self._last_linear_output = None
        if self.residual:
            out = x + out
        return out
    
    def extra_loss(self, sparsity_weight: float = 0.0, activation_sparsity_weight: float = 0.0) -> torch.Tensor:
        weights = self.linear.weight
        loss = torch.tensor(0.0, device=weights.device)
        
        if sparsity_weight > 0:
            # dim=1 because [out_features, in_features] weights shape (and we want to do loss separately for each output feature)
            loss += sparsity_weight * l1_to_max(weights, norm_dim=1)
        
        if activation_sparsity_weight > 0 and self._last_linear_output is not None:
            linear_output = self._last_linear_output
            # dim=1 because [batch, out_features] activation shape
            loss += activation_sparsity_weight * l1_to_max(linear_output, norm_dim=1)
        
        return loss


class MLP(nn.Module):
    def __init__(
        self,
        dim_list: list[int],
        residual: bool = False,
        use_norm: bool = True,
        sparsity_weight: float = 0.0,
    ):
        super().__init__()
        
        if residual and len(dim_list) > 2:
            mid = dim_list[1:-1]
            if min(mid) != max(mid):
                raise ValueError("When residual is True, the number of features in the middle layers must be the same.")

        self.sparsity_weight = sparsity_weight
        
        self.blocks = nn.ModuleList()
        for i in range(len(dim_list) - 2):
            block = MLPBlock(
                dim_list[i],
                dim_list[i + 1],
                residual=residual,
                use_norm=use_norm,
            )
            self.blocks.append(block)
        self.blocks.append(nn.Linear(dim_list[-2], dim_list[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
    
    def extra_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0)
        device = next(self.parameters()).device
        total_loss = total_loss.to(device)
        
        for block in self.blocks:
            if isinstance(block, MLPBlock):
                total_loss = total_loss + block.extra_loss(self.sparsity_weight)
            elif isinstance(block, nn.Linear):
                weights = block.weight
                total_loss = total_loss + self.sparsity_weight * l1_to_max(weights, norm_dim=1)
        
        return total_loss


class KAN(nn.Module):
    def __init__(
        self,
        dim_list: list[int],
        k: int = 4,
        copies: int = 4,
        residual: bool = False,
        sparsity_weight: float = 1e-2,
        activation_sparsity_weight: float = 1e-2,
    ):
        super().__init__()

        if residual and len(dim_list) > 2:
            mid = dim_list[1:-1]
            if min(mid) != max(mid):
                raise ValueError("When residual is True, the number of features in the middle layers must be the same.")

        self.sparsity_weight = sparsity_weight
        self.activation_sparsity_weight = activation_sparsity_weight
        
        self.blocks = nn.ModuleList()
        for i in range(len(dim_list) - 1):
            block = KANBlock(
                dim_list[i],
                dim_list[i + 1],
                k=k,
                copies=copies,
                residual=residual,
            )
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
    
    def extra_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0)
        device = next(self.parameters()).device
        total_loss = total_loss.to(device)
        
        for block in self.blocks:
            total_loss = total_loss + block.extra_loss(self.sparsity_weight, self.activation_sparsity_weight)
        
        return total_loss


def visualize_pointwise_relu_kan(block: 'KANBlock', save_path: str) -> None:
    activation = block.activation
    linear = block.linear
    copies = block.copy.k
    in_features = block.copy.input_dim
    
    midpoints = activation.midpoints.squeeze(0).detach().cpu().numpy()
    
    # Compute weight norms for each copy
    weight_norms = []
    for feat_idx in range(in_features):
        feat_norms = []
        for copy_idx in range(copies):
            copy_input_idx = feat_idx * copies + copy_idx
            norm = torch.norm(linear.weight[:, copy_input_idx]).item()
            feat_norms.append(norm)
        weight_norms.append(feat_norms)
    
    x_range = torch.linspace(-3, 3, 300)
    
    n = int(math.sqrt(in_features))
    m = math.ceil(in_features / n)
    
    fig, axes = plt.subplots(n, m, figsize=(4 * m, 3 * n))
    if in_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    with torch.no_grad():
        activation.eval()
        for feat_idx in range(in_features):
            ax = axes[feat_idx]
            x_np = x_range.numpy()
            
            # Compute min/max norms for this feature
            feat_norms = weight_norms[feat_idx]
            if feat_norms:
                min_norm = min(feat_norms)
                max_norm = max(feat_norms)
                norm_range = max_norm - min_norm if max_norm > min_norm else 1.0
            else:
                min_norm = 0
                norm_range = 1.0
            
            # Plot each copy
            for copy_idx in range(copies):
                copy_input_idx = feat_idx * copies + copy_idx
                
                # Create input tensor: only the copy_input_idx position varies
                x_tensor = torch.zeros(len(x_range), in_features * copies)
                x_tensor[:, copy_input_idx] = x_range
                
                y = activation(x_tensor)
                y_np = y[:, copy_input_idx].cpu().numpy()
                
                # Set alpha based on weight norm (normalized per-feature)
                norm = weight_norms[feat_idx][copy_idx]
                alpha = 0.01 + 0.99 * ((norm - min_norm) / norm_range) if norm_range > 0 else 0.5
                
                ax.plot(x_np, y_np, 'k-', linewidth=1.5, alpha=alpha, label=f'#{copy_idx}: {norm:.2f}')
                
                # Plot midpoints for this copy
                for k_idx in range(activation.k):
                    midpoint = midpoints[copy_input_idx, k_idx]
                    ax.axvline(midpoint, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

            ax.set_xlim(-3, 3)
            ax.set_xlabel('x')
            ax.set_ylabel(f'Activation')
            ax.set_title(f'Feature {feat_idx}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    for feat_idx in range(in_features, len(axes)):
        axes[feat_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_linear(linear: nn.Linear, save_path: str) -> None:
    weights = linear.weight.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(max(8, weights.shape[1] * 0.5), max(6, weights.shape[0] * 0.5)))
    im = ax.imshow(weights, aspect='auto', cmap='RdBu_r', vmin=-weights.std()*3, vmax=weights.std()*3)
    
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            text = ax.text(j, i, f'{weights[i, j]:.2f}', ha='center', va='center', 
                          color='black' if abs(weights[i, j]) < weights.std() else 'white',
                          fontsize=8)
    
    ax.set_xlabel('Input feature')
    ax.set_ylabel('Output feature')
    ax.set_title('Linear Weights')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_kan_block(block: KANBlock, layer_idx: int, folder_name: str) -> None:
    visualize_pointwise_relu_kan(block, os.path.join(folder_name, f'layer_{layer_idx}_activations.png'))
    visualize_linear(block.linear, os.path.join(folder_name, f'layer_{layer_idx}_weights.png'))


def visualize_kan(model: KAN, folder_name: str) -> None:
    os.makedirs(folder_name, exist_ok=True)
    for layer_idx, block in enumerate(model.blocks):
        visualize_kan_block(block, layer_idx, folder_name)


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
            
            if hasattr(model, 'extra_loss'):
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


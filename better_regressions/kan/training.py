import math
from typing import Protocol

import torch
import torch.nn as nn
from tqdm import tqdm


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
    clip_grad_norm: float | None = 1.0,
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
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                n_batches += 1

        if math.isnan(val_loss):
            return float("inf"), epoch

        val_loss = val_loss / n_batches
        if use_rmse:
            val_loss = math.sqrt(val_loss)

        pbar.set_description(f"lr: {optimizer.param_groups[0]['lr']:.4e} | val loss: {val_loss:.4e}")

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

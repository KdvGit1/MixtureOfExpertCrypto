"""CPU/GPU trainer with validation-only early stopping."""

from __future__ import annotations

import copy
import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from market_moe.models.losses import LossWeights, multitask_loss
from market_moe.models.moe import MultiTaskMoE


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 5
    gradient_clip: float = 1.0
    seed: int = 20260811
    branch_dropout_start_epoch: int = 3


@dataclass(slots=True)
class TrainingResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_loss: float
    config: dict[str, object]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: MultiTaskMoE,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig | None = None,
    loss_weights: LossWeights | None = None,
    *,
    device: str | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainingResult:
    config = config or TrainingConfig()
    loss_weights = loss_weights or LossWeights()
    seed_everything(config.seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=selected_device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float]] = []
    start_epoch = 0

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        best_state = checkpoint["best_state"]
        best_loss = float(checkpoint["best_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if stale_epochs >= config.patience:
            start_epoch = config.epochs

    for epoch in range(start_epoch, config.epochs):
        model.train()
        if epoch < config.branch_dropout_start_epoch:
            active_branch_dropout = model.branch_dropout_probability
            model.branch_dropout_probability = 0.0
        else:
            active_branch_dropout = None
        train_losses = []
        for features, target_return, target_direction, target_volatility in train_loader:
            features = features.to(selected_device)
            targets = [
                item.to(selected_device)
                for item in (target_return, target_direction, target_volatility)
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type,
                enabled=selected_device.type == "cuda",
            ):
                loss, _ = multitask_loss(
                    model(features), targets[0], targets[1], targets[2], weights=loss_weights
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach()))
        if active_branch_dropout is not None:
            model.branch_dropout_probability = active_branch_dropout

        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for features, target_return, target_direction, target_volatility in validation_loader:
                output = model(features.to(selected_device))
                targets = [
                    item.to(selected_device)
                    for item in (target_return, target_direction, target_volatility)
                ]
                loss, _ = multitask_loss(
                    output, targets[0], targets[1], targets[2], weights=loss_weights
                )
                validation_losses.append(float(loss))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_state": best_state,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                    "history": history,
                },
                temporary,
            )
            os.replace(temporary, checkpoint_path)
        if progress_callback is not None:
            progress_callback(epoch, history[-1])
        if stale_epochs >= config.patience:
            break
    model.load_state_dict(best_state)
    return TrainingResult(history, best_epoch, best_loss, asdict(config))

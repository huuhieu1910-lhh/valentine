"""Training: physics-informed branch training, ensemble-weight feedback and evaluation."""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from torch import nn

from .config import TrainConfig
from .data import DataBundle, WindowSet
from .models import PhysicsInformedEnsemble
from .physics import PhysicsCalibration, PhysicsInformedLoss


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, data: WindowSet, batch_size: int = 2048) -> torch.Tensor:
    model.eval()
    return torch.cat([model(b["x"]) for b in data.batches(batch_size)])


@torch.no_grad()
def evaluate_loss(model: nn.Module, loss_fn: PhysicsInformedLoss, data: WindowSet,
                  batch_size: int = 2048) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    for b in data.batches(batch_size):
        _, parts = loss_fn(model(b["x"]), b)
        for k, v in parts.items():
            sums[k] += v.item() * len(b["x"])
    return {k: v / len(data) for k, v in sums.items()}


def train_branch(branch: nn.Module, loss_fn: PhysicsInformedLoss, train: WindowSet, val: WindowSet,
                 cfg: TrainConfig, name: str = "branch", log: Callable[[str], None] = print) -> list[dict]:
    """Minimise L_theta for one branch with AdamW; keep the epoch with the best validation L_theta."""
    opt = torch.optim.AdamW(branch.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(cfg.seed)
    best, best_state, stale, history = math.inf, copy.deepcopy(branch.state_dict()), 0, []
    for epoch in range(1, cfg.epochs + 1):
        branch.train()
        running = 0.0
        for b in train.batches(cfg.batch_size, shuffle=True, generator=generator):
            loss, _ = loss_fn(branch(b["x"]), b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(branch.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * len(b["x"])
        val_parts = evaluate_loss(branch, loss_fn, val)
        history.append({"epoch": epoch, "train_loss": running / len(train),
                        **{f"val_{k}": v for k, v in val_parts.items()}})
        log(f"[{name}] epoch {epoch:3d}  train L {running / len(train):.4f}  val L {val_parts['total']:.4f}"
            f"  (mse {val_parts['mse']:.4f}, parabolic {val_parts['parabolic']:.5f}, ramp {val_parts['ramp']:.5f})")
        if val_parts["total"] < best - 1e-6:
            best, stale = val_parts["total"], 0
            best_state = copy.deepcopy(branch.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                log(f"[{name}] early stop, best val L {best:.4f}")
                break
    branch.load_state_dict(best_state)
    return history


def fit_ensemble_weights(model: PhysicsInformedEnsemble, loss_fn: PhysicsInformedLoss, val: WindowSet,
                         cfg: TrainConfig) -> list[float]:
    """Feedback path of Fig. 1: learn w = softmax(alpha) by minimising L_theta of the ensemble on validation.

    The branches are frozen; only the two ensemble logits are optimised.
    """
    members = torch.stack([predict(model.cnn, val), predict(model.transformer, val)], dim=-1)
    batch = val.as_batch()
    opt = torch.optim.Adam([model.ensemble.logits], lr=cfg.ensemble_lr)
    for _ in range(cfg.ensemble_steps):
        loss, _ = loss_fn(model.ensemble(members), batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.ensemble.weights.tolist()


@torch.no_grad()
def forecast(model: PhysicsInformedEnsemble, data: WindowSet, bundle: DataBundle,
             batch_size: int = 2048) -> dict[str, np.ndarray]:
    """Forecasts of both branches and the ensemble, plus the actual load, in MW."""
    model.eval()
    outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    for b in data.batches(batch_size):
        for k, v in model(b["x"]).items():
            outputs[k].append(v)
    result = {k: bundle.to_mw(torch.cat(v)) for k, v in outputs.items()}
    result["actual"] = bundle.to_mw(data.y)
    result["last_load"] = bundle.to_mw(data.x[:, -1, bundle.load_index])
    return result


def regression_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - actual
    return {
        "MAE_MW": float(np.abs(err).mean()),
        "RMSE_MW": float(np.sqrt((err ** 2).mean())),
        "MAPE_%": float(100 * np.abs(err / actual).mean()),
    }


def evaluate(model: PhysicsInformedEnsemble, data: WindowSet, bundle: DataBundle,
             calibration: PhysicsCalibration) -> dict[str, dict[str, float]]:
    """Accuracy and physics-violation rates of every member and of the ensemble."""
    fc = forecast(model, data, bundle)
    cpu = data.to("cpu")
    last_load = torch.as_tensor(fc["last_load"], dtype=torch.float32)
    metrics = {}
    for name in (*model.members, "ensemble"):
        checks = calibration.check(torch.as_tensor(fc[name], dtype=torch.float32), last_load,
                                   cpu.temp, cpu.hour, cpu.year)
        metrics[name] = {
            **regression_metrics(fc["actual"], fc[name]),
            "envelope_violation_%": 100 * checks["envelope_violation"].float().mean().item(),
            "ramp_violation_%": 100 * checks["ramp_violation"].float().mean().item(),
        }
    return metrics

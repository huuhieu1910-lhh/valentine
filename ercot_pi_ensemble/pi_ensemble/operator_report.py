"""Operator output of Fig. 1: forecast + attribution + regime flag, one row per target hour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .config import PhysicsConfig
from .data import DataBundle, WindowSet
from .explain import EnsembleExplanation
from .models import PhysicsInformedEnsemble
from .physics import PhysicsCalibration
from .train import forecast


def regime_flag(temp: np.ndarray, t0: float, deadband: float) -> np.ndarray:
    """Side of the parabolic envelope: heating below T0, cooling above, neutral within the deadband."""
    return np.where(temp < t0 - deadband, "heating", np.where(temp > t0 + deadband, "cooling", "neutral"))


def _top_drivers(contrib_mw: np.ndarray, features: tuple[str, ...], top_k: int) -> str:
    order = np.argsort(-np.abs(contrib_mw))[:top_k]
    return "; ".join(f"{features[j]} {contrib_mw[j]:+,.0f}" for j in order)


def operator_report(model: PhysicsInformedEnsemble, data: WindowSet, bundle: DataBundle,
                    calibration: PhysicsCalibration, cfg: PhysicsConfig | None = None,
                    explanation: EnsembleExplanation | None = None, top_k: int = 3) -> pd.DataFrame:
    """Forecasts in MW with physics checks, regime flag and (optionally) the top SHAP drivers.

    flags: ENVELOPE (forecast outside the parabolic envelope), RAMP (ramp above the hourly limit),
    EXTREME_TEMP (temperature outside the 1-99 % range seen in training), OK otherwise.
    """
    cfg = cfg or PhysicsConfig()
    fc = forecast(model, data, bundle)
    cpu = data.to("cpu")
    checks = calibration.check(torch.as_tensor(fc["ensemble"], dtype=torch.float32),
                               torch.as_tensor(fc["last_load"], dtype=torch.float32),
                               cpu.temp, cpu.hour, cpu.year)
    temp = cpu.temp.numpy()
    t_lo, t_hi = calibration.temp_range
    regime = regime_flag(temp, calibration.t0, cfg.regime_deadband)
    horizon = temp.shape[1]
    if explanation is not None:
        contrib_mw = [explanation.contributions("ensemble", h) * bundle.load_std for h in range(horizon)]

    rows = []
    for i in range(len(data)):
        for h in range(horizon):
            flags = [name for name, hit in (("ENVELOPE", checks["envelope_violation"][i, h]),
                                            ("RAMP", checks["ramp_violation"][i, h]),
                                            ("EXTREME_TEMP", not t_lo <= temp[i, h] <= t_hi)) if hit]
            row = {
                "timestamp": pd.Timestamp(data.times[i, h]),
                "horizon_h": h + 1,
                "forecast_mw": fc["ensemble"][i, h],
                "cnn_mw": fc["cnn"][i, h],
                "transformer_mw": fc["transformer"][i, h],
                "actual_mw": fc["actual"][i, h],
                "envelope_lower_mw": checks["lower"][i, h].item(),
                "envelope_upper_mw": checks["upper"][i, h].item(),
                "temp_c": temp[i, h],
                "regime": regime[i, h],
                "flags": "|".join(flags) or "OK",
            }
            if explanation is not None:
                row["shap_baseline_mw"] = bundle.load_mean + bundle.load_std * explanation.expected["ensemble"][h]
                row["top_drivers_mw"] = _top_drivers(contrib_mw[h][i], bundle.features, top_k)
            rows.append(row)
    return pd.DataFrame(rows)

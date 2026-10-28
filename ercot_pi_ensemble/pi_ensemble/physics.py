"""Physics-informed loss of Fig. 1: L_theta = L_MSE + lambda1 * L_parabolic + lambda2 * L_ramp.

L_parabolic keeps forecasts inside the ERCOT piecewise parabolic load-temperature envelope
(vertex T0 = 18.5 °C, separate heating and cooling arms). L_ramp keeps the hour-to-hour
change, starting from the last observed load, within the ramp limit of that hour of day.
Both are squared hinges, so they vanish for physically plausible forecasts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .config import TARGET, TEMPERATURE, TIMESTAMP, PhysicsConfig
from .data import year_fraction


def _fit_arms(temp: np.ndarray, load: np.ndarray, t0: float) -> np.ndarray:
    """Least squares a0 + a_heat * (T0 - T)+^2 + a_cool * (T - T0)+^2 with a_heat, a_cool >= 0."""
    basis = np.stack([np.ones_like(temp), np.clip(t0 - temp, 0, None) ** 2, np.clip(temp - t0, 0, None) ** 2], 1)
    active = [0, 1, 2]
    while True:
        coef = np.zeros(3)
        coef[active] = np.linalg.lstsq(basis[:, active], load, rcond=None)[0]
        negative = [i for i in active if i > 0 and coef[i] < 0]
        if not negative:
            return coef
        active.remove(negative[0])


def _annual_growth(years: np.ndarray, load: np.ndarray) -> float:
    """Slope of log(annual mean load) against the year, over (near-)complete years."""
    annual = pd.DataFrame({"year": years, "load": load}).groupby("year")["load"].agg(["mean", "size"])
    annual = annual[annual["size"] >= 300 * 24]
    if len(annual) < 2:
        return 0.0
    return float(np.polyfit(annual.index.to_numpy(float), np.log(annual["mean"].to_numpy()), 1)[0])


@dataclass
class PhysicsCalibration:
    """Piecewise parabolic envelope and ramp limits calibrated on the training years (MW, °C)."""

    t0: float
    lower: tuple[float, float, float]   # (a0 [MW], a_heat [MW/°C²], a_cool [MW/°C²]) of the lower envelope
    upper: tuple[float, float, float]   # same for the upper envelope
    ramp_limit: tuple[float, ...]       # max |dP/dt| for each hour of day (MW/h)
    growth: float                       # annual log-growth of the load level
    ref_year: float                     # year at which the coefficients above apply
    temp_range: tuple[float, float]     # 1st / 99th percentile of the training temperature

    @classmethod
    def fit(cls, frame: pd.DataFrame, cfg: PhysicsConfig | None = None) -> PhysicsCalibration:
        """frame: complete hourly training rows with timestamp, load_mw and temp_c."""
        cfg = cfg or PhysicsConfig()
        ts = frame[TIMESTAMP].reset_index(drop=True)
        temp = frame[TEMPERATURE].to_numpy(np.float64)
        load = frame[TARGET].to_numpy(np.float64)
        year = year_fraction(ts)
        ref_year = float(year.mean())
        growth = _annual_growth(ts.dt.year.to_numpy(), load) if cfg.use_growth else 0.0
        load = load * np.exp(-growth * (year - ref_year))

        # Envelope: parabolas through the low / high load quantiles of each temperature bin.
        bins = pd.DataFrame({"bin": np.floor(temp / cfg.bin_width), "t": temp, "p": load}).groupby("bin")
        q_lo, q_hi = cfg.envelope_quantiles
        table = pd.DataFrame({"t": bins["t"].mean(), "n": bins["p"].size(),
                              "lo": bins["p"].quantile(q_lo), "hi": bins["p"].quantile(q_hi)})
        table = table[table["n"] >= cfg.min_bin_count]
        t_bins = table["t"].to_numpy()
        lower = _fit_arms(t_bins, table["lo"].to_numpy(), cfg.t0) * (1 - cfg.envelope_margin)
        upper = _fit_arms(t_bins, table["hi"].to_numpy(), cfg.t0) * (1 + cfg.envelope_margin)

        # Ramp: high quantile of |P_t - P_(t-1)| for each hour of day of t.
        consecutive = (ts.diff() == pd.Timedelta(hours=1)).to_numpy()[1:]
        ramp = np.abs(np.diff(load))[consecutive]
        ramp_hour = ts.dt.hour.to_numpy()[1:][consecutive]
        overall = np.quantile(ramp, cfg.ramp_quantile)
        ramp_limit = tuple(
            float((np.quantile(ramp[ramp_hour == h], cfg.ramp_quantile) if (ramp_hour == h).any() else overall)
                  * (1 + cfg.ramp_margin))
            for h in range(24)
        )
        return cls(t0=cfg.t0, lower=tuple(map(float, lower)), upper=tuple(map(float, upper)),
                   ramp_limit=ramp_limit, growth=growth, ref_year=ref_year,
                   temp_range=(float(np.quantile(temp, 0.01)), float(np.quantile(temp, 0.99))))

    def growth_factor(self, year: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.growth * (year - self.ref_year))

    def envelope(self, temp: torch.Tensor, year: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Lower / upper load bound (MW) at temperature `temp` (°C) and fractional year `year`."""
        heat = (self.t0 - temp).clamp(min=0) ** 2
        cool = (temp - self.t0).clamp(min=0) ** 2
        scale = self.growth_factor(year)
        lower = (self.lower[0] + self.lower[1] * heat + self.lower[2] * cool) * scale
        upper = (self.upper[0] + self.upper[1] * heat + self.upper[2] * cool) * scale
        return lower, upper

    def ramp_bound(self, hour: torch.Tensor, year: torch.Tensor) -> torch.Tensor:
        """Largest plausible |hourly ramp| (MW/h) into hour `hour`."""
        limit = torch.as_tensor(self.ramp_limit, dtype=year.dtype, device=year.device)
        return limit[hour] * self.growth_factor(year)

    def check(self, pred_mw: torch.Tensor, last_load_mw: torch.Tensor, temp: torch.Tensor,
              hour: torch.Tensor, year: torch.Tensor) -> dict[str, torch.Tensor]:
        """Envelope bounds and violation masks for forecasts pred_mw of shape (N, horizon)."""
        lower, upper = self.envelope(temp, year)
        previous = torch.cat([last_load_mw.reshape(-1, 1), pred_mw[:, :-1]], dim=1)
        ramp = pred_mw - previous
        return {
            "lower": lower,
            "upper": upper,
            "ramp": ramp,
            "envelope_violation": (pred_mw < lower) | (pred_mw > upper),
            "ramp_violation": ramp.abs() > self.ramp_bound(hour, year),
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PhysicsCalibration:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


class PhysicsInformedLoss(nn.Module):
    """L_theta = L_MSE + lambda1 * L_parabolic + lambda2 * L_ramp on standardized load.

    `batch` carries the window inputs x (for the last observed load), the targets y and the
    temperature, hour and fractional year of the target hours (see WindowSet.batches).
    """

    def __init__(self, calibration: PhysicsCalibration, load_mean: float, load_std: float,
                 lambda1: float = 0.1, lambda2: float = 0.1, load_index: int = 0):
        super().__init__()
        self.calibration = calibration
        self.load_mean = load_mean
        self.load_std = load_std
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.load_index = load_index

    def forward(self, y_pred: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        mse = F.mse_loss(y_pred, batch["y"])

        lower, upper = self.calibration.envelope(batch["temp"], batch["year"])
        lower = (lower - self.load_mean) / self.load_std
        upper = (upper - self.load_mean) / self.load_std
        parabolic = (F.relu(lower - y_pred) ** 2 + F.relu(y_pred - upper) ** 2).mean()

        previous = torch.cat([batch["x"][:, -1:, self.load_index], y_pred[:, :-1]], dim=1)
        limit = self.calibration.ramp_bound(batch["hour"], batch["year"]) / self.load_std
        ramp = (F.relu((y_pred - previous).abs() - limit) ** 2).mean()

        total = mse + self.lambda1 * parabolic + self.lambda2 * ramp
        parts = {"total": total, "mse": mse, "parabolic": parabolic, "ramp": ramp}
        return total, {k: v.detach() for k, v in parts.items()}

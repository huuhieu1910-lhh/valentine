"""Hyper-parameters read off Fig. 1 and Table I."""

from __future__ import annotations

from dataclasses import dataclass

TIMESTAMP = "timestamp"
TARGET = "load_mw"
TEMPERATURE = "temp_c"

# Table I, in the channel order of the input tensor X (24 x 13).
FEATURES: tuple[str, ...] = (
    "load_mw",       # Load           Demand (MW), hourly ERCOT system demand
    "temp_c",        # Meteorological Air temp. (°C), three-station hourly mean
    "feels_like_c",  # Meteorological Feels-like (°C), heat-index / wind-chill
    "humidity",      # Meteorological Relative humidity (%)
    "wind_speed",    # Meteorological Surface wind (m/s)
    "precip_mm",     # Meteorological Hourly precipitation accumulation (mm)
    "weather_code",  # Meteorological W-data, encoded weather type
    "hour",          # Calendar 0-23
    "weekday",       # Calendar 1-7 (Monday = 1)
    "month",         # Calendar 1-12
    "day_of_year",   # Calendar 1-366; Table I lists 12 variables, Fig. 1 needs 13
    "is_weekend",    # Calendar binary flag
    "is_holiday",    # Calendar binary flag, U.S. federal calendar
)
RAW_COLUMNS: tuple[str, ...] = FEATURES[:7]


@dataclass(frozen=True)
class SplitConfig:
    """Train / Val / Test = 2018-2022 / 2023 / 2024-2025, as half-open [start, end)."""

    train: tuple[str, str] = ("2018-01-01", "2023-01-01")
    val: tuple[str, str] = ("2023-01-01", "2024-01-01")
    test: tuple[str, str] = ("2024-01-01", "2026-01-01")


@dataclass
class ModelConfig:
    n_features: int = len(FEATURES)  # X (24 x 13)
    seq_len: int = 24
    horizon: int = 1
    # Branch 1: CNN
    conv_kernel: int = 3             # Conv1D k = 3
    conv_channels: int = 64          # Conv1D c = 64
    pool_stride: int = 2             # MaxPool1D s = 2
    # Branch 2: Transformer
    d_model: int = 64                # Linear projection d = 64
    n_heads: int = 4                 # h = 4, so d_k = d_model / h = 16
    d_ff: int = 256                  # Feed-forward d_ff = 256
    n_blocks: int = 2                # x2 encoder blocks
    dropout: float = 0.1
    # Dense layer producing z_CNN / z_T
    dense_dim: int = 64


@dataclass
class PhysicsConfig:
    t0: float = 18.5                 # vertex of the piecewise parabolic envelope (°C)
    lambda1: float = 0.1             # weight of L_parabolic
    lambda2: float = 0.1             # weight of L_ramp
    bin_width: float = 1.0           # °C bins used to calibrate the envelope
    min_bin_count: int = 24
    envelope_quantiles: tuple[float, float] = (0.005, 0.995)
    envelope_margin: float = 0.05    # relative slack around the fitted envelope
    ramp_quantile: float = 0.999
    ramp_margin: float = 0.10
    use_growth: bool = True          # rescale the envelope with the annual load growth trend
    regime_deadband: float = 2.0     # |T - T0| below this is the neutral regime


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8                # early stopping
    lr_patience: int = 3             # epochs without improvement before halving the learning rate
    grad_clip: float = 1.0
    ensemble_steps: int = 300
    ensemble_lr: float = 0.05
    seed: int = 42
    device: str = "auto"

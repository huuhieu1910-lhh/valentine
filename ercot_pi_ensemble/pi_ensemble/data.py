"""ERCOT + ASOS data (Table I): loading, calendar channels, scaling and 24-hour windows."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from pandas.tseries.holiday import USFederalHolidayCalendar

from .config import FEATURES, RAW_COLUMNS, TARGET, TEMPERATURE, TIMESTAMP, ModelConfig, SplitConfig

# Ranges reported in Table I, used to shape the synthetic stand-in data.
TABLE_I_RANGES = {
    "load_mw": (29_360.0, 85_435.0),
    "temp_c": (-14.2, 43.1),
    "humidity": (4.0, 100.0),
    "wind_speed": (0.0, 24.3),
    "precip_mm": (0.0, 48.6),
}
BATCH_KEYS = ("x", "y", "temp", "hour", "year")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the calendar channels of Table I from the timestamp column."""
    df = df.copy()
    ts = df[TIMESTAMP]
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.day_of_week + 1
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.day_of_year
    df["is_weekend"] = (ts.dt.day_of_week >= 5).astype(np.float32)
    days = ts.dt.normalize()
    holidays = USFederalHolidayCalendar().holidays(start=days.min(), end=days.max())
    df["is_holiday"] = days.isin(holidays).astype(np.float32)
    return df


def year_fraction(ts: pd.Series) -> np.ndarray:
    """Calendar year plus the elapsed fraction of that year, e.g. 2021.5 in early July 2021."""
    days_in_year = np.where(ts.dt.is_leap_year, 366.0, 365.0)
    return (ts.dt.year + (ts.dt.day_of_year - 1 + ts.dt.hour / 24.0) / days_in_year).to_numpy(np.float64)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read an hourly ERCOT + ASOS file holding a timestamp column and the Table I variables.

    Calendar channels are derived from the timestamp and a textual weather_code (raw ASOS
    wxcodes) is label-encoded. Duplicate hours (DST) are dropped and gaps of up to six
    hours are filled, so the result is a complete hourly series.
    """
    df = pd.read_csv(path)
    missing = [c for c in (TIMESTAMP, *RAW_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    ts = pd.to_datetime(df[TIMESTAMP])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    df = df.assign(**{TIMESTAMP: ts}).sort_values(TIMESTAMP).drop_duplicates(TIMESTAMP)
    if not pd.api.types.is_numeric_dtype(df["weather_code"]):
        df["weather_code"] = pd.factorize(df["weather_code"].fillna("none"))[0]

    df = df.set_index(TIMESTAMP)[list(RAW_COLUMNS)].astype(float)
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="h"))
    continuous = [c for c in RAW_COLUMNS if c != "weather_code"]
    df[continuous] = df[continuous].interpolate(limit=6, limit_area="inside")
    df["weather_code"] = df["weather_code"].ffill(limit=6)
    if df.isna().any().any():
        raise ValueError(f"{path}: gaps longer than 6 hours remain after filling")
    return add_calendar_features(df.rename_axis(TIMESTAMP).reset_index())


# --------------------------------------------------------------------------- synthetic data


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = phi * acc + eps[i]
        out[i] = acc
    return out


def _event(ts: pd.Series, start: str, end: str) -> np.ndarray:
    """Smoothed 0/1 indicator of [start, end)."""
    inside = ((ts >= start) & (ts < end)).to_numpy(float)
    kernel = np.hanning(49)
    return np.convolve(inside, kernel / kernel.sum(), mode="same")


def _rescale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + (x - x.min()) * (hi - lo) / (x.max() - x.min())


def _precipitation(doy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Two-state Markov rain process (wetter in spring) with exponential hourly amounts."""
    p_start = 0.015 * (1.0 + 0.6 * np.sin(2 * np.pi * (doy - 60) / 365.25))
    u = rng.random(len(doy))
    raining = np.zeros(len(doy), dtype=bool)
    for i in range(1, len(doy)):
        raining[i] = u[i] < (0.7 if raining[i - 1] else p_start[i])
    return np.clip(np.where(raining, rng.exponential(1.8, len(doy)), 0.0), *TABLE_I_RANGES["precip_mm"])


def apparent_temperature(temp_c: np.ndarray, rh: np.ndarray, wind_ms: np.ndarray) -> np.ndarray:
    """NWS heat index when hot and humid, wind chill when cold and windy, air temperature otherwise."""
    t_f = temp_c * 9 / 5 + 32
    hi_f = (-42.379 + 2.04901523 * t_f + 10.14333127 * rh - 0.22475541 * t_f * rh
            - 6.83783e-3 * t_f**2 - 5.481717e-2 * rh**2 + 1.22874e-3 * t_f**2 * rh
            + 8.5282e-4 * t_f * rh**2 - 1.99e-6 * t_f**2 * rh**2)
    v = wind_ms * 3.6
    wind_chill = 13.12 + 0.6215 * temp_c - 11.37 * v**0.16 + 0.3965 * temp_c * v**0.16
    hot = (temp_c >= 26.7) & (rh >= 40)
    cold = (temp_c <= 10.0) & (v > 4.8)
    out = temp_c.copy()
    out[hot] = np.maximum((hi_f[hot] - 32) * 5 / 9, temp_c[hot])
    out[cold] = np.minimum(wind_chill[cold], temp_c[cold])
    return out


def make_synthetic_ercot_asos(start: str = "2018-01-01 00:00", end: str = "2025-12-31 23:00",
                              seed: int = 0, t0: float = 18.5) -> pd.DataFrame:
    """Synthetic stand-in with the columns and ranges of Table I (70,128 hourly rows for 2018-2025).

    Only meant for smoke tests and demos: the load follows a piecewise parabolic response to
    temperature around t0, daily/weekly/holiday profiles and ~3.5 %/yr growth.
    """
    rng = np.random.default_rng(seed)
    ts = pd.Series(pd.date_range(start, end, freq="h"), name=TIMESTAMP)
    n = len(ts)
    df = add_calendar_features(ts.to_frame())
    hour = df["hour"].to_numpy(float)
    doy = df["day_of_year"].to_numpy(float)
    years = year_fraction(ts) - ts.dt.year.iloc[0]
    diurnal = np.sin(2 * np.pi * (hour - 9) / 24)

    temp = 20.0 + 9.5 * np.sin(2 * np.pi * (doy - 115) / 365.25) + 5.0 * diurnal + _ar1(n, 0.995, 0.35, rng)
    temp -= 18.0 * _event(ts, "2021-02-13", "2021-02-20")   # Winter Storm Uri
    temp += 5.0 * _event(ts, "2023-07-25", "2023-08-25")    # summer 2023 heat wave
    temp = _rescale(temp, *TABLE_I_RANGES["temp_c"])

    precip = _precipitation(doy, rng)
    humidity = np.clip(68 - 15 * diurnal - 0.6 * (temp - 20) + _ar1(n, 0.97, 2.5, rng) + 20 * (precip > 0),
                       *TABLE_I_RANGES["humidity"])
    wind = np.clip(4.0 + 1.2 * np.sin(2 * np.pi * (hour - 14) / 24) + _ar1(n, 0.95, 0.8, rng) + 3 * (precip > 0),
                   *TABLE_I_RANGES["wind_speed"])
    code = np.zeros(n)                                  # 0 clear
    code[humidity > 80] = 1                             # 1 overcast / haze
    code[(humidity >= 97) & (precip == 0)] = 4          # 4 fog / mist
    code[precip > 0] = 2                                # 2 rain
    code[(precip > 6) & (temp > 15)] = 3                # 3 thunderstorm
    code[(precip > 0) & (temp < 1)] = 5                 # 5 snow / ice

    t_eff = pd.Series(temp).ewm(alpha=0.3, adjust=False).mean().to_numpy()   # building thermal inertia
    heat = np.clip(t0 - t_eff, 0, None) ** 2
    cool = np.clip(t_eff - t0, 0, None) ** 2
    profile = 1 + 0.10 * np.sin(2 * np.pi * (hour - 10) / 24) + 0.03 * np.sin(4 * np.pi * (hour - 4.5) / 24)
    calendar = 1 - 0.06 * df["is_weekend"].to_numpy() - 0.07 * df["is_holiday"].to_numpy()
    load = 1.035 ** years * (38_000 * profile * calendar
                             + 40 * heat * (1 + 0.2 * np.sin(2 * np.pi * (hour - 1) / 24))
                             + 85 * cool * (1 + 0.25 * np.sin(2 * np.pi * (hour - 10) / 24)))
    load *= 1 + _ar1(n, 0.9, 0.006, rng)

    df["load_mw"] = _rescale(load, *TABLE_I_RANGES["load_mw"])
    df["temp_c"] = temp
    df["feels_like_c"] = apparent_temperature(temp, humidity, wind)
    df["humidity"] = humidity
    df["wind_speed"] = wind
    df["precip_mm"] = precip
    df["weather_code"] = code
    return df[[TIMESTAMP, *FEATURES]]


# --------------------------------------------------------------------------- windows


@dataclass
class WindowSet:
    """Sliding windows: X (N, 24, 13) -> load at the next `horizon` hours."""

    x: torch.Tensor       # (N, seq_len, n_features) standardized inputs
    y: torch.Tensor       # (N, horizon) standardized load at the target hours
    temp: torch.Tensor    # (N, horizon) air temperature at the target hours (°C)
    hour: torch.Tensor    # (N, horizon) hour of day of the target hours
    year: torch.Tensor    # (N, horizon) fractional year of the target hours
    times: np.ndarray     # (N, horizon) target timestamps

    def __len__(self) -> int:
        return self.x.shape[0]

    def subset(self, index) -> WindowSet:
        index = np.asarray(index)
        picked = {f.name: getattr(self, f.name)[torch.as_tensor(index)] for f in fields(self) if f.name != "times"}
        return WindowSet(**picked, times=self.times[index])

    def to(self, device: torch.device | str) -> WindowSet:
        moved = {k: getattr(self, k).to(device) for k in BATCH_KEYS}
        return WindowSet(**moved, times=self.times)

    def as_batch(self) -> dict[str, torch.Tensor]:
        return {k: getattr(self, k) for k in BATCH_KEYS}

    def batches(self, batch_size: int, shuffle: bool = False,
                generator: torch.Generator | None = None) -> Iterator[dict[str, torch.Tensor]]:
        order = torch.randperm(len(self), generator=generator) if shuffle else torch.arange(len(self))
        order = order.to(self.x.device)
        for start in range(0, len(self), batch_size):
            idx = order[start:start + batch_size]
            yield {k: getattr(self, k)[idx] for k in BATCH_KEYS}


@dataclass
class DataBundle:
    train: WindowSet
    val: WindowSet
    test: WindowSet
    feature_mean: np.ndarray  # (n_features,) training-set statistics
    feature_std: np.ndarray
    features: tuple[str, ...] = FEATURES

    @property
    def load_index(self) -> int:
        return self.features.index(TARGET)

    @property
    def load_mean(self) -> float:
        return float(self.feature_mean[self.load_index])

    @property
    def load_std(self) -> float:
        return float(self.feature_std[self.load_index])

    def to_mw(self, y: torch.Tensor | np.ndarray) -> np.ndarray:
        """Standardized load -> MW."""
        y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        return y.astype(np.float64) * self.load_std + self.load_mean


def in_period(ts: pd.Series, period: tuple[str, str]) -> np.ndarray:
    start, end = (pd.Timestamp(p) for p in period)
    return ((ts >= start) & (ts < end)).to_numpy()


def prepare_data(df: pd.DataFrame, cfg: ModelConfig | None = None,
                 split: SplitConfig | None = None) -> DataBundle:
    """Standardize with training statistics and cut (seq_len x n_features) windows.

    A window's inputs are the seq_len hours before its first target hour; it belongs to
    the split that contains all of its target hours.
    """
    cfg = cfg or ModelConfig()
    split = split or SplitConfig()
    ts = df[TIMESTAMP].reset_index(drop=True)
    if not (ts.diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("expected a gap-free hourly series (see load_csv)")

    values = df[list(FEATURES)].to_numpy(np.float64)
    train_rows = in_period(ts, split.train)
    mean = values[train_rows].mean(axis=0)
    std = values[train_rows].std(axis=0)
    std[std < 1e-8] = 1.0
    scaled = ((values - mean) / std).astype(np.float32)

    seq_len, horizon = cfg.seq_len, cfg.horizon
    first = np.arange(seq_len, len(df) - horizon + 1)                       # first target hour per window
    x = sliding_window_view(scaled, seq_len, axis=0)[first - seq_len]       # (N, n_features, seq_len)
    x = x.transpose(0, 2, 1)

    def at_targets(column: np.ndarray) -> np.ndarray:
        return sliding_window_view(column, horizon)[first]                  # (N, horizon)

    y = at_targets(scaled[:, FEATURES.index(TARGET)])
    temp = at_targets(df[TEMPERATURE].to_numpy(np.float32))
    hour = at_targets(ts.dt.hour.to_numpy(np.int64))
    year = at_targets(year_fraction(ts).astype(np.float32))
    times = at_targets(ts.to_numpy())

    def window_set(period: tuple[str, str]) -> WindowSet:
        start, end = (pd.Timestamp(p).to_datetime64() for p in period)
        keep = (times[:, 0] >= start) & (times[:, -1] < end)
        as_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a[keep]))  # noqa: E731
        return WindowSet(x=as_tensor(x), y=as_tensor(y), temp=as_tensor(temp), hour=as_tensor(hour),
                         year=as_tensor(year), times=times[keep])

    return DataBundle(train=window_set(split.train), val=window_set(split.val), test=window_set(split.test),
                      feature_mean=mean, feature_std=std)

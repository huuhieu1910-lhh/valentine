"""Merge EIA-930 BALANCE files (ERCO demand) and IEM ASOS observations into the Table I CSV.

    python scripts/build_dataset.py --eia "ercot data.zip" --asos asos.csv --out ercot_asos_2018_2025.csv

--eia accepts a .zip of EIA930_BALANCE_*.csv files, a folder holding them, or a single csv.
The ASOS file is the IEM download (station, valid, tmpc or tmpf, relh, feel, sknt, p01i, wxcodes)
with timestamps in America/Chicago.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TZ = "America/Chicago"


def _read_eia_frames(path: Path) -> list[pd.DataFrame]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv") and "balance" in n.lower()]
            return [pd.read_csv(zf.open(n), low_memory=False) for n in sorted(names)]
    if path.is_dir():
        return [pd.read_csv(p, low_memory=False) for p in sorted(path.rglob("*BALANCE*.csv"))]
    return [pd.read_csv(path, low_memory=False)]


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def read_eia(path: Path) -> pd.Series:
    """Hourly ERCO demand (MW) indexed by local hour-beginning time (America/Chicago, naive)."""
    parts = []
    for df in _read_eia_frames(path):
        df.columns = [c.strip() for c in df.columns]
        df = df[df["Balancing Authority"].astype(str).str.strip() == "ERCO"]
        demand = _numeric(df["Demand (MW)"])
        if "Demand (MW) (Adjusted)" in df.columns:  # adjusted (imputed) values fill reporting gaps
            demand = _numeric(df["Demand (MW) (Adjusted)"]).fillna(demand)
        end_utc = pd.to_datetime(df["UTC Time at End of Hour"], format="mixed", utc=True)
        start_local = (end_utc - pd.Timedelta(hours=1)).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        parts.append(pd.Series(demand.to_numpy(), index=start_local.to_numpy(), name="load_mw"))
    if not parts:
        raise ValueError(f"no EIA930 BALANCE csv found in {path}")
    load = pd.concat(parts).dropna()
    return load[~load.index.duplicated()].sort_index()


def _mode(s: pd.Series) -> str:
    s = s.dropna()
    s = s[s.astype(str).str.strip() != ""]
    return s.mode().iloc[0] if len(s) else ""


def read_asos(path: Path) -> pd.DataFrame:
    """Three-station hourly mean in Table I units, indexed by local hour."""
    df = pd.read_csv(path, na_values=["M", "", "null"], comment="#", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["hour"] = pd.to_datetime(df["valid"]).dt.round("h")   # routine METARs come at hh:53
    temp = _numeric(df["tmpc"]) if "tmpc" in df else (_numeric(df["tmpf"]) - 32) * 5 / 9
    obs = pd.DataFrame({
        "station": df["station"], "hour": df["hour"],
        "temp_c": temp,
        "feels_like_c": (_numeric(df["feel"]) - 32) * 5 / 9,
        "humidity": _numeric(df["relh"]),
        "wind_speed": _numeric(df["sknt"]) * 0.514444,
        "precip_mm": _numeric(df["p01i"]) * 25.4,
        "wxcodes": df["wxcodes"].fillna("").astype(str).str.strip() if "wxcodes" in df else "",
    })
    per_station = obs.groupby(["station", "hour"]).agg(
        temp_c=("temp_c", "mean"), feels_like_c=("feels_like_c", "mean"), humidity=("humidity", "mean"),
        wind_speed=("wind_speed", "mean"), precip_mm=("precip_mm", "max"), wxcodes=("wxcodes", _mode))
    hourly = per_station.groupby("hour").agg(
        temp_c=("temp_c", "mean"), feels_like_c=("feels_like_c", "mean"), humidity=("humidity", "mean"),
        wind_speed=("wind_speed", "mean"), precip_mm=("precip_mm", "mean"), weather_code=("wxcodes", _mode))
    hourly["feels_like_c"] = hourly["feels_like_c"].fillna(hourly["temp_c"])
    hourly["precip_mm"] = hourly["precip_mm"].fillna(0.0)
    return hourly


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eia", type=Path, required=True)
    p.add_argument("--asos", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("ercot_asos_2018_2025.csv"))
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2026-01-01")
    args = p.parse_args()

    load = read_eia(args.eia)
    weather = read_asos(args.asos)
    index = pd.date_range(args.start, args.end, freq="h", inclusive="left")
    weather = weather.reindex(index)
    weather.iloc[:3] = weather.iloc[:3].bfill()   # first METAR of the range lands on 01:00
    df = weather.join(load.reindex(index))
    df.index.name = "timestamp"
    df = df[["load_mw", "temp_c", "feels_like_c", "humidity", "wind_speed", "precip_mm", "weather_code"]]
    df.to_csv(args.out)

    missing = df.drop(columns="weather_code").isna().mean() * 100
    print(f"{len(df):,} hours -> {args.out}")
    print("missing %:", missing.round(2).to_dict())
    print(df.describe().T[["min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()

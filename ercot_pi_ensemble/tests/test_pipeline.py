import json

import numpy as np
import pandas as pd
import pytest

from pi_ensemble.config import FEATURES, RAW_COLUMNS, TIMESTAMP, ModelConfig
from pi_ensemble.data import TABLE_I_RANGES, load_csv, make_synthetic_ercot_asos, prepare_data


@pytest.fixture(scope="module")
def frame():
    return make_synthetic_ercot_asos(seed=0)


def test_synthetic_data_matches_table_one(frame):
    assert len(frame) == 70_128
    assert list(frame.columns) == [TIMESTAMP, *FEATURES]
    for col, (lo, hi) in TABLE_I_RANGES.items():
        assert lo - 1e-6 <= frame[col].min() and frame[col].max() <= hi + 1e-6
    assert frame["load_mw"].min() == pytest.approx(29_360) and frame["load_mw"].max() == pytest.approx(85_435)
    assert sorted(frame["weekday"].unique()) == list(range(1, 8))
    holidays = frame.set_index(TIMESTAMP)["is_holiday"]
    assert holidays["2019-07-04 12:00"] == 1 and holidays["2023-11-23 08:00"] == 1
    assert holidays["2024-12-25 00:00"] == 1 and holidays["2024-12-26 00:00"] == 0


@pytest.mark.parametrize("horizon", [1, 24])
def test_windows_and_splits(frame, horizon):
    bundle = prepare_data(frame, ModelConfig(horizon=horizon))
    lost = horizon - 1  # windows whose targets would spill into the next split
    assert len(bundle.train) == 43_824 - 24 - lost  # 2018-2022, minus the first input window
    assert len(bundle.val) == 8_760 - lost          # 2023
    assert len(bundle.test) == 17_544 - lost        # 2024 (leap year) + 2025
    assert bundle.train.x.shape[1:] == (24, 13) and bundle.train.y.shape[1:] == (horizon,)

    i = 1_000
    row = int(np.flatnonzero(frame[TIMESTAMP].to_numpy() == bundle.test.times[i, 0])[0])
    expected = (frame[list(FEATURES)].to_numpy()[row - 24:row] - bundle.feature_mean) / bundle.feature_std
    np.testing.assert_allclose(bundle.test.x[i].numpy(), expected, atol=1e-4)
    np.testing.assert_allclose(bundle.to_mw(bundle.test.y[i]), frame["load_mw"].to_numpy()[row:row + horizon],
                               rtol=1e-5)
    assert bundle.test.hour[i, 0] == frame[TIMESTAMP][row].hour


def test_load_csv_repairs_hourly_series(frame, tmp_path):
    raw = frame[[TIMESTAMP, *RAW_COLUMNS]].iloc[:500].copy()
    raw["weather_code"] = raw["weather_code"].map({0: "", 1: "BR", 2: "RA", 3: "TS", 4: "FG", 5: "SN"})
    raw = pd.concat([raw.drop(index=[100, 101, 102]), raw.iloc[[50]]])  # 3-hour gap + duplicated hour
    path = tmp_path / "ercot_asos.csv"
    raw.to_csv(path, index=False)

    df = load_csv(path)
    assert len(df) == 500 and list(df.columns[:8]) == [TIMESTAMP, *RAW_COLUMNS]
    assert set(FEATURES) <= set(df.columns)
    assert (df[TIMESTAMP].diff().iloc[1:] == pd.Timedelta(hours=1)).all()
    assert not df[list(FEATURES)].isna().any().any()


def test_end_to_end_smoke(tmp_path):
    pytest.importorskip("shap")
    from pi_ensemble.run import load_checkpoint, main

    summary = main(["--synthetic", "--epochs", "1", "--max-train-windows", "512", "--explain-hours", "12",
                    "--background", "16", "--out", str(tmp_path)])
    for name in ("model.pt", "operator_report.csv", "shap_importance_mw.csv", "shap_heatmap.png", "summary.json"):
        assert (tmp_path / name).exists()
    assert sum(summary["ensemble_weights"].values()) == pytest.approx(1.0)
    assert json.loads((tmp_path / "summary.json").read_text())["test_metrics"]["ensemble"]["MAPE_%"] > 0

    report = pd.read_csv(tmp_path / "operator_report.csv")
    assert len(report) == 12
    assert set(report["regime"]) <= {"heating", "neutral", "cooling"}
    assert report["top_drivers_mw"].str.count(";").eq(2).all()

    model, calibration, scaling = load_checkpoint(tmp_path / "model.pt")
    assert calibration.t0 == 18.5 and scaling["features"] == list(FEATURES)

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pi_ensemble.config import TIMESTAMP, PhysicsConfig, SplitConfig
from pi_ensemble.data import in_period, make_synthetic_ercot_asos, year_fraction
from pi_ensemble.physics import PhysicsCalibration, PhysicsInformedLoss


@pytest.fixture(scope="module")
def train_frame():
    df = make_synthetic_ercot_asos(seed=1)
    return df[in_period(df[TIMESTAMP], SplitConfig().train)].reset_index(drop=True)


def flat_calibration(**overrides):
    params = dict(t0=18.5, lower=(30_000.0, 0.0, 0.0), upper=(80_000.0, 0.0, 0.0), ramp_limit=(1_000.0,) * 24,
                  growth=0.0, ref_year=2020.0, temp_range=(0.0, 40.0))
    return PhysicsCalibration(**{**params, **overrides})


def batch_for(y, last_load=0.0, temp=20.0):
    n, h = y.shape
    x = torch.zeros(n, 24, 13)
    x[:, -1, 0] = last_load
    return {"x": x, "y": torch.zeros(n, h), "temp": torch.full((n, h), temp),
            "hour": torch.zeros(n, h, dtype=torch.long), "year": torch.full((n, h), 2020.0)}


def test_envelope_calibration(train_frame):
    calib = PhysicsCalibration.fit(train_frame, PhysicsConfig())
    assert calib.t0 == 18.5
    assert calib.lower[1] > 0 and calib.lower[2] > 0       # heating and cooling arms open upwards
    assert calib.upper[1] > 0 and calib.upper[2] > 0
    assert 0.0 < calib.growth < 0.08
    assert len(calib.ramp_limit) == 24 and min(calib.ramp_limit) > 0

    temp = torch.linspace(-15, 45, 121)
    lower, upper = calib.envelope(temp, torch.full_like(temp, calib.ref_year))
    assert (lower < upper).all()
    assert abs(temp[upper.argmin()].item() - 18.5) <= 0.5  # vertex at T0

    t = torch.tensor(train_frame["temp_c"].to_numpy(), dtype=torch.float32)
    p = torch.tensor(train_frame["load_mw"].to_numpy(), dtype=torch.float32)
    yr = torch.tensor(year_fraction(train_frame[TIMESTAMP]), dtype=torch.float32)
    lower, upper = calib.envelope(t, yr)
    inside = ((p >= lower) & (p <= upper)).float().mean().item()
    assert inside > 0.99                                    # training load lies inside the envelope


def test_loss_terms():
    loss_fn = PhysicsInformedLoss(flat_calibration(), load_mean=50_000.0, load_std=10_000.0,
                                  lambda1=0.5, lambda2=2.0)
    # inside the envelope, ramp 500 MW below the 1,000 MW limit: only the MSE term remains
    y = torch.full((4, 1), 0.05, requires_grad=True)
    total, parts = loss_fn(y, batch_for(y))
    assert parts["parabolic"] == 0 and parts["ramp"] == 0
    assert torch.isclose(total, F.mse_loss(y, torch.zeros_like(y)))

    # 90,000 MW: 1 std above the 80,000 MW upper envelope; ramp 40,000 MW vs a 1,000 MW limit
    y = torch.full((4, 1), 4.0, requires_grad=True)
    total, parts = loss_fn(y, batch_for(y))
    assert torch.isclose(parts["parabolic"], torch.tensor(1.0))
    assert torch.isclose(parts["ramp"], torch.tensor((4.0 - 0.1) ** 2))
    assert torch.isclose(total, parts["mse"] + 0.5 * parts["parabolic"] + 2.0 * parts["ramp"])
    total.backward()
    assert y.grad is not None and (y.grad > 0).all()

    # 20,000 MW: 1 std below the 30,000 MW lower envelope
    y = torch.full((2, 1), -3.0)
    _, parts = loss_fn(y, batch_for(y, last_load=-3.0))
    assert torch.isclose(parts["parabolic"], torch.tensor(1.0)) and parts["ramp"] == 0


def test_ramp_chains_through_multi_step_forecast():
    loss_fn = PhysicsInformedLoss(flat_calibration(), load_mean=50_000.0, load_std=10_000.0)
    y = torch.tensor([[0.05, 0.10, 0.60]])  # ramps of 500, 500 and 5,000 MW
    _, parts = loss_fn(y, batch_for(y))
    assert torch.isclose(parts["ramp"], torch.tensor((0.5 - 0.1) ** 2 / 3))


def test_check_and_serialization():
    calib = flat_calibration()
    pred = torch.tensor([[50_000.0], [85_000.0], [52_500.0]])
    checks = calib.check(pred, torch.tensor([50_000.0, 85_000.0, 50_000.0]), torch.full((3, 1), 20.0),
                         torch.zeros(3, 1, dtype=torch.long), torch.full((3, 1), 2020.0))
    assert checks["envelope_violation"].squeeze(1).tolist() == [False, True, False]
    assert checks["ramp_violation"].squeeze(1).tolist() == [False, False, True]
    assert PhysicsCalibration.from_dict(json.loads(json.dumps(calib.to_dict()))) == calib


def test_growth_scales_envelope():
    calib = flat_calibration(growth=float(np.log(1.1)))
    lower, upper = calib.envelope(torch.tensor([20.0]), torch.tensor([2021.0]))
    assert torch.allclose(upper, torch.tensor([88_000.0])) and torch.allclose(lower, torch.tensor([33_000.0]))

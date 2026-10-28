"""End-to-end pipeline of Fig. 1.

    python -m pi_ensemble.run --synthetic             # demo on synthetic ERCOT-like data
    python -m pi_ensemble.run --data ercot_asos.csv   # ERCOT + ASOS 2018-2025 (Table I columns)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import FEATURES, TIMESTAMP, ModelConfig, PhysicsConfig, SplitConfig, TrainConfig
from .data import DataBundle, in_period, load_csv, make_synthetic_ercot_asos, prepare_data
from .explain import explain_ensemble, plot_heatmap
from .models import PhysicsInformedEnsemble
from .operator_report import operator_report
from .physics import PhysicsCalibration, PhysicsInformedLoss
from .train import evaluate, fit_ensemble_weights, resolve_device, set_seed, train_branch


def save_checkpoint(path: Path, model: PhysicsInformedEnsemble, bundle: DataBundle,
                    calibration: PhysicsCalibration) -> None:
    torch.save({
        "model": model.state_dict(),
        "model_cfg": asdict(model.cfg),
        "calibration": calibration.to_dict(),
        "features": list(bundle.features),
        "feature_mean": bundle.feature_mean.tolist(),
        "feature_std": bundle.feature_std.tolist(),
    }, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> tuple[PhysicsInformedEnsemble, PhysicsCalibration, dict]:
    """Return the trained model, its physics calibration and the input scaling statistics."""
    ckpt = torch.load(path, map_location=map_location)
    model = PhysicsInformedEnsemble(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    scaling = {k: ckpt[k] for k in ("features", "feature_mean", "feature_std")}
    return model.eval(), PhysicsCalibration.from_dict(ckpt["calibration"]), scaling


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Physics-informed interpretable CNN + Transformer ensemble (Fig. 1)")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="hourly CSV: timestamp + Table I columns")
    source.add_argument("--synthetic", action="store_true", help="use synthetic ERCOT-like data (demo)")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--horizon", type=int, default=ModelConfig.horizon)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--patience", type=int, default=TrainConfig.patience)
    p.add_argument("--lambda1", type=float, default=PhysicsConfig.lambda1, help="weight of L_parabolic")
    p.add_argument("--lambda2", type=float, default=PhysicsConfig.lambda2, help="weight of L_ramp")
    p.add_argument("--max-train-windows", type=int, default=None, help="random subset of training windows")
    p.add_argument("--explain-hours", type=int, default=168, help="last test windows explained with SHAP")
    p.add_argument("--background", type=int, default=100, help="SHAP background windows from training")
    p.add_argument("--device", default=TrainConfig.device)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    model_cfg = ModelConfig(horizon=args.horizon)
    physics_cfg = PhysicsConfig(lambda1=args.lambda1, lambda2=args.lambda2)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            patience=args.patience, seed=args.seed, device=args.device)
    split = SplitConfig()
    args.out.mkdir(parents=True, exist_ok=True)

    # Input tensor X (24 x 13): ERCOT + ASOS 2018-2025
    df = make_synthetic_ercot_asos(seed=args.seed) if args.synthetic else load_csv(args.data)
    bundle = prepare_data(df, model_cfg, split)
    print(f"{len(df):,} hourly rows | windows train {len(bundle.train):,} / val {len(bundle.val):,} / "
          f"test {len(bundle.test):,} | X {tuple(bundle.train.x.shape[1:])} on {device}")

    # Physics-informed loss calibrated on the ERCOT piecewise parabolic envelope (training years)
    calibration = PhysicsCalibration.fit(df[in_period(df[TIMESTAMP], split.train)], physics_cfg)
    loss_fn = PhysicsInformedLoss(calibration, bundle.load_mean, bundle.load_std,
                                  physics_cfg.lambda1, physics_cfg.lambda2, bundle.load_index)

    rng = np.random.default_rng(args.seed)
    train = bundle.train
    if args.max_train_windows and args.max_train_windows < len(train):
        train = train.subset(np.sort(rng.choice(len(train), args.max_train_windows, replace=False)))
    train, val, test = (s.to(device) for s in (train, bundle.val, bundle.test))

    # Branch 1 (CNN) and Branch 2 (Transformer), each trained on L_theta
    model = PhysicsInformedEnsemble(model_cfg).to(device)
    history = {
        "cnn": train_branch(model.cnn, loss_fn, train, val, train_cfg, name="CNN"),
        "transformer": train_branch(model.transformer, loss_fn, train, val, train_cfg, name="Transformer"),
    }

    # Weighted ensemble, weights fitted through the L_theta feedback on validation
    weights = dict(zip(model.members, fit_ensemble_weights(model, loss_fn, val, train_cfg)))
    metrics = evaluate(model, test, bundle, calibration)

    # SHAP DeepExplainer + operator forecast / attribution / regime flag on the latest test hours
    recent = test.subset(np.arange(max(0, len(test) - args.explain_hours), len(test)))
    background = train.x[torch.as_tensor(rng.choice(len(train), min(args.background, len(train)), replace=False))]
    explanation = explain_ensemble(model, background, recent.x)
    report = operator_report(model, recent, bundle, calibration, physics_cfg, explanation)

    importance = pd.DataFrame(
        {name: explanation.importance(name) * bundle.load_std for name in explanation.phi},
        index=pd.Index(FEATURES, name="feature"),
    ).sort_values("ensemble", ascending=False)
    shap_gap_mw = {k: v * bundle.load_std for k, v in explanation.gap.items()}

    save_checkpoint(args.out / "model.pt", model, bundle, calibration)
    report.to_csv(args.out / "operator_report.csv", index=False)
    importance.to_csv(args.out / "shap_importance_mw.csv")
    plot_heatmap(explanation, FEATURES, args.out / "shap_heatmap.png", scale=bundle.load_std)
    summary = {"ensemble_weights": weights, "test_metrics": metrics, "shap_completeness_gap_mw": shap_gap_mw,
               "physics_calibration": calibration.to_dict(), "history": history}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nEnsemble weights w':", {k: round(v, 3) for k, v in weights.items()})
    print("\nTest 2024-2025:")
    print(pd.DataFrame(metrics).T.round(2).to_string())
    print("\nSHAP importance (mean |phi|, MW):")
    print(importance.round(1).to_string())
    print("\nSHAP completeness gap (MW):", {k: round(v, 2) for k, v in shap_gap_mw.items()})
    print("\nOperator output (last 5 hours):")
    print(report.tail(5).to_string(index=False))
    print(f"\nSaved to {args.out.resolve()}")
    return summary


if __name__ == "__main__":
    main()

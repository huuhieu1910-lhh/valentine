"""SHAP DeepExplainer attributions of Fig. 1: phi_i^ens = w'_CNN * phi_i^CNN + w'_T * phi_i^T.

Because the ensemble is linear in its members and both members are explained against the
same background, the ensemble attribution is the weighted sum of the member attributions.

DeepExplainer is exact for the CNN branch (Conv / BN / ReLU / MaxPool / GAP / Linear all have
DeepLIFT rules). In the Transformer branch, LayerNorm, softmax and the attention products have
no DeepLIFT rule in SHAP and fall back to plain gradients, so those attributions are approximate;
the completeness gap max|f(x) - E[f] - sum(phi)| is reported for every model instead of hidden.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .models import PhysicsInformedEnsemble


def deep_shap(model: nn.Module, background: torch.Tensor, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """SHAP values (N, seq_len, n_features, horizon) and expected value (horizon,) of one branch.

    The branch is deep-copied because DeepExplainer attaches hooks and buffers to its layers.
    """
    import shap

    model = copy.deepcopy(model).eval()
    explainer = shap.DeepExplainer(model, background)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unrecognized nn.Module")
        values = explainer.shap_values(x, check_additivity=False)
    if isinstance(values, list):  # older SHAP releases return one array per output
        values = np.stack(values, axis=-1)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == x.ndim:
        values = values[..., None]
    return values, np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)


@dataclass
class EnsembleExplanation:
    """Attributions in standardized-load units, keyed by "cnn", "transformer" and "ensemble"."""

    phi: dict[str, np.ndarray]       # (N, seq_len, n_features, horizon)
    expected: dict[str, np.ndarray]  # (horizon,)
    gap: dict[str, float]            # max |f(x) - E[f] - sum(phi)|
    weights: np.ndarray              # w' = (w_CNN, w_T)

    def contributions(self, name: str = "ensemble", step: int = 0) -> np.ndarray:
        """Net contribution of each input channel over the whole window, (N, n_features)."""
        return self.phi[name][..., step].sum(axis=1)

    def importance(self, name: str = "ensemble", step: int = 0) -> np.ndarray:
        """Global importance mean_n |contribution|, (n_features,)."""
        return np.abs(self.contributions(name, step)).mean(axis=0)


def explain_ensemble(model: PhysicsInformedEnsemble, background: torch.Tensor,
                     x: torch.Tensor) -> EnsembleExplanation:
    weights = model.ensemble.weights.detach().cpu().double().numpy()
    phi, expected = {}, {}
    for name in model.members:
        phi[name], expected[name] = deep_shap(getattr(model, name), background, x)
    phi["ensemble"] = sum(w * phi[name] for w, name in zip(weights, model.members))
    expected["ensemble"] = sum(w * expected[name] for w, name in zip(weights, model.members))

    model.eval()
    with torch.no_grad():
        outputs = {k: v.cpu().double().numpy() for k, v in model(x).items()}
    gap = {k: float(np.abs(outputs[k] - expected[k] - phi[k].sum(axis=(1, 2))).max()) for k in phi}
    return EnsembleExplanation(phi=phi, expected=expected, gap=gap, weights=weights)


def plot_heatmap(explanation: EnsembleExplanation, features: tuple[str, ...], path: str | Path,
                 scale: float = 1.0, name: str = "ensemble", step: int = 0) -> None:
    """Time x feature heatmap of mean |phi|, the SHAP panel of Fig. 1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.abs(explanation.phi[name][..., step]).mean(axis=0) * scale   # (seq_len, n_features)
    seq_len = grid.shape[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(grid.T, aspect="auto", cmap="Reds")
    ax.set_yticks(range(len(features)), features)
    ticks = list(range(0, seq_len, 3)) + [seq_len - 1]
    ax.set_xticks(ticks, [f"t-{seq_len - 1 - i}" if i < seq_len - 1 else "t" for i in ticks])
    ax.set_xlabel("input hour")
    ax.set_title(f"SHAP DeepExplainer, {name}: mean |phi|")
    fig.colorbar(image, ax=ax, label="MW" if scale != 1.0 else "standardized load")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

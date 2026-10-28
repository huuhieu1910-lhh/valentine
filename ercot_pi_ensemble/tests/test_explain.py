import numpy as np
import pytest
import torch

pytest.importorskip("shap")

from pi_ensemble.config import ModelConfig  # noqa: E402
from pi_ensemble.explain import deep_shap, explain_ensemble  # noqa: E402
from pi_ensemble.models import PhysicsInformedEnsemble  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = PhysicsInformedEnsemble(ModelConfig()).train()
    with torch.no_grad():
        for _ in range(3):  # non-trivial BatchNorm running statistics
            m(torch.randn(64, 24, 13))
        m.ensemble.logits.copy_(torch.tensor([0.4, -0.4]))
    return m.eval()


def test_cnn_deep_shap_is_exact(model):
    background, x = torch.randn(50, 24, 13), torch.randn(6, 24, 13)
    phi, expected = deep_shap(model.cnn, background, x)
    assert phi.shape == (6, 24, 13, 1) and expected.shape == (1,)
    with torch.no_grad():
        out = model.cnn(x).double().numpy()
    np.testing.assert_allclose(phi.sum(axis=(1, 2)) + expected, out, atol=1e-4)


def test_ensemble_attribution_is_weighted_sum(model):
    state = {k: v.clone() for k, v in model.state_dict().items()}
    background, x = torch.randn(40, 24, 13), torch.randn(5, 24, 13)
    expl = explain_ensemble(model, background, x)

    w = model.ensemble.weights.detach().double().numpy()
    np.testing.assert_allclose(expl.weights, w)
    np.testing.assert_allclose(expl.phi["ensemble"], w[0] * expl.phi["cnn"] + w[1] * expl.phi["transformer"])
    np.testing.assert_allclose(expl.expected["ensemble"],
                               w[0] * expl.expected["cnn"] + w[1] * expl.expected["transformer"])
    assert expl.gap["cnn"] < 1e-4 and np.isfinite(expl.gap["transformer"])
    assert expl.contributions().shape == (5, 13) and expl.importance().shape == (13,)

    # explaining leaves the model untouched (no SHAP hooks, buffers or weight changes)
    assert all(not hasattr(m, "x") for m in model.modules())
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())

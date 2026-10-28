import torch

from pi_ensemble.config import ModelConfig
from pi_ensemble.models import CNNBranch, EncoderBlock, PhysicsInformedEnsemble, TransformerBranch


def test_cnn_branch_matches_figure():
    cfg = ModelConfig()
    branch = CNNBranch(cfg)
    conv1, bn1, relu1, pool1, conv2, bn2, relu2, pool2 = branch.features
    for conv in (conv1, conv2):
        assert conv.kernel_size == (3,) and conv.out_channels == 64
    assert conv1.in_channels == 13
    for bn in (bn1, bn2):
        assert isinstance(bn, torch.nn.BatchNorm1d)
    for relu in (relu1, relu2):
        assert isinstance(relu, torch.nn.ReLU)
    for pool in (pool1, pool2):
        assert isinstance(pool, torch.nn.MaxPool1d) and pool.stride == 2

    x = torch.randn(8, 24, 13)
    assert branch.features(x.transpose(1, 2)).shape == (8, 64, 6)   # 24 -> 12 -> 6
    assert branch.encode(x).shape == (8, cfg.dense_dim)              # z_CNN
    assert branch(x).shape == (8, 1)                                 # y_CNN


def test_transformer_branch_matches_figure():
    cfg = ModelConfig()
    branch = TransformerBranch(cfg)
    assert branch.input_proj.in_features == 13 and branch.input_proj.out_features == 64
    assert len(branch.blocks) == 2
    for block in branch.blocks:
        assert isinstance(block, EncoderBlock)
        assert list(dict(block.named_children()))[:5] == ["norm1", "attn", "drop1", "norm2", "ffn"]
        assert block.attn.n_heads == 4 and block.attn.d_k == 16
        assert block.ffn.net[0].out_features == 256

    x = torch.randn(8, 24, 13)
    assert branch.encode(x).shape == (8, cfg.dense_dim)  # z_T
    assert branch(x).shape == (8, 1)                     # y_T


def test_weighted_ensemble_is_convex_combination():
    model = PhysicsInformedEnsemble(ModelConfig()).eval()
    with torch.no_grad():
        model.ensemble.logits.copy_(torch.tensor([0.3, -0.2]))
        out = model(torch.randn(5, 24, 13))
    w = model.ensemble.weights
    assert torch.isclose(w.sum(), torch.tensor(1.0)) and (w > 0).all()
    assert torch.allclose(out["ensemble"], w[0] * out["cnn"] + w[1] * out["transformer"])


def test_multi_step_horizon():
    model = PhysicsInformedEnsemble(ModelConfig(horizon=24)).eval()
    out = model(torch.randn(3, 24, 13))
    assert all(v.shape == (3, 24) for v in out.values())

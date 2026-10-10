import pytest
import torch

from src.models import build_model, count_macs, count_params
from src.models.blocks import ECA, SE, AttentionGate, CBAMChannel, DSConv, SpatialAttention, Upsample
from src.utils import load_config

VARIANTS = [
    {},
    {"base_channels": 8},
    {"levels": 3},
    {"conv": "standard"},
    {"norm": "bn"},
    {"upsample": "pixelshuffle"},
    {"attention_gates": False},
    {"global_residual": False},
    {"attention": {"channel": "none", "spatial": False}},
    {"attention": {"channel": "eca", "spatial": False}},
    {"attention": {"channel": "se", "spatial": True}},
    {"attention": {"channel": "none", "spatial": True, "bottleneck": False}},
]


@pytest.mark.parametrize("cfg", VARIANTS)
@pytest.mark.parametrize("hw", [(64, 64), (40, 60), (37, 53)])
def test_output_shape_and_range(cfg, hw):
    m = build_model(cfg).eval()
    x = torch.rand(2, 3, *hw)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert y.min() >= 0 and y.max() <= 1


def test_training_mode_backward():
    m = build_model({}).train()
    x = torch.rand(2, 3, 32, 48)
    y = m(x)
    y.mean().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_base_config_is_lightweight():
    cfg = load_config("configs/base.yaml")
    m = build_model(cfg["model"])
    assert count_params(m) < 1_000_000
    macs = count_macs(m, 400, 600)
    assert macs is None or macs > 0


def test_ablation_switches_change_parameter_count():
    full = count_params(build_model({}))
    assert count_params(build_model({"attention": {"channel": "none", "spatial": False}})) < full
    assert count_params(build_model({"attention_gates": False})) < full
    assert count_params(build_model({"base_channels": 8})) < full < count_params(
        build_model({"base_channels": 32}))


def test_identity_like_residual():
    """With global residual and a zeroed head the model returns the input."""
    m = build_model({}).eval()
    torch.nn.init.zeros_(m.head.weight)
    torch.nn.init.zeros_(m.head.bias)
    x = torch.rand(1, 3, 24, 40)
    with torch.no_grad():
        assert torch.allclose(m(x), x)


@pytest.mark.parametrize("block", [ECA(16), SE(16), CBAMChannel(16), SpatialAttention()])
def test_attention_blocks_preserve_shape(block):
    x = torch.rand(2, 16, 9, 11)
    assert block(x).shape == x.shape


def test_misc_blocks():
    x = torch.rand(1, 8, 10, 12)
    assert DSConv(8, 16)(x).shape == (1, 16, 10, 12)
    assert AttentionGate(8, 8)(x, x).shape == x.shape
    assert Upsample(8, 4, "bilinear")(x).shape == (1, 4, 20, 24)
    assert Upsample(8, 4, "pixelshuffle")(x).shape == (1, 4, 20, 24)


def test_all_ablation_configs_build():
    from pathlib import Path
    for p in sorted(Path("configs/ablations").glob("[alw]*.yaml")):
        if p.name == "index.yaml":
            continue
        cfg = load_config(p)
        assert cfg["train"]["epochs"] == 100, p
        build_model(cfg["model"])

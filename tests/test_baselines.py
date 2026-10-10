"""The baseline architectures must load the official checkpoints strictly. Downloads
small weight files from GitHub, so it is skipped when offline."""
import urllib.request

import pytest
import torch

from src import baselines as B


def _online(url):
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=10)
        return True
    except Exception:
        return False


@pytest.mark.parametrize("key,url", [("zero_dce", B.ZERO_DCE_URL), ("zero_dce_pp", B.ZERO_DCE_PP_URL)])
def test_zero_dce_family_loads_official_weights(key, url, tmp_path):
    if not _online(url):
        pytest.skip("offline")
    fn, m, info = B.LOADERS[key]("lol_v1", tmp_path, torch.device("cpu"))
    y = fn(torch.rand(1, 3, 40, 60) * 0.2)
    assert y.shape == (1, 3, 40, 60) and info["weights"] == "official"


def test_zero_dce_shapes_without_weights():
    x = torch.rand(1, 3, 33, 47)
    assert B.ZeroDCE()(x).shape == x.shape
    assert B.ZeroDCEpp()(x).shape == x.shape

import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
from src.datasets import PairedLowLightDataset, UnpairedLowLightDataset, augment_pair  # noqa: E402


def _write(root: Path, split: str, n: int, size=(600, 400)):
    for i in range(n):
        for role, val in (("low", 10), ("high", 200)):
            d = root / split / role
            d.mkdir(parents=True, exist_ok=True)
            arr = np.full((size[1], size[0], 3), val, np.uint8)
            arr[0, 0] = (i, i, i)
            Image.fromarray(arr).save(d / f"{i}.png")


def test_train_crops_and_ranges(tmp_path):
    _write(tmp_path, "train", 3)
    ds = PairedLowLightDataset(tmp_path, "train", patch_size=256, repeat=2)
    assert len(ds) == 6
    s = ds[0]
    assert s["low"].shape == (3, 256, 256) and s["high"].shape == (3, 256, 256)
    assert s["low"].dtype == torch.float32
    assert 0.0 <= s["low"].min() and s["high"].max() <= 1.0


def test_test_split_is_full_resolution(tmp_path):
    _write(tmp_path, "test", 2)
    ds = PairedLowLightDataset(tmp_path, "test", patch_size=None, augment=False)
    assert ds[1]["low"].shape == (3, 400, 600)
    assert ds[1]["name"] == "1"


def test_augment_pair_keeps_alignment():
    rng = random.Random(0)
    low = np.random.RandomState(0).randint(0, 255, (40, 60, 3), np.uint8)
    for _ in range(20):
        a, b = augment_pair(low, low.copy(), 16, rng)
        assert a.shape == (16, 16, 3)
        np.testing.assert_array_equal(a, b)


def test_small_images_are_padded():
    a, b = augment_pair(np.zeros((10, 12, 3), np.uint8), np.zeros((10, 12, 3), np.uint8), 16,
                        random.Random(0))
    assert a.shape == b.shape == (16, 16, 3)


def test_unpaired(tmp_path):
    d = tmp_path / "test" / "low"
    d.mkdir(parents=True)
    Image.fromarray(np.zeros((50, 80, 3), np.uint8)).save(d / "a.jpg")
    ds = UnpairedLowLightDataset(tmp_path, max_side=40)
    assert ds[0]["low"].shape == (3, 25, 40)

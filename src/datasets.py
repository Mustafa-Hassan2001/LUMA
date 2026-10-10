"""Paired / unpaired low-light datasets.

Expects the layout produced by ``data/download.py``::

    <root>/{train,test}/{low,high}/<stem>.png

Training samples are random ``patch_size`` crops with random flips and 90-degree
rotations (the 8 dihedral transforms). Test samples are full resolution.
Tensors are float32 RGB in [0, 1], shape (3, H, W).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Keep OpenCV from spawning threads inside every DataLoader worker.
cv2.setNumThreads(0)


def list_images(folder: Path) -> list:
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))


def read_rgb(path: Path) -> np.ndarray:
    """Read an image as uint8 RGB (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().div_(255.0)


def augment_pair(low: np.ndarray, high: np.ndarray, patch: Optional[int], rng: random.Random):
    """Same random crop + dihedral transform for both images."""
    h, w = low.shape[:2]
    if patch:
        if h < patch or w < patch:  # pad tiny images (reflect) so a crop always fits
            ph, pw = max(0, patch - h), max(0, patch - w)
            low = cv2.copyMakeBorder(low, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            high = cv2.copyMakeBorder(high, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            h, w = low.shape[:2]
        y, x = rng.randint(0, h - patch), rng.randint(0, w - patch)
        low, high = low[y:y + patch, x:x + patch], high[y:y + patch, x:x + patch]
    if rng.random() < 0.5:
        low, high = low[:, ::-1], high[:, ::-1]
    if rng.random() < 0.5:
        low, high = low[::-1], high[::-1]
    k = rng.randint(0, 3)
    if k:
        low, high = np.rot90(low, k), np.rot90(high, k)
    return low, high


class PairedLowLightDataset(Dataset):
    """Low/high pairs matched by filename stem."""

    def __init__(self, root, split: str = "train", patch_size: Optional[int] = 256,
                 augment: bool = True, cache_in_memory: bool = False, repeat: int = 1):
        self.root = Path(root)
        self.split = split
        low_dir, high_dir = self.root / split / "low", self.root / split / "high"
        if not low_dir.exists() or not high_dir.exists():
            raise FileNotFoundError(f"expected {low_dir} and {high_dir}; "
                                    f"run data/download.py first")
        highs = {p.stem: p for p in list_images(high_dir)}
        self.pairs = [(p, highs[p.stem]) for p in list_images(low_dir) if p.stem in highs]
        if not self.pairs:
            raise RuntimeError(f"no low/high pairs found under {self.root / split}")
        self.patch_size = patch_size
        self.augment = augment
        self.repeat = max(1, int(repeat))
        self._cache = None
        if cache_in_memory:  # LOL is small (~500 pairs, ~350 MB decoded): fits in Colab RAM
            self._cache = [(read_rgb(a), read_rgb(b)) for a, b in self.pairs]

    def __len__(self):
        return len(self.pairs) * self.repeat

    def _load(self, i):
        if self._cache is not None:
            return self._cache[i]
        a, b = self.pairs[i]
        return read_rgb(a), read_rgb(b)

    def __getitem__(self, idx):
        i = idx % len(self.pairs)
        low, high = self._load(i)
        if low.shape != high.shape:
            raise ValueError(f"size mismatch for pair {self.pairs[i][0].name}: "
                             f"{low.shape} vs {high.shape}")
        if self.augment:
            # Python's ``random`` is reseeded per worker by ``seed_worker`` below,
            # so augmentations are reproducible for a fixed global seed.
            low, high = augment_pair(low, high, self.patch_size, random)
        return {"low": to_tensor(low), "high": to_tensor(high), "name": self.pairs[i][0].stem}


class UnpairedLowLightDataset(Dataset):
    """Images without ground truth (DICM, LIME, MEF, NPE, VV): ``<root>/test/low``."""

    def __init__(self, root, max_side: Optional[int] = None):
        folder = Path(root) / "test" / "low"
        if not folder.exists():
            folder = Path(root)
        self.files = list_images(folder)
        if not self.files:
            raise RuntimeError(f"no images found in {folder}")
        self.max_side = max_side

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = read_rgb(self.files[idx])
        if self.max_side and max(img.shape[:2]) > self.max_side:
            s = self.max_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return {"low": to_tensor(img), "name": self.files[idx].stem}


def seed_worker(worker_id: int) -> None:
    """Derive per-worker seeds from torch's base seed (set by the DataLoader generator)."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def build_loaders(cfg: dict, seed: int = 0):
    """Train + validation loaders from the ``data`` section of a config."""
    d = cfg["data"]
    root = Path(d["root"]) / d["train_set"]
    val_root = Path(d["root"]) / d.get("val_set", d["train_set"])
    train_ds = PairedLowLightDataset(root, "train", patch_size=d["patch_size"], augment=True,
                                     cache_in_memory=d.get("cache_in_memory", False),
                                     repeat=d.get("repeat", 1))
    val_ds = PairedLowLightDataset(val_root, "test", patch_size=None, augment=False)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=d["batch_size"], shuffle=True,
                              num_workers=d.get("num_workers", 2), pin_memory=True,
                              drop_last=True, worker_init_fn=seed_worker, generator=g,
                              persistent_workers=d.get("num_workers", 2) > 0)
    # Full-resolution images: batch size 1 (sizes can differ between datasets).
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=d.get("num_workers", 2), pin_memory=True)
    return train_loader, val_loader

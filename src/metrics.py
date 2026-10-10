"""Image quality metrics.

Protocol (kept identical for our model and every baseline):

* PSNR: on float RGB images in [0, 1] after clamping, ``10 log10(1 / MSE)``
  (same formula as the Retinexformer evaluation code).
* SSIM: MATLAB-style SSIM (11x11 Gaussian window, sigma 1.5, K1=0.01, K2=0.03,
  'valid' region) computed per RGB channel on uint8 images and averaged.
* LPIPS: AlexNet variant (``lpips`` package), inputs scaled to [-1, 1].
* NIQE: no-reference, on grayscale, via ``pyiqa`` (optional dependency).

GT-mean correction (scale the output so its mean grey level equals the ground
truth's, as in KinD/LLFlow) uses the ground truth at test time, so it is NOT our
main protocol. ``gt_mean_correct`` exists only to report clearly-labelled extra numbers.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import torch


def to_float_np(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W)/(3,H,W) tensor -> (H,W,3) float64 in [0,1]."""
    if t.dim() == 4:
        t = t[0]
    return t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float64)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def _ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(k, k.T)
    mu1 = cv2.filter2D(a, -1, win)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, win)[5:-5, 5:-5]
    s11 = cv2.filter2D(a * a, -1, win)[5:-5, 5:-5] - mu1 ** 2
    s22 = cv2.filter2D(b * b, -1, win)[5:-5, 5:-5] - mu2 ** 2
    s12 = cv2.filter2D(a * b, -1, win)[5:-5, 5:-5] - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred/gt float in [0,1], (H,W,3). Converted to uint8 like the reference code."""
    a = np.round(np.clip(pred, 0, 1) * 255).astype(np.float64)
    b = np.round(np.clip(gt, 0, 1) * 255).astype(np.float64)
    return float(np.mean([_ssim_channel(a[..., i], b[..., i]) for i in range(a.shape[2])]))


def gt_mean_correct(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Scale prediction so its mean grey value matches the GT (uses GT! extra metric only)."""
    gp = cv2.cvtColor(pred.astype(np.float32), cv2.COLOR_RGB2GRAY).mean()
    gg = cv2.cvtColor(gt.astype(np.float32), cv2.COLOR_RGB2GRAY).mean()
    return np.clip(pred * (gg / max(gp, 1e-8)), 0, 1)


class LPIPSMetric:
    def __init__(self, device):
        import lpips
        self.fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        def t(a):
            return torch.from_numpy(a).float().permute(2, 0, 1)[None].to(self.device) * 2 - 1
        return float(self.fn(t(pred), t(gt)).item())


class NIQEMetric:
    """Wrapper around pyiqa's NIQE (lower is better). Raises ImportError if pyiqa is missing."""

    def __init__(self, device):
        import pyiqa
        self.fn = pyiqa.create_metric("niqe", device=device)
        self.device = device

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> float:
        t = torch.from_numpy(img).float().permute(2, 0, 1)[None].to(self.device)
        return float(self.fn(t).item())


def try_metric(cls, device) -> Optional[object]:
    try:
        return cls(device)
    except Exception as e:  # missing package or weights download failure
        print(f"[metrics] {cls.__name__} unavailable: {type(e).__name__}: {e}")
        return None

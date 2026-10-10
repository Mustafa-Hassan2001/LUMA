import numpy as np
import pytest
import torch

from src.losses import CharbonnierLoss, ColorLoss, CombinedLoss, ssim_torch
from src.metrics import gt_mean_correct, psnr, ssim


def test_psnr_known_value():
    a = np.zeros((4, 4, 3))
    b = np.full((4, 4, 3), 0.1)
    assert psnr(a, b) == pytest.approx(20.0)  # 10*log10(1/0.01)
    assert psnr(a, a) == 100.0


def test_ssim_identity_and_degradation():
    rng = np.random.RandomState(0)
    img = rng.rand(64, 64, 3)
    assert ssim(img, img) == pytest.approx(1.0)
    noisy = np.clip(img + rng.normal(0, 0.2, img.shape), 0, 1)
    assert ssim(noisy, img) < 0.9


def test_ssim_matches_reference_implementation():
    """Compare with the Retinexformer utils.ssim formula (cv2 Gaussian 11x11, valid crop)."""
    import cv2
    rng = np.random.RandomState(1)
    a = rng.rand(48, 40, 3)
    b = np.clip(a * 0.7 + 0.1, 0, 1)

    def ref(i1, i2):
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        k = cv2.getGaussianKernel(11, 1.5)
        w = np.outer(k, k.transpose())
        mu1 = cv2.filter2D(i1, -1, w)[5:-5, 5:-5]
        mu2 = cv2.filter2D(i2, -1, w)[5:-5, 5:-5]
        s1 = cv2.filter2D(i1 ** 2, -1, w)[5:-5, 5:-5] - mu1 ** 2
        s2 = cv2.filter2D(i2 ** 2, -1, w)[5:-5, 5:-5] - mu2 ** 2
        s12 = cv2.filter2D(i1 * i2, -1, w)[5:-5, 5:-5] - mu1 * mu2
        return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) /
                ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean()
    ua = np.round(a * 255).astype(np.float64)
    ub = np.round(b * 255).astype(np.float64)
    expected = np.mean([ref(ub[..., c], ua[..., c]) for c in range(3)])
    assert ssim(b, a) == pytest.approx(expected, abs=1e-10)


def test_gt_mean_correction_matches_brightness():
    import cv2
    gt = np.random.RandomState(0).rand(16, 16, 3) * 0.8
    pred = gt * 0.5
    c = gt_mean_correct(pred, gt)
    g = lambda x: cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2GRAY).mean()
    assert g(c) == pytest.approx(g(gt), rel=1e-3)


def test_losses_zero_for_identical_inputs():
    x = torch.rand(2, 3, 32, 32)
    assert CharbonnierLoss()(x, x) == pytest.approx(1e-3, rel=1e-3)
    assert ColorLoss()(x, x) == pytest.approx(0.0, abs=1e-5)
    assert ssim_torch(x, x) == pytest.approx(1.0, abs=1e-5)


def test_combined_loss_without_vgg():
    crit = CombinedLoss({"charbonnier": 1.0, "ssim": 0.3, "perceptual": 0.0, "color": 0.1})
    x, y = torch.rand(2, 3, 32, 32, requires_grad=True), torch.rand(2, 3, 32, 32)
    total, parts = crit(x, y)
    total.backward()
    assert set(parts) == {"charbonnier", "ssim", "color"}
    assert x.grad is not None


def test_all_zero_weights_rejected():
    with pytest.raises(ValueError):
        CombinedLoss({"charbonnier": 0})

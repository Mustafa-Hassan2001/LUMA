"""Training losses.

total = w_charb * Charbonnier + w_ssim * (1 - SSIM) + w_perc * VGG19 perceptual + w_color * color

Weights come from the ``loss`` section of the config; a weight of 0 disables the
term (and, for the perceptual loss, avoids loading VGG at all).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, x, y):
        return torch.sqrt((x.float() - y.float()) ** 2 + self.eps2).mean()


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()
    return w


def ssim_torch(x, y, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
    """Differentiable SSIM (Gaussian window, per channel, averaged). Inputs (B,C,H,W)."""
    x, y = x.float(), y.float()
    c = x.shape[1]
    w = _gaussian_window(window_size, sigma, c, x.device)
    pad = window_size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    sxx = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean()


class SSIMLoss(nn.Module):
    def forward(self, x, y):
        return 1.0 - ssim_torch(x, y)


class VGGPerceptualLoss(nn.Module):
    """L1 distance between VGG19 features (relu1_2, relu2_2, relu3_4, relu4_4)."""

    LAYERS = {3: 1.0, 8: 1.0, 17: 1.0, 26: 1.0}  # indices into vgg19.features (after ReLU)

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19
        feats = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[: max(self.LAYERS) + 1]
        for p in feats.parameters():
            p.requires_grad_(False)
        self.features = feats.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):  # VGG stays in eval mode
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        loss = 0.0
        for i, layer in enumerate(self.features):  # features are fp16 under autocast
            x, y = layer(x), layer(y)
            if i in self.LAYERS:
                loss = loss + self.LAYERS[i] * F.l1_loss(x.float(), y.detach().float())
        return loss


class ColorLoss(nn.Module):
    """Mean angle between predicted and target RGB vectors (1 - cosine similarity)."""

    def forward(self, x, y):
        return (1.0 - F.cosine_similarity(x.float(), y.float(), dim=1, eps=1e-6)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.w = {k: float(cfg.get(k, 0.0)) for k in ("charbonnier", "ssim", "perceptual", "color")}
        self.terms = nn.ModuleDict()
        if self.w["charbonnier"]:
            self.terms["charbonnier"] = CharbonnierLoss(cfg.get("charbonnier_eps", 1e-3))
        if self.w["ssim"]:
            self.terms["ssim"] = SSIMLoss()
        if self.w["perceptual"]:
            self.terms["perceptual"] = VGGPerceptualLoss()
        if self.w["color"]:
            self.terms["color"] = ColorLoss()
        if not self.terms:
            raise ValueError("all loss weights are zero")

    def forward(self, pred, target):
        parts = {name: term(pred, target) for name, term in self.terms.items()}
        total = sum(self.w[n] * v for n, v in parts.items())
        return total, {n: float(v.detach()) for n, v in parts.items()}

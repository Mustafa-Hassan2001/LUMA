"""Building blocks for the lightweight Attention-UNet.

* ``DSConv``          depthwise-separable 3x3 conv (depthwise 3x3 + pointwise 1x1)
* ``ConvBlock``       two convs (DS or standard) + norm + activation, optional residual
* ``ECA`` / ``SE``    channel attention (Wang et al. 2020 / Hu et al. 2018)
* ``CBAMChannel``     CBAM channel attention (avg + max pooled MLP)
* ``SpatialAttention``CBAM spatial attention (7x7 conv over [avg, max] maps)
* ``AttentionBlock``  configurable channel and/or spatial attention
* ``AttentionGate``   additive attention gate on skip connections (Oktay et al. 2018)
* ``Upsample``        bilinear + 1x1 conv, or PixelShuffle
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(inplace=True), "gelu": nn.GELU(), "silu": nn.SiLU(inplace=True),
            "lrelu": nn.LeakyReLU(0.2, inplace=True)}[name]


def make_norm(name: str, ch: int) -> nn.Module:
    if name in (None, "none"):
        return nn.Identity()
    if name == "bn":
        return nn.BatchNorm2d(ch)
    if name == "in":
        return nn.InstanceNorm2d(ch, affine=True)
    if name == "gn":
        return nn.GroupNorm(min(8, ch), ch)
    raise ValueError(f"unknown norm {name}")


class DSConv(nn.Module):
    """Depthwise 3x3 -> pointwise 1x1."""

    def __init__(self, cin: int, cout: int, k: int = 3, bias: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, padding=k // 2, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


def make_conv(kind: str, cin: int, cout: int) -> nn.Module:
    if kind == "ds":
        return DSConv(cin, cout)
    if kind == "standard":
        return nn.Conv2d(cin, cout, 3, padding=1)
    raise ValueError(f"unknown conv type {kind}")


class ConvBlock(nn.Module):
    """conv-norm-act-conv-norm-act with an optional 1x1 residual shortcut."""

    def __init__(self, cin: int, cout: int, conv: str = "ds", norm: str = "none",
                 act: str = "lrelu", residual: bool = True):
        super().__init__()
        self.body = nn.Sequential(
            make_conv(conv, cin, cout), make_norm(norm, cout), make_act(act),
            make_conv(conv, cout, cout), make_norm(norm, cout), make_act(act))
        self.skip = None
        if residual:
            self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        y = self.body(x)
        return y + self.skip(x) if self.skip is not None else y


class ECA(nn.Module):
    """Efficient Channel Attention: 1D conv over globally pooled channels."""

    def __init__(self, ch: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(ch) + b) / gamma))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)                       # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(1, 2))            # B,1,C
        return x * torch.sigmoid(y.transpose(1, 2).unsqueeze(-1))


class SE(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        hid = max(4, ch // reduction)
        self.fc = nn.Sequential(nn.Conv2d(ch, hid, 1), nn.ReLU(inplace=True), nn.Conv2d(hid, ch, 1))

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x.mean((2, 3), keepdim=True)))


class CBAMChannel(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        hid = max(4, ch // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(ch, hid, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(hid, ch, 1, bias=False))

    def forward(self, x):
        a = self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True))
        return x * torch.sigmoid(a)


class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        m = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(m))


CHANNEL_ATTN = {"eca": ECA, "se": SE, "cbam": CBAMChannel}


class AttentionBlock(nn.Module):
    """Channel attention (none/eca/se/cbam) followed by optional spatial attention.

    channel='cbam' + spatial=True is the full CBAM module.
    """

    def __init__(self, ch: int, channel: str = "cbam", spatial: bool = True):
        super().__init__()
        self.channel = CHANNEL_ATTN[channel](ch) if channel not in (None, "none") else nn.Identity()
        self.spatial = SpatialAttention() if spatial else nn.Identity()

    def forward(self, x):
        return self.spatial(self.channel(x))


class AttentionGate(nn.Module):
    """Additive attention gate (Attention U-Net). ``g`` is the decoder (gating) signal,
    ``x`` the encoder skip feature at the same resolution."""

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int | None = None):
        super().__init__()
        inter_ch = inter_ch or max(4, x_ch // 2)
        self.wg = nn.Conv2d(g_ch, inter_ch, 1, bias=False)
        self.wx = nn.Conv2d(x_ch, inter_ch, 1, bias=True)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=True)

    def forward(self, g, x):
        a = torch.sigmoid(self.psi(F.relu(self.wg(g) + self.wx(x), inplace=True)))
        return x * a


class Upsample(nn.Module):
    """x2 upsampling, then project to ``cout`` channels."""

    def __init__(self, cin: int, cout: int, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        if mode == "bilinear":
            self.proj = nn.Conv2d(cin, cout, 1)
        elif mode == "pixelshuffle":
            self.proj = nn.Sequential(nn.Conv2d(cin, cout * 4, 1), nn.PixelShuffle(2))
        else:
            raise ValueError(f"unknown upsample mode {mode}")

    def forward(self, x):
        if self.mode == "bilinear":
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.proj(x)

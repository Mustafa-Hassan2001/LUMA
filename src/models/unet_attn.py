"""Lightweight Attention-UNet for low-light enhancement.

    input ─ stem ─ E0 ───────────────────────────── gate ─ D0 ─ head ─(+ input)─ clamp
                   └ down ─ E1 ──────────────── gate ─ D1 ┘
                            └ down ─ E2 ───── gate ─ D2 ┘
                                     └ down ─ bottleneck(+attn) ┘

Every component can be switched from the ``model`` section of the config::

    model:
      base_channels: 16        # channels = [b, 2b, 4b, 8b] for levels=4
      levels: 4                # 3 or 4 resolution levels
      conv: ds                 # ds (depthwise-separable) | standard
      norm: none               # none | bn | in | gn
      act: lrelu
      attention:
        channel: cbam          # none | eca | se | cbam
        spatial: true          # CBAM spatial attention
        bottleneck: true       # apply attention in the bottleneck
        decoder: true          # apply attention after every decoder block
      attention_gates: true    # gate the skip connections
      upsample: bilinear       # bilinear | pixelshuffle
      global_residual: true    # output = input + residual
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AttentionBlock, AttentionGate, ConvBlock, Upsample, make_act

DEFAULTS = {
    "base_channels": 16, "channels": None, "levels": 4, "conv": "ds", "norm": "none",
    "act": "lrelu", "residual_blocks": True,
    "attention": {"channel": "cbam", "spatial": True, "bottleneck": True, "decoder": True},
    "attention_gates": True, "upsample": "bilinear", "global_residual": True,
    "in_channels": 3, "out_channels": 3,
}


def _merge(cfg: dict) -> dict:
    out = {**DEFAULTS, **{k: v for k, v in (cfg or {}).items() if k != "attention"}}
    out["attention"] = {**DEFAULTS["attention"], **((cfg or {}).get("attention") or {})}
    return out


class AttnUNet(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        c = _merge(cfg)
        self.cfg = c
        chs = c["channels"] or [c["base_channels"] * 2 ** i for i in range(c["levels"])]
        assert len(chs) >= 2, "need at least 2 levels"
        self.chs = chs
        self.factor = 2 ** (len(chs) - 1)
        att = c["attention"]
        use_ch = att["channel"] not in (None, "none")
        any_attn = use_ch or att["spatial"]

        def block(cin, cout):
            return ConvBlock(cin, cout, c["conv"], c["norm"], c["act"], c["residual_blocks"])

        def attn(ch):
            return AttentionBlock(ch, att["channel"], att["spatial"]) if any_attn else nn.Identity()

        self.stem = nn.Sequential(nn.Conv2d(c["in_channels"], chs[0], 3, padding=1),
                                  make_act(c["act"]))
        self.enc = nn.ModuleList([block(chs[i], chs[i]) for i in range(len(chs) - 1)])
        # Learned 2x2 stride-2 downsampling (cheap, no information thrown away like max-pool).
        self.down = nn.ModuleList([nn.Conv2d(chs[i], chs[i + 1], 2, stride=2)
                                   for i in range(len(chs) - 1)])
        self.bottleneck = block(chs[-1], chs[-1])
        self.bottleneck_attn = attn(chs[-1]) if att["bottleneck"] else nn.Identity()

        self.up = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.up.append(Upsample(chs[i + 1], chs[i], c["upsample"]))
            self.gates.append(AttentionGate(chs[i], chs[i]) if c["attention_gates"] else nn.Identity())
            self.fuse.append(nn.Conv2d(2 * chs[i], chs[i], 1))
            self.dec.append(block(chs[i], chs[i]))
            self.dec_attn.append(attn(chs[i]) if att["decoder"] else nn.Identity())
        self.head = nn.Conv2d(chs[0], c["out_channels"], 3, padding=1)
        self.global_residual = c["global_residual"]
        self.use_gates = c["attention_gates"]
        self.export_mode = False

    def forward(self, x):
        h, w = x.shape[-2:]
        if self.export_mode:  # ONNX: caller guarantees H, W are multiples of self.factor
            ph = pw = 0
        else:
            ph, pw = (-h) % self.factor, (-w) % self.factor
        inp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x

        f = self.stem(inp)
        skips = []
        for enc, down in zip(self.enc, self.down):
            f = enc(f)
            skips.append(f)
            f = down(f)
        f = self.bottleneck_attn(self.bottleneck(f))
        for up, gate, fuse, dec, dattn, skip in zip(self.up, self.gates, self.fuse, self.dec,
                                                    self.dec_attn, reversed(skips)):
            f = up(f)
            s = gate(f, skip) if self.use_gates else skip
            f = dattn(dec(fuse(torch.cat([f, s], 1))))
        out = self.head(f)
        if self.global_residual:
            out = out + inp
        if ph or pw:
            out = out[..., :h, :w]
        # Clamp only at inference: clamping during training zeroes the gradient of
        # every saturated pixel; the losses already pull outputs into [0, 1].
        return out if self.training else out.clamp(0, 1)


def build_model(model_cfg: dict) -> nn.Module:
    name = (model_cfg or {}).get("name", "attn_unet")
    if name != "attn_unet":
        raise ValueError(f"unknown model {name}")
    return AttnUNet(**{k: v for k, v in model_cfg.items() if k != "name"})


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, h: int = 400, w: int = 600) -> float | None:
    """Multiply-accumulates for one (1,3,h,w) forward pass, via thop (None if unavailable).

    FLOPs are commonly reported as 2 x MACs; we always state which one we print.
    """
    try:
        from thop import profile
    except ImportError:
        return None
    import copy
    m = copy.deepcopy(model).eval().cpu()
    with torch.no_grad():
        macs, _ = profile(m, inputs=(torch.zeros(1, 3, h, w),), verbose=False)
    return float(macs)


def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils import load_config

    ap = argparse.ArgumentParser(description="Print parameter count and MACs of a config's model")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.opts)
    model = build_model(cfg["model"])
    print(model)
    print(f"parameters: {count_params(model):,}")
    for h, w in ((400, 600), (1080, 1920)):
        macs = count_macs(model, h, w)
        print(f"MACs @ {w}x{h}: " + (f"{macs / 1e9:.3f} G" if macs else "thop not installed"))


if __name__ == "__main__":
    main()

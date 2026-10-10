"""Qualitative figures from saved outputs (``evaluate.py --save_images``).

    python src/figures.py --results /content/drive/MyDrive/lowlight-attn-unet/results \
        --data_root /content/data --dataset lol_v1 --methods zero_dce retinexformer ours

For each chosen test image: Input | <methods...> | Ground truth, with a zoomed
crop (red box) under every panel. The crop is placed where the ground truth has
the most detail (highest local variance), the same box for all panels.
Images: ``<results>/figures/<dataset>_<name>.png`` and a combined grid.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import list_images, read_rgb  # noqa: E402

PRETTY = {"ours": "Ours", "ours_lolv1": "Ours", "ours_lolv2real": "Ours", "ours_lolv2syn": "Ours",
          "zero_dce": "Zero-DCE", "zero_dce_pp": "Zero-DCE++", "retinexnet": "RetinexNet",
          "enlightengan": "EnlightenGAN", "retinexformer": "Retinexformer"}


def best_crop(gt: np.ndarray, frac: float = 0.25):
    g = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = g.shape
    ch, cw = max(16, int(h * frac)), max(16, int(w * frac))
    mean = cv2.blur(g, (cw, ch))
    var = cv2.blur(g * g, (cw, ch)) - mean ** 2
    m = ch // 2, cw // 2
    var[: m[0]], var[h - m[0]:], var[:, : m[1]], var[:, w - m[1]:] = -1, -1, -1, -1
    cy, cx = np.unravel_index(np.argmax(var), var.shape)
    return int(cy - ch // 2), int(cx - cw // 2), ch, cw


def panel(img, box, title):
    y, x, h, w = box
    crop = cv2.resize(img[y:y + h, x:x + w], (img.shape[1], int(img.shape[1] * h / w)),
                      interpolation=cv2.INTER_NEAREST)
    boxed = img.copy()
    cv2.rectangle(boxed, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), max(2, img.shape[1] // 200))
    return title, boxed, crop


def make_figure(names_imgs, box, out_path: Path, dpi=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(names_imgs)
    fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 4.2),
                           gridspec_kw={"height_ratios": [2, 1.4]})
    ax = np.array(ax).reshape(2, n)
    for j, (title, img) in enumerate(names_imgs):
        t, boxed, crop = panel(img, box, title)
        ax[0, j].imshow(boxed)
        ax[0, j].set_title(t, fontsize=10)
        ax[1, j].imshow(crop)
        for a in ax[:, j]:
            a.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dataset", default="lol_v1")
    ap.add_argument("--methods", nargs="+", default=["ours"])
    ap.add_argument("--images", nargs="*", default=None, help="image stems (default: first 4)")
    ap.add_argument("--max_images", type=int, default=4)
    args = ap.parse_args()

    res, data = Path(args.results), Path(args.data_root) / args.dataset
    low_dir, high_dir = data / "test" / "low", data / "test" / "high"
    stems = args.images or [p.stem for p in list_images(low_dir)][: args.max_images]
    methods = [m for m in args.methods if (res / "outputs" / m / args.dataset).exists()]
    missing = set(args.methods) - set(methods)
    if missing:
        print(f"[fig] no saved outputs for {sorted(missing)} (run evaluate with --save_images)")
    made = []
    for s in stems:
        low = read_rgb(next(low_dir.glob(s + ".*")))
        panels = [("Input", low)]
        for m in methods:
            p = res / "outputs" / m / args.dataset / f"{s}.png"
            if p.exists():
                panels.append((PRETTY.get(m, m), read_rgb(p)))
        gt_path = next(high_dir.glob(s + ".*"), None) if high_dir.exists() else None
        gt = read_rgb(gt_path) if gt_path else None
        if gt is not None:
            panels.append(("Ground truth", gt))
        box = best_crop(gt if gt is not None else panels[-1][1])
        out = res / "figures" / f"{args.dataset}_{s}.png"
        make_figure(panels, box, out)
        made.append(out)
        print(f"[fig] {out}")
    return made


if __name__ == "__main__":
    main()

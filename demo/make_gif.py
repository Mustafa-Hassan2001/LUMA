"""Make a before/after "wipe" GIF from real model outputs (for the README).

    python demo/make_gif.py --low /content/data/lol_v1/test/low \
        --enhanced .../results/outputs/ours_lolv1/lol_v1 --out results/figures/demo.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def wipe_frames(low: Image.Image, enh: Image.Image, steps=24, hold=8):
    w, h = low.size
    a, b = np.asarray(low), np.asarray(enh)
    xs = list(np.linspace(0, w, steps).astype(int))
    xs = [0] * hold + xs + [w] * hold + xs[::-1]
    frames = []
    for x in xs:
        f = a.copy()
        f[:, :x] = b[:, :x]
        im = Image.fromarray(f)
        d = ImageDraw.Draw(im)
        if 0 < x < w:
            d.line([(x, 0), (x, h)], fill=(255, 255, 255), width=2)
        d.text((8, 6), "enhanced" if x > w // 2 else "input", fill=(255, 255, 255))
        frames.append(im)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", required=True)
    ap.add_argument("--enhanced", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--width", type=int, default=480)
    args = ap.parse_args()
    enh_files = sorted(Path(args.enhanced).glob("*.png"))[: args.n]
    if not enh_files:
        raise SystemExit("no enhanced images found (run evaluate.py with --save_images)")
    frames = []
    for e in enh_files:
        low = next(Path(args.low).glob(e.stem + ".*"))
        a, b = Image.open(low).convert("RGB"), Image.open(e).convert("RGB")
        s = args.width / a.width
        size = (args.width, int(a.height * s))
        frames += wipe_frames(a.resize(size), b.resize(size))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=60, loop=0,
                   optimize=True)
    print(f"saved {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()

"""Gradio demo: upload a low-light photo, see before/after and the inference time.

    python demo/app.py --checkpoint /content/drive/MyDrive/lowlight-attn-unet/runs/ours_lolv1/best.pth --share

``--share`` prints a public *.gradio.live link (needed on Colab). Without a
checkpoint the app refuses to start rather than showing untrained output.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import load_checkpoint_model  # noqa: E402
from src.utils import get_device, gpu_name  # noqa: E402


class Enhancer:
    def __init__(self, checkpoint, device="auto", fp16=True):
        self.device = get_device(device)
        self.model, self.cfg, ck = load_checkpoint_model(checkpoint, self.device)
        self.fp16 = fp16 and self.device.type == "cuda"
        if self.fp16:
            self.model = self.model.half()
        self.epoch = int(ck.get("epoch", -1)) + 1
        self(np.zeros((64, 64, 3), np.uint8))  # warm-up (cuDNN autotune)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255
        x = x.to(self.device)
        if self.fp16:
            x = x.half()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        y = self.model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        out = (y[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        return out, ms


def build_ui(enh: Enhancer, examples=None):
    import gradio as gr

    def run(img, max_side):
        if img is None:
            return None, None, "Upload an image first."
        img = img[..., :3]
        h, w = img.shape[:2]
        if max_side and max(h, w) > max_side:
            import cv2
            s = max_side / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out, ms = enh(img)
        info = (f"**{img.shape[1]}×{img.shape[0]}** px · model time **{ms:.1f} ms** "
                f"(≈{1000 / ms:.0f} FPS) on {gpu_name()} ({'FP16' if enh.fp16 else 'FP32'})")
        return (img, out), out, info

    with gr.Blocks(title="LUMA low-light enhancement") as ui:
        gr.Markdown("# LUMA · Lightweight Attention-UNet for low-light enhancement\n"
                    "Upload a dark photo. The model brightens it and removes noise/colour casts. "
                    f"Checkpoint epoch {enh.epoch}.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="Low-light input")
                max_side = gr.Slider(256, 4096, value=1920, step=64,
                                     label="Resize longest side to (px)")
                btn = gr.Button("Enhance", variant="primary")
            with gr.Column():
                if hasattr(gr, "ImageSlider"):
                    cmp = gr.ImageSlider(label="Before / after (drag the slider)")
                else:
                    cmp = gr.Gallery(label="Before / after", columns=2)
                out = gr.Image(type="numpy", label="Enhanced (right-click to save)")
                info = gr.Markdown()
        btn.click(run, [inp, max_side], [cmp, out, info])
        if examples:
            gr.Examples(examples=[[e] for e in examples], inputs=inp)
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--examples", default=None, help="folder of example low-light images")
    ap.add_argument("--share", action="store_true", help="public link (use on Colab)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    enh = Enhancer(args.checkpoint, fp16=not args.fp32)
    ex = None
    if args.examples:
        ex = sorted(str(p) for p in Path(args.examples).iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})[:8]
    build_ui(enh, ex).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()

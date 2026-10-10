"""Train the Attention-UNet with automatic resume.

    python src/train.py --config configs/base.yaml
    python src/train.py --config configs/ablations/a01_no_attention.yaml --max_hours 2.5
    python src/train.py --config configs/base.yaml --opts train.epochs=5 data.batch_size=4

Everything for a run lives in ``<paths.runs>/<run_name>/``::

    config.yaml        exact config used (saved before training starts)
    latest.pth         full training state, written every epoch (atomic)
    best.pth           best validation PSNR so far
    metrics.jsonl      one JSON line per validation
    train.log          human-readable log
    tb/                TensorBoard logs
    samples/           input | output | ground truth at every validation

Re-running the same command resumes from ``latest.pth`` automatically. Use
``--max_hours`` to stop cleanly before a Colab session ends.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import build_loaders  # noqa: E402
from src.losses import CombinedLoss  # noqa: E402
from src.metrics import psnr, ssim, to_float_np  # noqa: E402
from src.models import build_model, count_macs, count_params  # noqa: E402
from src.utils import (get_device, git_commit, gpu_name, load_config, save_config,  # noqa: E402
                       set_seed)


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


def lr_lambda_factory(total_steps: int, warmup_steps: int, min_ratio: float):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return f


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


@torch.no_grad()
def validate(model, loader, device, amp: bool, max_samples: int = 4):
    model.eval()
    ps, ss, samples = [], [], []
    for batch in loader:
        low = batch["low"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(low)
        pred, gt = to_float_np(out.float()), to_float_np(batch["high"])
        ps.append(psnr(pred, gt))
        ss.append(ssim(pred, gt))
        if len(samples) < max_samples:
            samples.append(np.concatenate([to_float_np(batch["low"]), pred, gt], axis=1))
    model.train()
    return float(np.mean(ps)), float(np.mean(ss)), samples


def save_samples(samples, path: Path):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    w = max(s.shape[1] for s in samples)
    rows = [np.pad(s, ((0, 0), (0, w - s.shape[1]), (0, 0))) for s in samples]
    grid = (np.concatenate(rows, axis=0) * 255).round().astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return grid


def train(cfg: dict, run_name: str, max_hours: float | None = None, device_pref: str = "auto"):
    device = get_device(device_pref)
    run_dir = Path(cfg["paths"]["runs"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "train.log")
    seed = int(cfg.get("seed", 42))
    set_seed(seed, deterministic=cfg["train"].get("deterministic", False))

    tc = cfg["train"]
    amp = bool(tc.get("amp", True)) and device.type == "cuda"
    train_loader, val_loader = build_loaders(cfg, seed=seed)
    model = build_model(cfg["model"]).to(device)
    criterion = CombinedLoss(cfg["loss"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]),
                            weight_decay=float(tc.get("weight_decay", 1e-4)),
                            betas=tuple(tc.get("betas", (0.9, 0.999))))
    epochs = int(tc["epochs"])
    steps_per_epoch = len(train_loader)
    total = epochs * steps_per_epoch
    warm = int(float(tc.get("warmup_epochs", 0)) * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(total, warm, float(tc.get("min_lr", 1e-6)) / float(tc["lr"])))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, step, best = 0, 0, {"psnr": -1.0, "ssim": 0.0, "epoch": -1}
    elapsed_before = 0.0
    latest = run_dir / "latest.pth"
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
        elapsed_before = ck.get("elapsed_s", 0.0)
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        if torch.cuda.is_available() and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        log(f"RESUMED from {latest} at epoch {start_epoch} (best PSNR so far {best['psnr']:.3f})")
        old = (run_dir / "config.yaml").read_text() if (run_dir / "config.yaml").exists() else ""
        import yaml
        if old and yaml.safe_load(old) != cfg:
            save_config(cfg, run_dir / f"config_resume_epoch{start_epoch}.yaml")
            log("WARNING: config differs from the original run; saved the new one separately")
    else:
        save_config(cfg, run_dir / "config.yaml")
        info = {"params": count_params(model), "macs_600x400": count_macs(model, 400, 600),
                "gpu": gpu_name(), "git_commit": git_commit(), "torch": torch.__version__,
                "steps_per_epoch": steps_per_epoch, "total_steps": total}
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
        log(f"NEW RUN {run_name}: {json.dumps(info)}")

    # Different shuffling order after a resume (otherwise every resumed session
    # would replay the data order of epoch 0).
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(seed + start_epoch)

    if start_epoch >= epochs:
        log("training already finished")
        return best

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(str(run_dir / "tb"))
    val_every = int(tc.get("val_every", 5))
    clip = float(tc.get("grad_clip", 0) or 0)
    t0 = time.time()
    stop_reason = "finished"
    model.train()
    for epoch in range(start_epoch, epochs):
        ep_t, run_loss, n, skipped = time.time(), 0.0, 0, 0
        for batch in train_loader:
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(low)
                # Pixel/SSIM/color terms run in fp32 (they cast internally); VGG runs in fp16.
                loss, parts = criterion(out.float(), high)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss += float(loss.detach())
            n += 1
            if step % 50 == 0:
                tb.add_scalar("train/loss", float(loss.detach()), step)
                for k, v in parts.items():
                    tb.add_scalar(f"train/{k}", v, step)
                tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
        msg = (f"epoch {epoch + 1}/{epochs} loss {run_loss / max(n, 1):.4f} "
               f"lr {sched.get_last_lr()[0]:.2e} time {time.time() - ep_t:.0f}s")
        if skipped:
            msg += f" (skipped {skipped} non-finite steps)"

        is_last = epoch + 1 == epochs
        if (epoch + 1) % val_every == 0 or is_last:
            vp, vs, samples = validate(model, val_loader, device, amp)
            msg += f" | val PSNR {vp:.3f} SSIM {vs:.4f}"
            tb.add_scalar("val/psnr", vp, epoch + 1)
            tb.add_scalar("val/ssim", vs, epoch + 1)
            grid = save_samples(samples, run_dir / "samples" / f"epoch_{epoch + 1:04d}.jpg")
            tb.add_image("val/low_out_gt", grid, epoch + 1, dataformats="HWC")
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"epoch": epoch + 1, "step": step, "val_psnr": vp,
                                    "val_ssim": vs, "train_loss": run_loss / max(n, 1)}) + "\n")
            if vp > best["psnr"]:
                best = {"psnr": vp, "ssim": vs, "epoch": epoch + 1}
                atomic_save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                             "best": best}, run_dir / "best.pth")
                msg += " *best*"
        elapsed = elapsed_before + time.time() - t0
        atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "step": step, "best": best, "config": cfg,
                     "elapsed_s": elapsed,
                     "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                             "python": random.getstate(),
                             "cuda": torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None}}, latest)
        log(msg)
        if max_hours and (time.time() - t0) / 3600 > max_hours and not is_last:
            stop_reason = f"time budget of {max_hours} h reached"
            break
    tb.close()
    summary = {"run": run_name, "best_val_psnr": best["psnr"], "best_val_ssim": best["ssim"],
               "best_epoch": best["epoch"], "last_epoch": epoch + 1, "epochs": epochs,
               "total_train_hours": round((elapsed_before + time.time() - t0) / 3600, 3),
               "status": stop_reason}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"STOP ({stop_reason}). best val PSNR {best['psnr']:.3f} at epoch {best['epoch']}")
    if stop_reason != "finished":
        log("Re-run the same command to resume from latest.pth.")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--opts", nargs="*", default=[], help="overrides like train.epochs=10")
    ap.add_argument("--run_name", default=None, help="default: the config's `experiment`")
    ap.add_argument("--max_hours", type=float, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opts)
    train(cfg, args.run_name or cfg.get("experiment", "run"), args.max_hours, args.device)


if __name__ == "__main__":
    main()

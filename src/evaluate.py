"""Evaluate a trained checkpoint (or any enhancement function) on the test sets.

    python src/evaluate.py --checkpoint /content/drive/MyDrive/lowlight-attn-unet/runs/base/best.pth \
        --method ours --datasets lol_v1 lol_v2_real lol_v2_synthetic DICM LIME MEF NPE VV

Writes, under ``--out`` (default ``<paths.results>``)::

    metrics/<method>/<dataset>.json   per-image + mean metrics (main protocol)
    summary.csv                       one row per (method, dataset), updated in place
    outputs/<method>/<dataset>/*.png  enhanced images (with --save_images)

Main protocol = NO GT-mean correction. With ``--gt_mean`` an extra, separately
labelled block ``gt_mean_corrected`` is added to each JSON (and the CSV gets
``*_gtmean`` columns); it is never mixed into the main numbers.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import PairedLowLightDataset, UnpairedLowLightDataset  # noqa: E402
from src.metrics import (LPIPSMetric, NIQEMetric, gt_mean_correct, psnr, ssim,  # noqa: E402
                         to_float_np, try_metric)
from src.models import build_model  # noqa: E402
from src.utils import get_device, git_commit, gpu_name, load_json, save_json  # noqa: E402

PAIRED_SETS = ["lol_v1", "lol_v2_real", "lol_v2_synthetic"]
UNPAIRED_SETS = ["DICM", "LIME", "MEF", "NPE", "VV"]
CSV_FIELDS = ["method", "dataset", "n", "psnr", "ssim", "lpips", "niqe", "niqe_input",
              "psnr_gtmean", "ssim_gtmean", "lpips_gtmean", "trained_on", "source", "notes"]


def load_checkpoint_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["model"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ck


def model_fn(model, device, amp: bool = False) -> Callable:
    @torch.no_grad()
    def fn(low: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            return model(low.to(device)).float().clamp(0, 1)
    return fn


def _save_png(arr: np.ndarray, path: Path):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor((np.clip(arr, 0, 1) * 255).round().astype(np.uint8),
                                        cv2.COLOR_RGB2BGR))


def evaluate_paired(fn, root: Path, device, lpips_m=None, gt_mean=False, save_dir=None):
    ds = PairedLowLightDataset(root, "test", patch_size=None, augment=False)
    per = []
    for i in range(len(ds)):
        s = ds[i]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(s["low"][None])
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        pred, gt = to_float_np(out), to_float_np(s["high"])
        if pred.shape != gt.shape:
            raise ValueError(f"output shape {pred.shape} != GT {gt.shape} for {s['name']}")
        r = {"name": s["name"], "psnr": psnr(pred, gt), "ssim": ssim(pred, gt),
             "time_s_incl_transfer": dt}
        if lpips_m:
            r["lpips"] = lpips_m(pred, gt)
        if gt_mean:
            pc = gt_mean_correct(pred, gt)
            r["gt_mean_corrected"] = {"psnr": psnr(pc, gt), "ssim": ssim(pc, gt)}
            if lpips_m:
                r["gt_mean_corrected"]["lpips"] = lpips_m(pc, gt)
        if save_dir:
            _save_png(pred, Path(save_dir) / f"{s['name']}.png")
        per.append(r)
    keys = ["psnr", "ssim"] + (["lpips"] if lpips_m else [])
    mean = {k: float(np.mean([r[k] for r in per])) for k in keys}
    res = {"n": len(per), "mean": mean, "per_image": per}
    if gt_mean:
        res["gt_mean_corrected"] = {
            "WARNING": "uses the ground-truth mean brightness at test time; not the main protocol",
            "mean": {k: float(np.mean([r["gt_mean_corrected"][k] for r in per])) for k in keys}}
    return res


def evaluate_unpaired(fn, root: Path, device, niqe_m=None, save_dir=None, max_side=None):
    ds = UnpairedLowLightDataset(root, max_side=max_side)
    per = []
    for i in range(len(ds)):
        s = ds[i]
        out = fn(s["low"][None])
        pred = to_float_np(out)
        r = {"name": s["name"]}
        if niqe_m:
            r["niqe"] = niqe_m(pred)
            r["niqe_input"] = niqe_m(to_float_np(s["low"]))
        if save_dir:
            _save_png(pred, Path(save_dir) / f"{s['name']}.png")
        per.append(r)
    mean = {}
    if niqe_m:
        mean = {"niqe": float(np.mean([r["niqe"] for r in per])),
                "niqe_input": float(np.mean([r["niqe_input"] for r in per]))}
    return {"n": len(per), "mean": mean, "per_image": per}


def update_summary_csv(path: Path, row: dict):
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [r for r in csv.DictReader(f)
                    if not (r["method"] == row["method"] and r["dataset"] == row["dataset"])]
    rows.append({k: row.get(k, "") for k in CSV_FIELDS})
    rows.sort(key=lambda r: (r["method"], r["dataset"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _fmt(v, nd=4):
    return "" if v is None else f"{v:.{nd}f}"


def run_evaluation(fn, method: str, datasets, data_root: Path, out: Path, device,
                   gt_mean=False, save_images=False, trained_on="", source="", notes="",
                   extra_info=None, unpaired_max_side=None):
    lpips_m = try_metric(LPIPSMetric, device)
    niqe_m = None
    if any(d in UNPAIRED_SETS for d in datasets):
        niqe_m = try_metric(NIQEMetric, device)
    results = {}
    for d in datasets:
        root = data_root / d
        if not root.exists():
            print(f"[eval] skip {d}: {root} not found")
            continue
        save_dir = out / "outputs" / method / d if save_images else None
        t = time.time()
        if d in UNPAIRED_SETS:
            res = evaluate_unpaired(fn, root, device, niqe_m, save_dir, unpaired_max_side)
        else:
            res = evaluate_paired(fn, root, device, lpips_m, gt_mean, save_dir)
        res.update({"method": method, "dataset": d, "trained_on": trained_on, "source": source,
                    "notes": notes, "protocol": "no GT-mean correction; PSNR on float [0,1]; "
                    "MATLAB-style SSIM on uint8 per channel; LPIPS alex",
                    "gpu": gpu_name(), "git_commit": git_commit(), **(extra_info or {})})
        save_json(res, out / "metrics" / method / f"{d}.json")
        m = res["mean"]
        g = res.get("gt_mean_corrected", {}).get("mean", {})
        update_summary_csv(out / "summary.csv", {
            "method": method, "dataset": d, "n": res["n"], "psnr": _fmt(m.get("psnr"), 3),
            "ssim": _fmt(m.get("ssim")), "lpips": _fmt(m.get("lpips")),
            "niqe": _fmt(m.get("niqe"), 3), "niqe_input": _fmt(m.get("niqe_input"), 3),
            "psnr_gtmean": _fmt(g.get("psnr"), 3), "ssim_gtmean": _fmt(g.get("ssim")),
            "lpips_gtmean": _fmt(g.get("lpips")), "trained_on": trained_on, "source": source,
            "notes": notes})
        print(f"[eval] {method:<16} {d:<18} n={res['n']:<4} "
              + " ".join(f"{k}={v:.4f}" for k, v in m.items())
              + (" | GT-mean corrected: " + " ".join(f"{k}={v:.4f}" for k, v in g.items())
                 if g else "") + f"  ({time.time() - t:.0f}s)")
        results[d] = res
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="best.pth / latest.pth of a run")
    ap.add_argument("--method", default=None, help="name in the results (default: run folder)")
    ap.add_argument("--datasets", nargs="+", default=PAIRED_SETS + UNPAIRED_SETS)
    ap.add_argument("--data_root", default=None, help="default: data.root of the run's config")
    ap.add_argument("--out", default=None, help="default: paths.results of the run's config")
    ap.add_argument("--gt_mean", action="store_true",
                    help="ALSO report GT-mean-corrected scores (labelled separately)")
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--amp", action="store_true", help="FP16 inference (main results use FP32)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    model, cfg, ck = load_checkpoint_model(args.checkpoint, device)
    method = args.method or Path(args.checkpoint).parent.name
    data_root = Path(args.data_root or cfg["data"]["root"])
    out = Path(args.out or cfg["paths"]["results"])
    info = {"checkpoint": str(args.checkpoint), "checkpoint_epoch": int(ck.get("epoch", -1)) + 1,
            "precision": "fp16" if args.amp else "fp32"}
    run_info = Path(args.checkpoint).parent / "run_info.json"
    if run_info.exists():
        info["run_info"] = load_json(run_info)
    run_evaluation(model_fn(model, device, args.amp), method, args.datasets, data_root, out,
                   device, gt_mean=args.gt_mean, save_images=args.save_images,
                   trained_on=cfg["data"]["train_set"], source="ours (trained in this repo)",
                   extra_info=info)


if __name__ == "__main__":
    main()

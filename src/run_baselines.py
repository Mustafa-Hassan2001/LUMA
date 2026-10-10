"""Evaluate and benchmark the baselines with the same code used for our model.

    python src/run_baselines.py --data_root /content/data \
        --out /content/drive/MyDrive/lowlight-attn-unet/results \
        --cache /content/drive/MyDrive/lowlight-attn-unet/baselines

For every baseline: download weights (cached on Drive), evaluate on the requested
datasets (metrics JSON + summary.csv rows), benchmark params/MACs/FPS, and record
the status in ``<out>/baselines/status.json``. If a baseline fails, the error is
recorded and the comparison table falls back to the numbers in
``results/baselines/reported.yaml``, marked "reported".
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import LOADERS, PER_DATASET_WEIGHTS  # noqa: E402
from src.benchmark import benchmark_module, time_module  # noqa: E402
from src.evaluate import PAIRED_SETS, UNPAIRED_SETS, run_evaluation  # noqa: E402
from src.utils import get_device, gpu_name, load_json, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=list(LOADERS))
    ap.add_argument("--datasets", nargs="+", default=PAIRED_SETS + UNPAIRED_SETS)
    ap.add_argument("--data_root", default="/content/data")
    ap.add_argument("--out", default="/content/drive/MyDrive/lowlight-attn-unet/results")
    ap.add_argument("--cache", default="/content/drive/MyDrive/lowlight-attn-unet/baselines")
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--gt_mean", action="store_true")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--unpaired_max_side", type=int, default=None,
                    help="downscale very large unpaired images (e.g. 1600) to avoid OOM")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    out, cache, data_root = Path(args.out), Path(args.cache), Path(args.data_root)
    status_path = out / "baselines" / "status.json"
    status = load_json(status_path) if status_path.exists() else {}

    for method in args.methods:
        status[method] = {"gpu": gpu_name(), "datasets": {}}
        groups = ([[d] for d in args.datasets] if method in PER_DATASET_WEIGHTS
                  else [args.datasets])
        module = info = None
        for ds_group in groups:
            try:
                fn, module, info = LOADERS[method](ds_group[0], cache, device)
                run_evaluation(fn, method, ds_group, data_root, out, device,
                               gt_mean=args.gt_mean, save_images=args.save_images,
                               trained_on=info.get("trained_on", ""),
                               source=f"{info['weights']} - {info['source']}",
                               notes=info.get("notes", ""), extra_info={"baseline_info": info},
                               unpaired_max_side=args.unpaired_max_side)
                for d in ds_group:
                    status[method]["datasets"][d] = "ok"
            except Exception as e:
                traceback.print_exc()
                for d in ds_group:
                    status[method]["datasets"][d] = f"FAILED: {type(e).__name__}: {e}"
                print(f"[baselines] {method} failed on {ds_group}: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        # Efficiency (architecture is the same for every dataset's weights).
        try:
            if module is not None:
                b = benchmark_module(module, device, runs=args.runs, warmup=20)
            elif info is not None:  # ONNX-only baseline (EnlightenGAN): time the wrapper itself
                b = {"params": info.get("params"), "gpu": gpu_name(), "results": {},
                     "notes": "timed through the ONNX Runtime wrapper incl. I/O; MACs n/a"}
                for h, w in ((400, 600), (1080, 1920)):
                    x = torch.rand(1, 3, h, w)
                    b["results"][f"{w}x{h}"] = {"onnx": time_module(fn, x, 10, max(20, args.runs // 5),
                                                                    torch.device("cpu"))}
            else:
                raise RuntimeError("model never loaded")
            b["method"] = method
            b["baseline_info"] = info
            save_json(b, out / "benchmark" / f"{method}.json")
            status[method]["benchmark"] = "ok"
        except Exception as e:
            traceback.print_exc()
            status[method]["benchmark"] = f"FAILED: {type(e).__name__}: {e}"
        save_json(status, status_path)

    print("\n[baselines] status:")
    for m, s in status.items():
        print(f"  {m:<14} " + ", ".join(f"{d}: {'ok' if v == 'ok' else 'FAILED'}"
                                        for d, v in s["datasets"].items())
              + f" | benchmark: {'ok' if s.get('benchmark') == 'ok' else s.get('benchmark')}")


if __name__ == "__main__":
    main()

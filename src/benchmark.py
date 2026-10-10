"""Efficiency benchmark: params, MACs, latency and FPS.

    python src/benchmark.py --checkpoint .../runs/base/best.pth --method ours
    python src/benchmark.py --config configs/ablations/w08.yaml --method w08   # untrained is fine

Timing protocol: model in eval mode, ``torch.no_grad``, random input of the given
size already on the GPU, ``--warmup`` untimed runs, then ``--runs`` timed runs, each
bracketed by ``torch.cuda.synchronize()``. Reported: mean / median / p90 latency (ms)
and FPS = 1000 / mean latency. Pre/post-processing and host<->device copies are
excluded (stated in the JSON). FP16 uses ``model.half()``.
Output: ``<out>/benchmark/<method>.json``.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import build_model, count_macs, count_params  # noqa: E402
from src.utils import get_device, git_commit, gpu_name, load_config, save_json  # noqa: E402

DEFAULT_SIZES = [(400, 600), (1080, 1920)]


@torch.no_grad()
def time_module(fn, x, warmup: int, runs: int, device) -> dict:
    for _ in range(warmup):
        fn(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    a = np.array(times)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p90_ms": float(np.percentile(a, 90)), "std_ms": float(a.std()),
            "fps": float(1000.0 / a.mean()), "runs": runs, "warmup": warmup}


def benchmark_module(model: torch.nn.Module, device, sizes=DEFAULT_SIZES, warmup=50, runs=200,
                     fp16=True, macs=True) -> dict:
    model = model.eval().to(device)
    res = {"params": count_params(model), "gpu": gpu_name(), "device": str(device),
           "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
           "git_commit": git_commit(), "batch_size": 1,
           "timing_excludes": "image decode/encode, host<->device copies", "results": {}}
    for h, w in sizes:
        key = f"{w}x{h}"
        r = {}
        if macs:
            m = count_macs(model, h, w)
            r["macs_G"] = m / 1e9 if m else None
            r["flops_G_2xMACs"] = 2 * m / 1e9 if m else None
        x = torch.rand(1, 3, h, w, device=device)
        try:
            r["fp32"] = time_module(model, x, warmup, runs, device)
        except torch.cuda.OutOfMemoryError:
            r["fp32"] = "out of memory"
            torch.cuda.empty_cache()
        if fp16 and device.type == "cuda":
            import copy
            mh = copy.deepcopy(model).half()
            try:
                r["fp16"] = time_module(mh, x.half(), warmup, runs, device)
            except torch.cuda.OutOfMemoryError:
                r["fp16"] = "out of memory"
            del mh
            torch.cuda.empty_cache()
        res["results"][key] = r
        print(f"[bench] {key}: " + ", ".join(
            f"{p}: {v['mean_ms']:.2f} ms ({v['fps']:.1f} FPS)" if isinstance(v, dict) else f"{p}: {v}"
            for p, v in r.items() if p in ("fp32", "fp16"))
            + (f" | MACs {r['macs_G']:.3f} G" if r.get("macs_G") else ""))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint")
    g.add_argument("--config")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", default=None, help="results folder (default: paths.results)")
    ap.add_argument("--sizes", nargs="+", default=["600x400", "1920x1080"], help="WxH")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        model = build_model(cfg["model"])
        model.load_state_dict(ck["model"])
    else:
        cfg = load_config(args.config, args.opts)
        model = build_model(cfg["model"])
    sizes = [tuple(int(v) for v in s.lower().split("x"))[::-1] for s in args.sizes]
    res = benchmark_module(model, device, sizes, args.warmup, args.runs, not args.no_fp16)
    res.update({"method": args.method, "model_config": cfg["model"]})
    if device.type != "cuda":
        res["WARNING"] = "measured on CPU; not comparable to GPU numbers"
    out = Path(args.out or cfg["paths"]["results"])
    save_json(res, out / "benchmark" / f"{args.method}.json")
    print(f"[bench] saved {out / 'benchmark' / (args.method + '.json')}")


if __name__ == "__main__":
    main()

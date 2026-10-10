"""Export to ONNX, check it against PyTorch, and benchmark ONNX Runtime.

    python src/export.py --checkpoint .../runs/base/best.pth --method ours

Produces ``<out>/export/<method>_fp32.onnx`` (and ``_fp16.onnx`` on GPU), and
``<out>/benchmark/<method>_onnx.json`` with:
* max / mean absolute difference between ONNX Runtime and PyTorch outputs,
* ORT latency/FPS at 600x400 and 1920x1080 (CUDA provider if available, else CPU).

The ONNX graph has dynamic height/width; inputs must be multiples of
``model.factor`` (8 for 4 levels). 600x400 and 1920x1080 both are.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import build_model  # noqa: E402
from src.utils import get_device, gpu_name, save_json  # noqa: E402


def export_onnx(model, path: Path, device, fp16=False, opset=17):
    m = copy.deepcopy(model).eval().to(device)
    m.export_mode = True
    x = torch.rand(1, 3, 400, 600, device=device)
    if fp16:
        m, x = m.half(), x.half()
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["input"], output_names=["output"], opset_version=opset,
                  dynamic_axes={"input": {2: "height", 3: "width"},
                                "output": {2: "height", 3: "width"}})
    try:  # torch>=2.5 defaults to the dynamo exporter in some versions; force the stable one
        torch.onnx.export(m, (x,), str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(m, (x,), str(path), **kwargs)
    return path


def ort_session(path: Path):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    return ort.InferenceSession(str(path), providers=providers), providers[0]


def check_and_bench(model, onnx_path: Path, device, fp16: bool, sizes, warmup, runs):
    sess, provider = ort_session(onnx_path)
    dtype = np.float16 if fp16 else np.float32
    rep = {"onnx_file": onnx_path.name, "provider": provider, "precision": "fp16" if fp16 else "fp32",
           "size_mb": round(onnx_path.stat().st_size / 2 ** 20, 3), "check": {}, "results": {}}
    ref = copy.deepcopy(model).eval().to(device)
    if fp16:
        ref = ref.half()
    for h, w in sizes:
        x = torch.rand(1, 3, h, w, device=device)
        x = x.half() if fp16 else x
        with torch.no_grad():
            y_pt = ref(x).float().cpu().numpy()
        y_ort = sess.run(None, {"input": x.cpu().numpy().astype(dtype)})[0].astype(np.float32)
        diff = np.abs(y_pt - y_ort)
        rep["check"][f"{w}x{h}"] = {"max_abs_diff": float(diff.max()),
                                    "mean_abs_diff": float(diff.mean())}
        xin = x.cpu().numpy().astype(dtype)
        for _ in range(warmup):
            sess.run(None, {"input": xin})
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            sess.run(None, {"input": xin})
            ts.append((time.perf_counter() - t) * 1000)
        a = np.array(ts)
        rep["results"][f"{w}x{h}"] = {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
                                      "fps": float(1000 / a.mean()), "runs": runs,
                                      "note": "includes host<->device copies (numpy I/O)"}
        print(f"[onnx] {rep['precision']} {w}x{h} {provider}: max|diff|={diff.max():.2e} "
              f"{a.mean():.2f} ms ({1000 / a.mean():.1f} FPS)")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", default="ours")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sizes", nargs="+", default=["600x400", "1920x1080"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["model"])
    model.load_state_dict(ck["model"])
    model.eval()
    out = Path(args.out or cfg["paths"]["results"])
    sizes = [tuple(int(v) for v in s.lower().split("x"))[::-1] for s in args.sizes]
    for h, w in sizes:
        if h % model.factor or w % model.factor:
            raise SystemExit(f"{w}x{h} is not a multiple of {model.factor}")

    report = {"method": args.method, "gpu": gpu_name(), "runs": []}
    p32 = export_onnx(model, out / "export" / f"{args.method}_fp32.onnx", device)
    report["runs"].append(check_and_bench(model, p32, device, False, sizes, args.warmup, args.runs))
    if device.type == "cuda":
        p16 = export_onnx(model, out / "export" / f"{args.method}_fp16.onnx", device, fp16=True)
        report["runs"].append(check_and_bench(model, p16, device, True, sizes, args.warmup,
                                              args.runs))
    import onnxruntime
    report["onnxruntime"] = onnxruntime.__version__
    save_json(report, out / "benchmark" / f"{args.method}_onnx.json")
    print(f"[onnx] saved {out / 'benchmark' / (args.method + '_onnx.json')}")


if __name__ == "__main__":
    main()

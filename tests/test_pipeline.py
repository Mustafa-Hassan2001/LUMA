"""End-to-end smoke test on a tiny fake dataset (CPU): train -> resume -> evaluate
-> benchmark -> ONNX export. The VGG perceptual loss is disabled because it needs
to download ImageNet weights."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.utils import load_config

ROOT = Path(__file__).resolve().parent.parent


def fake_dataset(root: Path, name="lol_v1", n_train=6, n_test=2, size=(48, 40)):
    rng = np.random.RandomState(0)
    for split, n in (("train", n_train), ("test", n_test)):
        for i in range(n):
            high = rng.randint(80, 255, (size[1], size[0], 3)).astype(np.uint8)
            low = (high * 0.15).astype(np.uint8)
            for role, img in (("low", low), ("high", high)):
                d = root / name / split / role
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(d / f"{i}.png")


@pytest.fixture
def cfg(tmp_path):
    fake_dataset(tmp_path / "data")
    c = load_config(ROOT / "configs" / "base.yaml", [
        f"data.root={tmp_path / 'data'}", "data.patch_size=32", "data.batch_size=2",
        "data.repeat=1", "data.num_workers=0", "data.cache_in_memory=false",
        "model.base_channels=4", "loss.perceptual=0", "train.epochs=2", "train.val_every=1",
        "train.warmup_epochs=1", f"paths.runs={tmp_path / 'runs'}",
        f"paths.results={tmp_path / 'results'}"])
    return c


def test_train_resume_evaluate_benchmark_export(cfg, tmp_path):
    from src.train import train
    best = train(cfg, "smoke", device_pref="cpu")
    run = tmp_path / "runs" / "smoke"
    for f in ("config.yaml", "latest.pth", "best.pth", "metrics.jsonl", "train.log",
              "run_info.json", "train_summary.json"):
        assert (run / f).exists(), f
    assert best["epoch"] >= 1
    assert len((run / "metrics.jsonl").read_text().strip().splitlines()) == 2

    # Resume: raise the epoch budget and continue from latest.pth.
    cfg["train"]["epochs"] = 3
    train(cfg, "smoke", device_pref="cpu")
    assert "RESUMED from" in (run / "train.log").read_text()
    assert json.loads((run / "train_summary.json").read_text())["last_epoch"] == 3

    from src.evaluate import load_checkpoint_model, model_fn, run_evaluation
    dev = torch.device("cpu")
    model, mcfg, _ = load_checkpoint_model(run / "best.pth", dev)
    res = run_evaluation(model_fn(model, dev), "smoke", ["lol_v1"], tmp_path / "data",
                         tmp_path / "results", dev, gt_mean=True, save_images=True)
    r = res["lol_v1"]
    assert r["n"] == 2 and 0 < r["mean"]["psnr"] < 100
    assert "gt_mean_corrected" in r and "WARNING" in r["gt_mean_corrected"]
    assert (tmp_path / "results" / "summary.csv").exists()
    assert len(list((tmp_path / "results" / "outputs" / "smoke" / "lol_v1").glob("*.png"))) == 2

    from src.benchmark import benchmark_module
    b = benchmark_module(model, dev, sizes=[(40, 48)], warmup=1, runs=3, fp16=False)
    assert b["results"]["48x40"]["fp32"]["fps"] > 0

    pytest.importorskip("onnxruntime")
    from src.export import check_and_bench, export_onnx
    p = export_onnx(model, tmp_path / "m.onnx", dev)
    rep = check_and_bench(model, p, dev, False, [(40, 48), (64, 96)], 1, 2)
    for v in rep["check"].values():
        assert v["max_abs_diff"] < 1e-4

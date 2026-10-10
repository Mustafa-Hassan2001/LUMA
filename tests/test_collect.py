"""collect_results builds tables and report macros only from files that exist."""
import json

from src import collect_results as C


def _metrics(root, method, ds, psnr, ssim, lpips=0.1, **extra):
    p = root / "metrics" / method / f"{ds}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"n": 15, "mean": {"psnr": psnr, "ssim": ssim, "lpips": lpips},
                             "trained_on": "lol_v1", **extra}))


def _bench(root, method, params, macs, fps):
    p = root / "benchmark" / f"{method}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"params": params, "gpu": "Tesla T4", "results": {
        "600x400": {"macs_G": macs, "fp32": {"mean_ms": 1000 / fps, "fps": fps},
                    "fp16": {"mean_ms": 500 / fps, "fps": 2 * fps}}}}))


def test_collect_tables_and_macros(tmp_path, monkeypatch):
    res = tmp_path / "results"
    _metrics(res, "ours_lolv1", "lol_v1", 21.5, 0.80)
    _bench(res, "ours_lolv1", 144000, 2.6, 300.0)
    _metrics(res, "zero_dce", "lol_v1", 14.9, 0.56)
    _bench(res, "zero_dce", 79416, 19.0, 500.0)
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\ny\n")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    (tmp_path / "configs" / "ablations").mkdir(parents=True)
    (tmp_path / "configs" / "ablations" / "index.yaml").write_text(
        "width:\n  title: Width\n  rows:\n    - {run: w08, label: w8}\n")
    monkeypatch.setattr("sys.argv", ["x", "--results", str(res), "--readme", str(readme)])
    C.main()
    txt = readme.read_text()
    assert "21.50" in txt and "Zero-DCE" in txt and "old" not in txt
    nums = (tmp_path / "report" / "generated" / "numbers.tex").read_text()
    assert r"\newcommand{\OursPSNRLolVOne}{21.50}" in nums
    assert r"\newcommand{\OursFPS}{300.0}" in nums
    assert r"\OursPSNRLolVTwoReal}{\pending{}}" in nums      # missing -> pending, never invented
    assert (res / "tables" / "comparison_lol_v1.tex").exists()
    assert not (res / "tables" / "ablation_width.md").exists()  # no ablation results yet

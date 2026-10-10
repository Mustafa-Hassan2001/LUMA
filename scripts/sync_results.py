"""Copy result files from Google Drive into the repo's results/ folder and rebuild tables.

    python scripts/sync_results.py --drive_results /content/drive/MyDrive/lowlight-attn-unet/results

Copies metrics/, benchmark/, baselines/status.json, tables/, figures/ and data/
(JSON, CSV, small images). Large files (enhanced outputs, ONNX models) stay on Drive.
Then runs src/collect_results.py so README tables and report numbers match the files.
Optionally commits and pushes with --push (needs a GitHub token, see notebook 02).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEEP = ("metrics", "benchmark", "baselines", "figures", "data", "summary.csv")
SKIP_SUFFIX = {".onnx", ".pth"}
MAX_BYTES = 5 * 2 ** 20


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive_results", required=True)
    ap.add_argument("--runs", default=None, help="Drive runs folder: copies each run's "
                    "config.yaml, metrics.jsonl, train_summary.json, run_info.json")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--message", default="Update results from Colab")
    args = ap.parse_args()
    src, dst = Path(args.drive_results), ROOT / "results"
    n = 0
    for item in KEEP:
        s = src / item
        if not s.exists():
            continue
        files = [s] if s.is_file() else [p for p in s.rglob("*") if p.is_file()]
        for f in files:
            if f.suffix in SKIP_SUFFIX or f.stat().st_size > MAX_BYTES:
                continue
            rel = f.relative_to(src)
            if rel.parts[0] == "baselines" and f.name == "reported.yaml":
                continue  # the repo's reported.yaml is the source of truth
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / rel)
            n += 1
    if args.runs:
        for run in Path(args.runs).iterdir():
            for name in ("config.yaml", "metrics.jsonl", "train_summary.json", "run_info.json"):
                f = run / name
                if f.exists():
                    (dst / "runs" / run.name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst / "runs" / run.name / name)
                    n += 1
    print(f"copied {n} files into {dst}")
    subprocess.run([sys.executable, str(ROOT / "src" / "collect_results.py"),
                    "--results", str(dst)], check=True)
    if args.push:
        subprocess.run(["git", "-C", str(ROOT), "add", "results", "README.md", "report/generated"],
                       check=True)
        r = subprocess.run(["git", "-C", str(ROOT), "commit", "-m", args.message])
        if r.returncode == 0:
            subprocess.run(["git", "-C", str(ROOT), "push"], check=True)
        else:
            print("nothing to commit")


if __name__ == "__main__":
    main()

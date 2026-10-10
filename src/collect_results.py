"""Build every table / plot / report number from the saved results. Nothing is typed by hand.

    python src/collect_results.py --results results --runs /content/drive/MyDrive/lowlight-attn-unet/runs

Reads
    <results>/metrics/<method>/<dataset>.json   (evaluate.py / run_baselines.py)
    <results>/benchmark/<method>.json           (benchmark.py / run_baselines.py)
    <results>/benchmark/<method>_onnx.json      (export.py)
    <results>/baselines/status.json             (run_baselines.py)
    results/baselines/reported.yaml             (published numbers, used only as fallback)
    configs/ablations/index.yaml                (which runs form each ablation table)
Writes
    <results>/tables/*.md and *.tex
    <results>/figures/psnr_vs_macs.png
    report/generated/numbers.tex    (\\newcommand macros used by report/main.tex)
    README.md                       (section between <!-- RESULTS:START/END --> markers)
Missing results show as "–" in tables and "[pending]" in the report, never a made-up value.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRETTY = {"zero_dce": "Zero-DCE", "zero_dce_pp": "Zero-DCE++", "retinexnet": "RetinexNet",
          "enlightengan": "EnlightenGAN", "retinexformer": "Retinexformer"}
BASELINES = ["zero_dce", "zero_dce_pp", "retinexnet", "enlightengan", "retinexformer"]
DS_PRETTY = {"lol_v1": "LOL-v1", "lol_v2_real": "LOL-v2-Real",
             "lol_v2_synthetic": "LOL-v2-Synthetic"}
DASH = "–"


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fmt(v, nd):
    return DASH if v is None else f"{v:.{nd}f}"


class Results:
    def __init__(self, root: Path):
        self.root = root

    def metrics(self, method, dataset):
        return jload(self.root / "metrics" / method / f"{dataset}.json")

    def mean(self, method, dataset, key):
        m = self.metrics(method, dataset)
        return None if m is None else m.get("mean", {}).get(key)

    def gtmean(self, method, dataset, key):
        m = self.metrics(method, dataset)
        return None if m is None else m.get("gt_mean_corrected", {}).get("mean", {}).get(key)

    def bench(self, method):
        return jload(self.root / "benchmark" / f"{method}.json")

    def eff(self, method, size="600x400", prec="fp32"):
        """(params, MACs G, FPS, gpu) from a benchmark JSON."""
        b = self.bench(method)
        if not b:
            return None, None, None, None
        r = b.get("results", {}).get(size, {})
        t = r.get(prec) if isinstance(r.get(prec), dict) else r.get("onnx")
        return b.get("params"), r.get("macs_G"), (t or {}).get("fps"), b.get("gpu")


# --------------------------------------------------------------------------- table writers
def md_table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace(DASH, "--").replace("↑", r"$\uparrow$")
            .replace("↓", r"$\downarrow$").replace("†", r"$^\dagger$").replace("×", r"$\times$"))


def tex_table(header, rows, caption, label, bold_best=None):
    cols = "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\resizebox{\columnwidth}{!}{%",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(tex_escape(h) for h in header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(tex_escape(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_table(out_dir: Path, name, header, rows, caption):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.md").write_text(f"**{caption}**\n\n" + md_table(header, rows))
    (out_dir / f"{name}.tex").write_text(tex_table(header, rows, caption, f"tab:{name}"))
    return md_table(header, rows)


# --------------------------------------------------------------------------- tables
def main_table(R: Results, ours: dict):
    """Our model on every paired set. ``ours`` maps dataset -> method name to read."""
    rows, rows_gt = [], []
    for ds in DS_PRETTY:
        m = ours.get(ds)
        if not m or R.metrics(m, ds) is None:
            continue
        meta = R.metrics(m, ds)
        rows.append([DS_PRETTY[ds], DS_PRETTY.get(meta.get("trained_on"), meta.get("trained_on", "")),
                     meta["n"],
                     fmt(R.mean(m, ds, "psnr"), 2), fmt(R.mean(m, ds, "ssim"), 3),
                     fmt(R.mean(m, ds, "lpips"), 3)])
        if R.gtmean(m, ds, "psnr") is not None:
            rows_gt.append([DS_PRETTY[ds], fmt(R.gtmean(m, ds, "psnr"), 2),
                            fmt(R.gtmean(m, ds, "ssim"), 3), fmt(R.gtmean(m, ds, "lpips"), 3)])
    return rows, rows_gt


def load_reported():
    p = ROOT / "results" / "baselines" / "reported.yaml"
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def comparison_rows(R: Results, ds: str, ours_method: str, reported: dict):
    rows, points = [], []
    for b in BASELINES + [ours_method]:
        name = "Ours" if b == ours_method else PRETTY[b]
        params, macs, fps, _ = R.eff(b)
        psnr_, ssim_, lp = R.mean(b, ds, "psnr"), R.mean(b, ds, "ssim"), R.mean(b, ds, "lpips")
        src = "measured"
        if psnr_ is None and b != ours_method:
            rep = (reported.get(b) or {}).get(ds) or {}
            if rep.get("psnr") is not None:
                psnr_, ssim_, lp = rep.get("psnr"), rep.get("ssim"), rep.get("lpips")
                src = "reported†"
                params = params or (reported.get(b) or {}).get("params")
                macs = macs or (reported.get(b) or {}).get("macs_G_600x400")
        if psnr_ is None:
            continue
        info = (R.metrics(b, ds) or {}).get("baseline_info", {})
        if "UNOFFICIAL" in str(info.get("weights", "")):
            src += " (unofficial port)"
        rows.append([name, fmt(psnr_, 2), fmt(ssim_, 3), fmt(lp, 3),
                     DASH if params is None else f"{params / 1e6:.3f}",
                     fmt(macs, 2), fmt(fps, 1), src])
        if macs:
            points.append((name, macs, psnr_, src.startswith("reported")))
    return rows, points


def ablation_tables(R: Results, ds="lol_v1"):
    idx = yaml.safe_load((ROOT / "configs" / "ablations" / "index.yaml").read_text())
    out = {}
    for key, spec in idx.items():
        rows = []
        for r in spec["rows"]:
            run = r["run"]
            params, macs, fps, _ = R.eff(run)
            rows.append([r["label"], fmt(R.mean(run, ds, "psnr"), 2), fmt(R.mean(run, ds, "ssim"), 3),
                         fmt(R.mean(run, ds, "lpips"), 3),
                         DASH if params is None else f"{params / 1e3:.1f}", fmt(macs, 2),
                         fmt(fps, 1)])
        out[key] = (spec["title"], rows)
    return out


def efficiency_rows(R: Results, method: str):
    b = R.bench(method)
    if not b:
        return []
    rows = []
    for size, r in b.get("results", {}).items():
        for prec in ("fp32", "fp16"):
            t = r.get(prec)
            if isinstance(t, dict):
                rows.append([f"PyTorch {prec.upper()}", size, fmt(r.get("macs_G"), 2),
                             fmt(t["mean_ms"], 2), fmt(t["fps"], 1)])
    onnx = R.bench(f"{method}_onnx")
    for run in (onnx or {}).get("runs", []):
        for size, t in run.get("results", {}).items():
            rows.append([f"ONNX Runtime {run['precision'].upper()} ({run['provider'].replace('ExecutionProvider', '')})",
                         size, DASH, fmt(t["mean_ms"], 2), fmt(t["fps"], 1)])
    return rows


def plot_psnr_vs_macs(points, path: Path, ds_name: str):
    if not points:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, macs, psnr_, reported in points:
        ours = name == "Ours"
        ax.scatter(macs, psnr_, s=90 if ours else 50, marker="*" if ours else ("o" if not reported else "s"),
                   facecolors=("#d62728" if ours else ("none" if reported else "#1f77b4")),
                   edgecolors="#d62728" if ours else "#1f77b4", zorder=3)
        ax.annotate(name + (" (reported)" if reported else ""), (macs, psnr_),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("GMACs per 600×400 image (log scale)")
    ax.set_ylabel(f"PSNR on {ds_name} (dB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Quality vs. compute")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- report macros
# LaTeX macro names may only contain letters.
MACRO_DS = {"lol_v1": "LolVOne", "lol_v2_real": "LolVTwoReal", "lol_v2_synthetic": "LolVTwoSyn"}
MACRO_BASELINE = {"zero_dce": "ZeroDCE", "zero_dce_pp": "ZeroDCEPP", "retinexnet": "RetinexNet",
                  "enlightengan": "EnlightenGAN", "retinexformer": "Retinexformer"}


def write_numbers_tex(R: Results, ours_method: str, path: Path):
    def v(x, nd):
        return r"\pending{}" if x is None else f"{x:.{nd}f}"
    params, macs, fps, gpu = R.eff(ours_method)
    _, macs_fhd, fps_fhd, _ = R.eff(ours_method, "1920x1080")
    _, _, fps16, _ = R.eff(ours_method, "600x400", "fp16")
    _, _, fps16_fhd, _ = R.eff(ours_method, "1920x1080", "fp16")
    m = {
        "OursParamsK": v(params / 1e3 if params else None, 1),
        "OursParamsM": v(params / 1e6 if params else None, 3),
        "OursMACs": v(macs, 2), "OursMACsFHD": v(macs_fhd, 2),
        "OursFPS": v(fps, 1), "OursFPSHalf": v(fps16, 1),
        "OursFPSFHD": v(fps_fhd, 1), "OursFPSFHDHalf": v(fps16_fhd, 1),
        "GPUName": gpu.replace("_", " ") if gpu else r"\pending{}",
    }
    for ds in DS_PRETTY:
        for k, nd in (("psnr", 2), ("ssim", 3), ("lpips", 3)):
            m[f"Ours{k.upper()}{MACRO_DS[ds]}"] = v(R.mean(ours_method, ds, k), nd)
            m[f"Ours{k.upper()}{MACRO_DS[ds]}GTMean"] = v(R.gtmean(ours_method, ds, k), nd)
    for b in BASELINES:
        m[f"{MACRO_BASELINE[b]}PSNRLolVOne"] = v(R.mean(b, "lol_v1", "psnr"), 2)
    for ds in ("DICM", "LIME", "MEF", "NPE", "VV"):
        m[f"OursNIQE{ds}"] = v(R.mean(ours_method, ds, "niqe"), 2)
    lines = ["% AUTO-GENERATED by src/collect_results.py from results/. Do not edit.",
             r"\providecommand{\pending}{\textcolor{red}{[pending]}}"]
    lines += [rf"\newcommand{{\{k}}}{{{val}}}" for k, val in m.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return m


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--ours", default="ours_lolv1", help="method name of the main model")
    ap.add_argument("--ours_v2real", default="ours_lolv2real")
    ap.add_argument("--ours_v2syn", default="ours_lolv2syn")
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--no_readme", action="store_true")
    args = ap.parse_args()

    R = Results(Path(args.results))
    tdir = Path(args.results) / "tables"
    reported = load_reported()
    md = []

    # Main results: prefer in-domain models for LOL-v2 if they exist.
    ours_by_ds = {"lol_v1": args.ours}
    for ds, m in (("lol_v2_real", args.ours_v2real), ("lol_v2_synthetic", args.ours_v2syn)):
        ours_by_ds[ds] = m if R.metrics(m, ds) else args.ours
    rows, rows_gt = main_table(R, ours_by_ds)
    hdr = ["Test set", "Trained on", "Images", "PSNR ↑", "SSIM ↑", "LPIPS ↓"]
    if rows:
        md.append("### Main results (no GT-mean correction)\n\n" + write_table(
            tdir, "main_results", hdr, rows,
            "Our model on the paired test sets. No GT-mean brightness correction."))
    if rows_gt:
        md.append("<details><summary>Supplementary: GT-mean-corrected scores (uses ground-truth "
                  "brightness at test time; not comparable to the main table)</summary>\n\n" +
                  write_table(tdir, "main_results_gtmean", ["Test set", "PSNR ↑", "SSIM ↑", "LPIPS ↓"],
                              rows_gt, "Supplementary: GT-mean-corrected scores "
                              "(output rescaled to the ground-truth mean brightness).")
                  + "\n</details>\n")

    # Unpaired NIQE.
    nrows = []
    for ds in ("DICM", "LIME", "MEF", "NPE", "VV"):
        row = [ds, fmt(R.mean(args.ours, ds, "niqe_input"), 2)]
        row += [fmt(R.mean(b, ds, "niqe"), 2) for b in BASELINES + [args.ours]]
        if any(c != DASH for c in row[1:]):
            nrows.append(row)
    if nrows:
        md.append("### No-reference quality (NIQE ↓) on unpaired sets\n\n" + write_table(
            tdir, "niqe", ["Set", "Input"] + [PRETTY[b] for b in BASELINES] + ["Ours"], nrows,
            "NIQE (lower is better) on test sets without ground truth."))

    # Comparison with baselines.
    status = jload(Path(args.results) / "baselines" / "status.json") or {}
    for ds in DS_PRETTY:
        ours_m = ours_by_ds.get(ds, args.ours)
        crow, points = comparison_rows(R, ds, ours_m, reported)
        if len(crow) > 1:
            gpu = R.eff(ours_m)[3] or "GPU"
            cap = (f"Comparison on {DS_PRETTY[ds]}. Same metric code for all measured rows. "
                   f"MACs at 600×400; FPS = FP32, batch 1, 600×400 on {gpu}. "
                   "† reported in the literature, not re-measured.")
            md.append(f"### Comparison on {DS_PRETTY[ds]}\n\n" + write_table(
                tdir, f"comparison_{ds}", ["Method", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "Params (M)",
                                           "GMACs", "FPS", "Source"], crow, cap))
            if ds == "lol_v1" and plot_psnr_vs_macs(points, Path(args.results) / "figures" /
                                                     "psnr_vs_macs.png", DS_PRETTY[ds]):
                md.append("![PSNR vs MACs](results/figures/psnr_vs_macs.png)\n")
    if status:
        failed = {m: s for m, s in status.items()
                  if any(v != "ok" for v in s.get("datasets", {}).values())}
        if failed:
            md.append("Baselines that could not be run on every set: " + ", ".join(
                f"{PRETTY.get(m, m)}" for m in failed) + " (see `results/baselines/status.json`).\n")

    # Efficiency.
    erows = efficiency_rows(R, args.ours)
    if erows:
        gpu = R.eff(args.ours)[3]
        md.append(f"### Efficiency ({gpu})\n\n" + write_table(
            tdir, "efficiency", ["Runtime", "Resolution", "GMACs", "Latency (ms)", "FPS"], erows,
            f"Inference speed of our model, batch 1, on {gpu} (model compute only)."))

    # Ablations.
    for key, (title, arows) in ablation_tables(R).items():
        if any(r[1] != DASH for r in arows):
            md.append(f"### Ablation: {title} (LOL-v1, short schedule)\n\n" + write_table(
                tdir, f"ablation_{key}", ["Variant", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "Params (K)",
                                          "GMACs", "FPS"], arows,
                f"Ablation of {title.lower()} on LOL-v1 (all variants trained with the same "
                f"100-epoch schedule)."))

    macros = write_numbers_tex(R, args.ours, ROOT / "report" / "generated" / "numbers.tex")
    # Tables for the report.
    gen = ROOT / "report" / "generated"
    for f in tdir.glob("*.tex") if tdir.exists() else []:
        (gen / f.name).write_text(f.read_text())

    body = "\n".join(md) if md else ("_No results yet. Run the notebooks; this section is "
                                     "regenerated by `src/collect_results.py`._\n")
    if not args.no_readme:
        readme = Path(args.readme)
        txt = readme.read_text()
        start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if start in txt and end in txt:
            new = txt[: txt.index(start) + len(start)] + "\n" + body + "\n" + txt[txt.index(end):]
            readme.write_text(new)
            print("[collect] README results section updated")
    print(body)
    pend = [k for k, v in macros.items() if "pending" in v]
    print(f"[collect] report macros: {len(macros) - len(pend)} filled, {len(pend)} pending")


if __name__ == "__main__":
    main()

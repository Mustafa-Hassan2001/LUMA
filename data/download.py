"""Download, normalize and verify the low-light enhancement datasets.

Every paired dataset ends up in one standard layout::

    <data_root>/<name>/{train,test}/{low,high}/<stem>.png

and every unpaired (no ground truth) test set in::

    <data_root>/<NAME>/test/low/<file>

Sources are tried in order (Drive cache -> manual file on Drive -> Hugging Face
-> Google Drive via gdown) until one produces a dataset that passes
verification. Folder names on the hosts are NOT hard-coded: after downloading,
the extracted tree is scanned and folders are recognised by name
(``Low/low/input`` -> low, ``High/high/normal/gt/target`` -> high, ``our485/Train``
-> train, ``eval15/Test`` -> test).

Typical Colab usage::

    python data/download.py --datasets all \
        --data_root /content/data \
        --cache_dir /content/drive/MyDrive/lowlight-attn-unet/datasets
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz", ".rar", ".7z")
REPO_ROOT = Path(__file__).resolve().parent.parent

# Folder-name vocabulary used to normalise the many published layouts.
LOW_TOKENS = {"low", "input", "inputs", "lq", "lowlight", "low_light", "dark"}
HIGH_TOKENS = {"high", "normal", "gt", "target", "targets", "groundtruth",
               "ground_truth", "normal_light", "highlight", "high_light", "label"}
TRAIN_TOKENS = {"train", "training", "our485", "trainset", "train_set"}
TEST_TOKENS = {"test", "testing", "eval15", "eval", "val", "valid",
               "validation", "testset", "test_set"}
SKIP_TOKENS = {"__macosx"}


# --------------------------------------------------------------------------- #
# Dataset specifications
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    """One place a dataset can come from."""
    kind: str                       # "hf" | "gdrive_file" | "gdrive_folder" | "url"
    ref: str                        # repo id / Drive id / URL
    label: str                      # human readable
    hf_match: Optional[str] = None  # regex on HF repo paths selecting what to download
    filename: Optional[str] = None  # name to save a single file as

    @property
    def link(self) -> str:
        if self.kind == "hf":
            return f"https://huggingface.co/datasets/{self.ref}"
        if self.kind == "gdrive_file":
            return f"https://drive.google.com/file/d/{self.ref}/view"
        if self.kind == "gdrive_folder":
            return f"https://drive.google.com/drive/folders/{self.ref}"
        return self.ref


@dataclass
class PairedSpec:
    name: str
    expected: dict                              # {"train": n, "test": n}
    classify: Callable[[list], bool]            # path components -> belongs to this dataset?
    sources: list = field(default_factory=list)
    manual_names: tuple = ()                    # filenames accepted in the manual/ folder


def _has(parts, pattern):
    return any(re.search(pattern, p) for p in parts)


def _is_v2(parts):
    return _has(parts, r"v[-_ ]?2|real_captured|synthetic")


def _is_v1(parts):
    return not _is_v2(parts)


def _is_v2_real(parts):
    return _is_v2(parts) and _has(parts, r"real") and not _has(parts, r"synthetic")


def _is_v2_syn(parts):
    return _is_v2(parts) and _has(parts, r"synthetic")


# Sources were checked against the Retinexformer README (Google Drive IDs) on
# 2026-10-09. Hugging Face repos are explored at run time with list_repo_files;
# ``hf_match`` only selects which of the *listed* files to download.
LOLV2_GDRIVE = Source("gdrive_file", "1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U",
                      "Google Drive (Retinexformer README, LOL-v2)", filename="LOL-v2.zip")
DATA4LLIE = "apeanut/data4LLIE"
# Real and Synthetic share one pattern so the files are downloaded only once.
LOLV2_HF_MATCH = r"(?i)lol[-_ ]?v2|real_captured|synthetic"

PAIRED = {
    "lol_v1": PairedSpec(
        name="lol_v1",
        expected={"train": 485, "test": 15},
        classify=_is_v1,
        sources=[
            Source("hf", "geekyrakshit/LoL-Dataset", "Hugging Face geekyrakshit/LoL-Dataset",
                   hf_match=r".*"),
            Source("hf", DATA4LLIE, "Hugging Face apeanut/data4LLIE (LOL-v1 part)",
                   hf_match=r"(?i)lol[-_ ]?(v1|dataset)|our485|eval15|^lol/"),
            Source("gdrive_file", "1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H",
                   "Google Drive (Retinexformer README, LOL-v1)", filename="LOL-v1.zip"),
        ],
        manual_names=("LOL-v1.zip", "LOLdataset.zip", "lol_dataset.zip", "LOL.zip"),
    ),
    "lol_v2_real": PairedSpec(
        name="lol_v2_real",
        expected={"train": 689, "test": 100},
        classify=_is_v2_real,
        sources=[
            Source("hf", DATA4LLIE, "Hugging Face apeanut/data4LLIE (LOL-v2)",
                   hf_match=LOLV2_HF_MATCH),
            LOLV2_GDRIVE,
        ],
        manual_names=("LOL-v2.zip", "LOLv2.zip", "LOL_v2.zip"),
    ),
    "lol_v2_synthetic": PairedSpec(
        name="lol_v2_synthetic",
        expected={"train": 900, "test": 100},
        classify=_is_v2_syn,
        sources=[
            Source("hf", DATA4LLIE, "Hugging Face apeanut/data4LLIE (LOL-v2)",
                   hf_match=LOLV2_HF_MATCH),
            LOLV2_GDRIVE,
        ],
        manual_names=("LOL-v2.zip", "LOLv2.zip", "LOL_v2.zip"),
    ),
}

UNPAIRED_NAMES = ["DICM", "LIME", "MEF", "NPE", "VV"]
UNPAIRED_SOURCES = [
    Source("gdrive_folder", "1RR50EJYGIHaUYwq4NtK7dx8faMSvX8Xp",
           "Google Drive folder (Retinexformer README, 'datasets that have no ground truth')"),
    Source("gdrive_folder", "0B_FjaR958nw_djVQanJqeEhUM1k",
           "Google Drive folder (BIMEF README)"),
]
# Extra direct link listed in the BIMEF README (DICM only).
DICM_URL = Source("url", "http://mcl.korea.ac.kr/projects/LDR/LDR_TEST_IMAGES_DICM.zip",
                  "DICM direct link (BIMEF README)", filename="DICM.zip")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
class QuotaError(RuntimeError):
    pass


def log(msg=""):
    print(msg, flush=True)


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")


def lower_parts(rel: Path) -> list:
    return [p.lower() for p in rel.parts]


def extract_archives(root: Path) -> None:
    """Recursively extract every archive under ``root`` (in place, next to it)."""
    done = set()
    while True:
        todo = [p for p in root.rglob("*")
                if p.is_file() and p.name.lower().endswith(ARCHIVE_EXTS) and p not in done]
        if not todo:
            return
        for arc in todo:
            done.add(arc)
            dest = arc.parent / re.sub(r"(\.tar)?\.(zip|tar|gz|tgz|rar|7z)$", "", arc.name,
                                       flags=re.I)
            log(f"  extracting {arc.name} ...")
            try:
                if arc.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(arc) as z:
                        z.extractall(dest)
                elif arc.name.lower().endswith((".tar", ".tar.gz", ".tgz")):
                    with tarfile.open(arc) as t:
                        try:
                            t.extractall(dest, filter="data")
                        except TypeError:  # Python < 3.12 has no ``filter``
                            t.extractall(dest)
                else:
                    shutil.unpack_archive(str(arc), str(dest))
            except Exception as e:  # corrupt / unsupported archive: report, keep going
                log(f"  ! could not extract {arc.name}: {e}")


def gdown_error_is_quota(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in ("too many users", "quota", "cannot retrieve the public link",
                                  "permission", "access denied", "429"))


def print_quota_help(source: Source, manual_dir: Path, names) -> None:
    bar = "=" * 78
    log(f"\n{bar}\nGOOGLE DRIVE REFUSED THE DOWNLOAD (usually the daily download quota).\n{bar}")
    log(f"Source: {source.label}\nLink:   {source.link}\n")
    log("Fix it manually (takes ~2 minutes, only needed once):")
    log("  1. Open the link above in your browser (logged in to the SAME Google account")
    log("     you mount in Colab).")
    if source.kind == "gdrive_file":
        log("  2. Click the download icon (top-right) and save the zip to your computer,")
        log("     OR right-click the file -> 'Organize' -> 'Add shortcut' won't work for")
        log("     quota, so prefer downloading it.")
        log(f"  3. Upload the zip into this Drive folder:\n       {manual_dir}")
        log(f"     and name it one of: {', '.join(names)}")
    else:
        log("  2. Select all items -> right-click -> 'Download' (Drive zips them for you).")
        log(f"  3. Upload the downloaded zip(s) into this Drive folder:\n       {manual_dir}")
        log(f"     (any name containing {', '.join(names)} or 'unpaired' works).")
    log("  4. Re-run this script. It looks in that folder before trying the internet.")
    log("  Alternatively just wait a few hours: the Drive quota resets daily.")
    log(bar + "\n")


# --------------------------------------------------------------------------- #
# Fetchers: each one puts raw files into ``dest`` and returns nothing.
# --------------------------------------------------------------------------- #
def fetch_hf(source: Source, dest: Path) -> None:
    from huggingface_hub import list_repo_files, snapshot_download

    files = list_repo_files(source.ref, repo_type="dataset")
    log(f"  {len(files)} files in {source.ref}")
    pat = re.compile(source.hf_match or ".*")
    wanted = [f for f in files
              if pat.search(f) and (Path(f).suffix.lower() in IMG_EXTS
                                    or f.lower().endswith(ARCHIVE_EXTS))]
    if not wanted:
        tops = sorted({f.split("/")[0] for f in files})
        raise FileNotFoundError(f"no matching image/archive files; top-level entries: {tops[:30]}")
    log(f"  downloading {len(wanted)} matching files (e.g. {wanted[:3]})")
    snapshot_download(source.ref, repo_type="dataset", local_dir=str(dest),
                      allow_patterns=wanted, max_workers=8)


def fetch_gdrive_file(source: Source, dest: Path) -> None:
    import gdown
    out = dest / (source.filename or f"{source.ref}.zip")
    try:
        res = gdown.download(id=source.ref, output=str(out), quiet=False)
    except Exception as e:
        if gdown_error_is_quota(e):
            raise QuotaError(str(e)) from e
        raise
    if not res or not out.exists() or out.stat().st_size < 1024:
        raise QuotaError("gdown returned no file (likely quota or permission problem)")
    # A quota page is sometimes saved as a tiny HTML "zip".
    if out.suffix == ".zip" and not zipfile.is_zipfile(out):
        raise QuotaError("downloaded file is not a zip (Google Drive returned an HTML page)")


def fetch_gdrive_folder(source: Source, dest: Path) -> None:
    import inspect
    import gdown
    kwargs = dict(id=source.ref, output=str(dest), quiet=False)
    # gdown<6 refuses folders with >50 files unless remaining_ok=True.
    if "remaining_ok" in inspect.signature(gdown.download_folder).parameters:
        kwargs["remaining_ok"] = True
    try:
        res = gdown.download_folder(**kwargs)
    except Exception as e:
        if gdown_error_is_quota(e):
            raise QuotaError(str(e)) from e
        raise
    if not res:
        raise QuotaError("gdown downloaded nothing from the folder")


def fetch_url(source: Source, dest: Path) -> None:
    out = dest / (source.filename or Path(source.ref).name)
    urllib.request.urlretrieve(source.ref, out)


FETCHERS = {"hf": fetch_hf, "gdrive_file": fetch_gdrive_file,
            "gdrive_folder": fetch_gdrive_folder, "url": fetch_url}


# --------------------------------------------------------------------------- #
# Normalisation of paired datasets
# --------------------------------------------------------------------------- #
def classify_image(rel: Path):
    """Return (split, role) for an image path relative to the raw root, or None."""
    parts = lower_parts(rel.parent)
    if any(p in SKIP_TOKENS for p in parts):
        return None
    split = role = None
    for p in parts:
        if p in TRAIN_TOKENS:
            split = "train"
        elif p in TEST_TOKENS:
            split = "test"
        if p in LOW_TOKENS:
            role = "low"
        elif p in HIGH_TOKENS:
            role = "high"
    if split is None or role is None:
        return None
    return split, role


def canonical_stem(stem: str) -> str:
    """'low00001' / 'normal00001' / '00001' all map to '00001'."""
    return re.sub(r"^(low|normal|high|input|gt|target)[-_]?", "", stem, flags=re.I)


def normalize_paired(spec: PairedSpec, raw: Path, out: Path) -> dict:
    """Copy images from an arbitrary raw tree into ``out/{split}/{role}``."""
    found = {}
    for img in sorted(raw.rglob("*")):
        if not img.is_file() or not is_image(img):
            continue
        rel = img.relative_to(raw)
        if not spec.classify(lower_parts(rel.parent)):
            continue
        sr = classify_image(rel)
        if sr is None:
            continue
        key = (*sr, canonical_stem(img.stem))
        if key in found:
            continue  # same image present twice (e.g. zip + extracted copy)
        found[key] = img
    if not found:
        raise FileNotFoundError("no images matching this dataset were found in the download")
    if out.exists():
        shutil.rmtree(out)
    for (split, role, stem), src in found.items():
        d = out / split / role
        d.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() == ".png":
            shutil.copy2(src, d / f"{stem}.png")
        else:  # convert everything to lossless PNG for a uniform layout
            import cv2
            cv2.imwrite(str(d / f"{stem}.png"), cv2.imread(str(src), cv2.IMREAD_UNCHANGED))
    return {"n_files": len(found)}


def verify_paired(spec: PairedSpec, root: Path, check_sizes: bool = True) -> dict:
    """Check pair counts, matching filenames and matching image sizes."""
    report = {"name": spec.name, "ok": True, "splits": {}, "problems": []}
    for split, n_exp in spec.expected.items():
        low_dir, high_dir = root / split / "low", root / split / "high"
        lows = {p.stem: p for p in low_dir.glob("*") if is_image(p)} if low_dir.exists() else {}
        highs = {p.stem: p for p in high_dir.glob("*") if is_image(p)} if high_dir.exists() else {}
        missing_high = sorted(set(lows) - set(highs))
        missing_low = sorted(set(highs) - set(lows))
        pairs = sorted(set(lows) & set(highs))
        sizes, size_mismatch = {}, []
        if check_sizes and pairs:
            from PIL import Image
            for s in pairs:
                with Image.open(lows[s]) as a, Image.open(highs[s]) as b:
                    if a.size != b.size:
                        size_mismatch.append(s)
                    sizes[f"{a.size[0]}x{a.size[1]}"] = sizes.get(f"{a.size[0]}x{a.size[1]}", 0) + 1
        info = {"expected": n_exp, "low": len(lows), "high": len(highs), "pairs": len(pairs),
                "missing_high": missing_high[:10], "missing_low": missing_low[:10],
                "size_mismatch": size_mismatch[:10], "sizes": sizes}
        report["splits"][split] = info
        if len(pairs) != n_exp:
            report["problems"].append(f"{split}: {len(pairs)} pairs, expected {n_exp}")
        if missing_high or missing_low:
            report["problems"].append(
                f"{split}: {len(missing_high)} low without high, {len(missing_low)} high without low")
        if size_mismatch:
            report["problems"].append(f"{split}: {len(size_mismatch)} pairs with different sizes")
    report["ok"] = not report["problems"]
    return report


# --------------------------------------------------------------------------- #
# Unpaired sets
# --------------------------------------------------------------------------- #
def normalize_unpaired(raw: Path, data_root: Path, names) -> dict:
    """Find folders named DICM/LIME/... (or zips that were extracted to them)."""
    counts = {}
    for name in names:
        imgs = {}
        for img in sorted(raw.rglob("*")):
            if not img.is_file() or not is_image(img):
                continue
            parts = lower_parts(img.relative_to(raw).parent)
            if any(p in SKIP_TOKENS for p in parts):
                continue
            # Skip method outputs / reports shipped next to the data in the BIMEF folder.
            if any("__" in p or "@" in p or p in {"out", "results", "result"} for p in parts):
                continue
            if not any(re.fullmatch(rf"{name.lower()}([-_ ]?(data|dataset|images?))?", p)
                       or p == f"ldr_test_images_{name.lower()}" for p in parts):
                continue
            imgs.setdefault(img.name, img)
        if not imgs:
            continue
        dst = data_root / name / "test" / "low"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for fname, src in imgs.items():
            shutil.copy2(src, dst / fname)
        counts[name] = len(imgs)
    return counts


# --------------------------------------------------------------------------- #
# Drive cache (zip of the *normalised* dataset)
# --------------------------------------------------------------------------- #
def cache_zip_path(cache_dir: Optional[Path], name: str) -> Optional[Path]:
    return cache_dir / f"{name}.zip" if cache_dir else None


def save_cache(cache_dir: Optional[Path], name: str, folder: Path) -> None:
    if not cache_dir:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_zip_path(cache_dir, name)
    tmp = target.with_suffix(".zip.part")
    log(f"  caching {name} -> {target}")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:  # PNGs are already compressed
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder.parent))
    tmp.replace(target)  # atomic: a half-written cache never looks valid


def load_cache(cache_dir: Optional[Path], name: str, data_root: Path) -> bool:
    z = cache_zip_path(cache_dir, name)
    if not z or not z.exists():
        return False
    log(f"  found cached copy {z}; unzipping to {data_root}")
    if (data_root / name).exists():
        shutil.rmtree(data_root / name)
    try:
        with zipfile.ZipFile(z) as f:
            f.extractall(data_root)
        return True
    except zipfile.BadZipFile:
        log("  ! cache zip is corrupt; ignoring it")
        return False


def manual_files(manual_dir: Optional[Path], names) -> list:
    if not manual_dir or not manual_dir.exists():
        return []
    lower = [n.lower() for n in names]
    return [p for p in manual_dir.iterdir()
            if p.is_file() and any(n.lower().replace(".zip", "") in p.name.lower() for n in lower)]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def prepare_paired(spec: PairedSpec, data_root: Path, cache_dir, manual_dir, work: Path,
                   force: bool, raw_cache: dict) -> dict:
    log(f"\n### {spec.name}")
    out = data_root / spec.name
    if not force:
        if out.exists() and verify_paired(spec, out, check_sizes=False)["ok"]:
            log("  already present in data_root")
            return verify_paired(spec, out)
        if load_cache(cache_dir, spec.name, data_root):
            rep = verify_paired(spec, out)
            if rep["ok"]:
                rep["source"] = "drive cache"
                return rep
            log(f"  ! cached copy failed verification: {rep['problems']}; re-downloading")

    attempts = []
    manual = manual_files(manual_dir, spec.manual_names)
    candidates = ([("manual", Source("manual", str(m), f"manual file {m.name}")) for m in manual]
                  + [("net", s) for s in spec.sources])
    quota_hits = []
    for how, src in candidates:
        log(f"  -> trying {src.label}")
        key = (src.kind, src.ref, src.hf_match)
        raw = raw_cache.get(key)
        try:
            if raw is None:
                raw = work / f"raw_{len(raw_cache)}"
                raw.mkdir(parents=True, exist_ok=True)
                if how == "manual":
                    shutil.copy2(src.ref, raw / Path(src.ref).name)
                else:
                    FETCHERS[src.kind](src, raw)
                extract_archives(raw)
                raw_cache[key] = raw
            normalize_paired(spec, raw, out)
            rep = verify_paired(spec, out)
            if rep["ok"]:
                rep["source"] = src.label
                save_cache(cache_dir, spec.name, out)
                return rep
            attempts.append(f"{src.label}: verification failed {rep['problems']}")
            log(f"  ! verification failed: {rep['problems']}")
        except QuotaError as e:
            quota_hits.append(src)
            attempts.append(f"{src.label}: Google Drive quota/permission ({e})")
            log(f"  ! Google Drive refused: {e}")
        except Exception as e:
            attempts.append(f"{src.label}: {type(e).__name__}: {e}")
            log(f"  ! failed: {type(e).__name__}: {e}")
            if "--debug" in sys.argv:
                traceback.print_exc()
    for src in quota_hits:
        print_quota_help(src, manual_dir, spec.manual_names)
    return {"name": spec.name, "ok": False, "problems": attempts, "splits": {}}


def prepare_unpaired(data_root: Path, cache_dir, manual_dir, work: Path, force: bool) -> dict:
    log("\n### unpaired (DICM, LIME, MEF, NPE, VV)")
    counts = {}
    if not force:
        for n in UNPAIRED_NAMES:
            d = data_root / n / "test" / "low"
            if d.exists() and any(is_image(p) for p in d.iterdir()):
                counts[n] = sum(1 for p in d.iterdir() if is_image(p))
            elif load_cache(cache_dir, n, data_root):
                counts[n] = sum(1 for p in d.iterdir() if is_image(p))
    missing = [n for n in UNPAIRED_NAMES if n not in counts]
    sources_used = {n: "cache/present" for n in counts}
    if missing:
        manual = manual_files(manual_dir, UNPAIRED_NAMES + ["unpaired"])
        candidates = ([Source("manual", str(m), f"manual file {m.name}") for m in manual]
                      + UNPAIRED_SOURCES + [DICM_URL])
        quota_hits = []
        for src in candidates:
            if not missing:
                break
            if src is DICM_URL and "DICM" not in missing:
                continue
            log(f"  -> trying {src.label} for {missing}")
            raw = work / f"raw_unpaired_{abs(hash(src.ref))}"
            raw.mkdir(parents=True, exist_ok=True)
            try:
                if src.kind == "manual":
                    shutil.copy2(src.ref, raw / Path(src.ref).name)
                else:
                    FETCHERS[src.kind](src, raw)
                extract_archives(raw)
                got = normalize_unpaired(raw, data_root, missing)
                for n, c in got.items():
                    counts[n] = c
                    sources_used[n] = src.label
                    save_cache(cache_dir, n, data_root / n)
                    # gdown<6 silently stops at 50 files per folder.
                    if c == 50:
                        log(f"  ! {n} has exactly 50 images: the Drive folder listing may "
                            f"have been truncated by gdown")
                missing = [n for n in missing if n not in counts]
            except QuotaError as e:
                quota_hits.append(src)
                log(f"  ! Google Drive refused: {e}")
            except Exception as e:
                log(f"  ! failed: {type(e).__name__}: {e}")
        if missing:
            for src in quota_hits:
                print_quota_help(src, manual_dir, missing)
    return {"name": "unpaired", "ok": not missing, "counts": counts,
            "missing": missing, "sources": sources_used}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def save_preview(spec_name: str, root: Path, out_png: Path, n: int = 6, seed: int = 0) -> None:
    """Grid of low (top) / high (bottom) test pairs, or a single row for unpaired sets."""
    from PIL import Image

    rng = random.Random(seed)
    low_dir, high_dir = root / "test" / "low", root / "test" / "high"
    stems = sorted(p.name for p in low_dir.iterdir() if is_image(p))
    if not stems:
        return
    pick = rng.sample(stems, min(n, len(stems)))
    tile_w, tile_h = 240, 160
    rows = [low_dir] + ([high_dir] if high_dir.exists() else [])
    canvas = Image.new("RGB", (tile_w * len(pick), tile_h * len(rows)), (0, 0, 0))
    for r, d in enumerate(rows):
        for c, name in enumerate(pick):
            cand = d / name
            if not cand.exists():
                cand = next(iter(d.glob(Path(name).stem + ".*")), None)
            if cand is None:
                continue
            with Image.open(cand) as im:
                im = im.convert("RGB")
                im.thumbnail((tile_w, tile_h))
                canvas.paste(im, (c * tile_w + (tile_w - im.width) // 2,
                                  r * tile_h + (tile_h - im.height) // 2))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def pixel_hashes(folder: Path) -> dict:
    """md5 of decoded pixels, so re-encoded copies of the same image still match."""
    import hashlib
    from PIL import Image
    out = {}
    for p in sorted(folder.iterdir()) if folder.exists() else []:
        if is_image(p):
            with Image.open(p) as im:
                out[hashlib.md5(im.convert("RGB").tobytes()).hexdigest()] = p.name
    return out


def check_overlap(data_root: Path) -> Optional[dict]:
    """How many LOL-v1 images also appear in LOL-v2-Real (a known source of leakage)."""
    v1, v2 = data_root / "lol_v1", data_root / "lol_v2_real"
    if not (v1.exists() and v2.exists()):
        return None
    res = {}
    for a in ("train", "test"):
        ha = pixel_hashes(v1 / a / "low")
        for b in ("train", "test"):
            hb = pixel_hashes(v2 / b / "low")
            res[f"lol_v1/{a} in lol_v2_real/{b}"] = len(set(ha) & set(hb))
    return res


def print_summary(paired_reports, unpaired_report) -> None:
    log("\n" + "=" * 78)
    log(f"{'dataset':<20}{'split':<7}{'expected':>9}{'pairs':>8}{'low':>7}{'high':>7}"
        f"  {'sizes':<18}{'status'}")
    log("-" * 78)
    for rep in paired_reports:
        if not rep["splits"]:
            log(f"{rep['name']:<20}{'-':<7}{'':>9}{'':>8}{'':>7}{'':>7}  {'':<18}FAILED")
            continue
        for split, s in rep["splits"].items():
            sizes = ",".join(f"{k}({v})" for k, v in s["sizes"].items())
            status = "OK" if s["pairs"] == s["expected"] and not s["missing_high"] \
                and not s["missing_low"] and not s["size_mismatch"] else "MISMATCH"
            log(f"{rep['name']:<20}{split:<7}{s['expected']:>9}{s['pairs']:>8}{s['low']:>7}"
                f"{s['high']:>7}  {sizes[:18]:<18}{status}")
    for n in UNPAIRED_NAMES if unpaired_report else []:
        c = unpaired_report["counts"].get(n)
        log(f"{n:<20}{'test':<7}{'n/a':>9}{'-':>8}{(c if c is not None else '-'):>7}{'-':>7}"
            f"  {'':<18}{'OK (no GT)' if c else 'MISSING'}")
    log("=" * 78)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["all"],
                    help="any of: lol_v1 lol_v2_real lol_v2_synthetic unpaired all")
    ap.add_argument("--data_root", default="/content/data",
                    help="where the normalised datasets go (fast local disk)")
    ap.add_argument("--cache_dir", default=None,
                    help="folder (e.g. on Google Drive) holding <name>.zip caches")
    ap.add_argument("--manual_dir", default=None,
                    help="folder where you can drop manually downloaded zips "
                         "(default: <cache_dir>/manual)")
    ap.add_argument("--results_dir", default=str(REPO_ROOT / "results" / "data"),
                    help="where the summary JSON and preview grids are written")
    ap.add_argument("--force", action="store_true", help="ignore caches and re-download")
    ap.add_argument("--keep_raw", action="store_true", help="keep the raw download folder")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    wanted = set(args.datasets)
    if "all" in wanted:
        wanted = set(PAIRED) | {"unpaired"}
    unknown = wanted - set(PAIRED) - {"unpaired"}
    if unknown:
        ap.error(f"unknown dataset(s): {sorted(unknown)}")

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    manual_dir = Path(args.manual_dir) if args.manual_dir else \
        (cache_dir / "manual" if cache_dir else None)
    if manual_dir:
        manual_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    work = Path(tempfile.mkdtemp(prefix="llie_raw_", dir=str(data_root)))
    raw_cache: dict = {}
    t0 = time.time()
    paired_reports, unpaired_report = [], None
    try:
        for name in [n for n in PAIRED if n in wanted]:
            paired_reports.append(prepare_paired(PAIRED[name], data_root, cache_dir, manual_dir,
                                                 work, args.force, raw_cache))
        if "unpaired" in wanted:
            unpaired_report = prepare_unpaired(data_root, cache_dir, manual_dir, work, args.force)
    finally:
        if not args.keep_raw:
            shutil.rmtree(work, ignore_errors=True)

    for rep in paired_reports:
        if rep["ok"]:
            save_preview(rep["name"], data_root / rep["name"],
                         results_dir / f"preview_{rep['name']}.png")
    if unpaired_report:
        for n in unpaired_report["counts"]:
            save_preview(n, data_root / n, results_dir / f"preview_{n}.png")

    print_summary(paired_reports, unpaired_report)
    overlap = check_overlap(data_root)
    if overlap:
        log("Identical low-light images shared between LOL-v1 and LOL-v2-Real "
            "(pixel hash):")
        for k, v in overlap.items():
            log(f"  {k:<40}{v:>5}")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {"data_root": str(data_root), "elapsed_s": round(time.time() - t0, 1),
               "paired": paired_reports, "unpaired": unpaired_report,
               "lol_v1_v2_overlap": overlap}
    with open(results_dir / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"summary -> {results_dir / 'data_summary.json'}; previews -> {results_dir}")

    failed = [r["name"] for r in paired_reports if not r["ok"]]
    if unpaired_report and not unpaired_report["ok"]:
        failed.append("unpaired: " + ",".join(unpaired_report["missing"]))
    if failed:
        log(f"\nNOT READY: {failed}. Read the messages above for what went wrong.")
        for r in paired_reports:
            if not r["ok"]:
                for p in r["problems"]:
                    log(f"  - {r['name']}: {p}")
        return 1
    log("\nAll requested datasets are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

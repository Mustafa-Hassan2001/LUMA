"""Offline tests for data/download.py.

They build tiny fake copies of each published folder layout, so the
normalisation / verification / caching logic is exercised without internet.
"""
import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("download", ROOT / "data" / "download.py")
dl = importlib.util.module_from_spec(spec)
sys.modules["download"] = dl
spec.loader.exec_module(dl)


def _img(path: Path, size=(6, 4), val=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), val, np.uint8)).save(path)


def _zip_dir(src: Path, out: Path):
    with zipfile.ZipFile(out, "w") as z:
        for f in src.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(src))


def make_lol_v1_drive(tmp: Path) -> Path:
    """LOLdataset/our485/{low,high}, LOLdataset/eval15/{low,high} (original release)."""
    tree = tmp / "tree_v1"
    for split, n, off in (("our485", 485, 100), ("eval15", 15, 1)):
        for i in range(n):
            for role in ("low", "high"):
                _img(tree / "LOLdataset" / split / role / f"{i + off}.png")
    _img(tree / "__MACOSX" / "LOLdataset" / "our485" / "low" / "junk.png")
    out = tmp / "LOL-v1.zip"
    _zip_dir(tree, out)
    return out


def make_lol_v2_drive(tmp: Path) -> Path:
    """LOL-v2/Real_captured/{Train,Test}/{Low,Normal} with low*/normal* prefixes + Synthetic."""
    tree = tmp / "tree_v2"
    for split, n, off in (("Train", 689, 1), ("Test", 100, 690)):
        for i in range(n):
            _img(tree / "LOL-v2" / "Real_captured" / split / "Low" / f"low{i + off:05d}.png")
            _img(tree / "LOL-v2" / "Real_captured" / split / "Normal" / f"normal{i + off:05d}.png")
    for split, n in (("Train", 900), ("Test", 100)):
        for i in range(n):
            for role in ("Low", "Normal"):
                _img(tree / "LOL-v2" / "Synthetic" / split / role / f"r{split}{i:05d}t.png",
                     size=(4, 4))
    out = tmp / "LOL-v2.zip"
    _zip_dir(tree, out)
    return out


def test_classify_image_vocabulary():
    c = dl.classify_image
    assert c(Path("LOLdataset/our485/low/1.png")) == ("train", "low")
    assert c(Path("LOLdataset/eval15/high/1.png")) == ("test", "high")
    assert c(Path("LOLv1/Train/input/1.png")) == ("train", "low")
    assert c(Path("LOLv1/Test/target/1.png")) == ("test", "high")
    assert c(Path("LOL-v2/Real_captured/Test/Normal/x.png")) == ("test", "high")
    assert c(Path("LOL-v2/Synthetic/Train/GT/x.png")) == ("train", "high")
    assert c(Path("__MACOSX/our485/low/1.png")) is None
    assert c(Path("some/other/1.png")) is None


def test_dataset_membership():
    parts = lambda s: [p.lower() for p in Path(s).parts]
    assert dl._is_v1(parts("LOLdataset/our485/low"))
    assert not dl._is_v1(parts("LOL-v2/Real_captured/Train/Low"))
    assert dl._is_v2_real(parts("LOL-v2/Real_captured/Train/Low"))
    assert dl._is_v2_real(parts("LOLv2/Real_captured/Test/Normal"))
    assert not dl._is_v2_real(parts("LOLv2/Synthetic/Test/Normal"))
    assert dl._is_v2_syn(parts("LOL_v2/Synthetic/Train/Low"))


def test_canonical_stem():
    assert dl.canonical_stem("low00001") == "00001"
    assert dl.canonical_stem("normal00690") == "00690"
    assert dl.canonical_stem("r000da54ft") == "r000da54ft"
    assert dl.canonical_stem("146") == "146"


def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network fetcher must not be called in this test")
    monkeypatch.setattr(dl, "FETCHERS", {k: boom for k in dl.FETCHERS})


def test_lol_v1_from_manual_zip_then_cache(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    cache, manual, data = tmp_path / "cache", tmp_path / "cache" / "manual", tmp_path / "data"
    manual.mkdir(parents=True)
    make_lol_v1_drive(tmp_path).rename(manual / "LOL-v1.zip")
    work = tmp_path / "work"
    rep = dl.prepare_paired(dl.PAIRED["lol_v1"], data, cache, manual, work, False, {})
    assert rep["ok"], rep["problems"]
    assert rep["splits"]["train"]["pairs"] == 485 and rep["splits"]["test"]["pairs"] == 15
    assert (data / "lol_v1" / "test" / "low" / "1.png").exists()
    assert (cache / "lol_v1.zip").exists()

    # Second run on a fresh machine: must come from the Drive cache, not the manual zip.
    (manual / "LOL-v1.zip").unlink()
    data2 = tmp_path / "data2"
    rep2 = dl.prepare_paired(dl.PAIRED["lol_v1"], data2, cache, manual, work, False, {})
    assert rep2["ok"] and rep2["source"] == "drive cache"


def test_lol_v2_real_and_synthetic_from_one_zip(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    manual = tmp_path / "manual"
    manual.mkdir()
    make_lol_v2_drive(tmp_path).rename(manual / "LOL-v2.zip")
    data, work, raw_cache = tmp_path / "data", tmp_path / "work", {}
    real = dl.prepare_paired(dl.PAIRED["lol_v2_real"], data, None, manual, work, False, raw_cache)
    syn = dl.prepare_paired(dl.PAIRED["lol_v2_synthetic"], data, None, manual, work, False,
                            raw_cache)
    assert real["ok"], real["problems"]
    assert syn["ok"], syn["problems"]
    assert (data / "lol_v2_real" / "test" / "high" / "00690.png").exists()
    assert syn["splits"]["train"]["sizes"] == {"4x4": 900}


def test_alternative_layout_from_hf(tmp_path, monkeypatch):
    """A Retinexformer-style 'LOLv1/Train/input|target' tree delivered by the HF fetcher."""
    def fake_hf(source, dest):
        for split, n in (("Train", 485), ("Test", 15)):
            for i in range(n):
                _img(dest / "LOLv1" / split / "input" / f"{i}.png")
                _img(dest / "LOLv1" / split / "target" / f"{i}.png")
    monkeypatch.setattr(dl, "FETCHERS", {**dl.FETCHERS, "hf": fake_hf})
    rep = dl.prepare_paired(dl.PAIRED["lol_v1"], tmp_path / "d", None, None, tmp_path / "w",
                            False, {})
    assert rep["ok"], rep["problems"]
    assert rep["source"].startswith("Hugging Face")


def test_verification_catches_bad_pairs(tmp_path):
    root = tmp_path / "lol_v1"
    for i in range(485):
        _img(root / "train" / "low" / f"{i}.png")
        _img(root / "train" / "high" / f"{i}.png")
    for i in range(15):
        _img(root / "test" / "low" / f"{i}.png")
    for i in range(14):
        _img(root / "test" / "high" / f"{i}.png", size=(5, 4) if i == 0 else (6, 4))
    rep = dl.verify_paired(dl.PAIRED["lol_v1"], root)
    assert not rep["ok"]
    assert rep["splits"]["test"]["missing_high"] == ["14"]
    assert rep["splits"]["test"]["size_mismatch"] == ["0"]


def test_quota_error_falls_back_and_explains(tmp_path, monkeypatch, capsys):
    def quota(source, dest):
        raise dl.QuotaError("Too many users have viewed or downloaded this file recently")
    def hf_fail(source, dest):
        raise ConnectionError("hf down")
    monkeypatch.setattr(dl, "FETCHERS", {"hf": hf_fail, "gdrive_file": quota,
                                         "gdrive_folder": quota, "url": quota})
    rep = dl.prepare_paired(dl.PAIRED["lol_v1"], tmp_path / "d", tmp_path / "c",
                            tmp_path / "c" / "manual", tmp_path / "w", False, {})
    out = capsys.readouterr().out
    assert not rep["ok"]
    assert "GOOGLE DRIVE REFUSED THE DOWNLOAD" in out
    assert "1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H" in out
    assert len(rep["problems"]) == 3  # two HF sources + Drive were all tried


def test_unpaired_normalisation(tmp_path):
    raw = tmp_path / "raw"
    for i in range(3):
        _img(raw / "DICM" / f"{i:02d}.jpg")
        _img(raw / "VV" / f"P{i}.jpg")
    _img(raw / "LIME_zip" / "LIME" / "1.bmp")
    _img(raw / "NPE-ex1" / "x.jpg")                 # extra set: ignored
    _img(raw / "results__dong@VV" / "P0__dong.png")  # method outputs: ignored
    _img(raw / "MEF" / "out" / "loe.png")            # metric outputs: ignored
    _img(raw / "MEF" / "Balloons.png")
    got = dl.normalize_unpaired(raw, tmp_path / "data", dl.UNPAIRED_NAMES)
    assert got == {"DICM": 3, "LIME": 1, "MEF": 1, "VV": 3}
    assert (tmp_path / "data" / "VV" / "test" / "low" / "P0.jpg").exists()


def test_main_writes_summary_and_preview(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    cache = tmp_path / "cache"
    (cache / "manual").mkdir(parents=True)
    make_lol_v1_drive(tmp_path).rename(cache / "manual" / "LOL-v1.zip")
    res = tmp_path / "results"
    code = dl.main(["--datasets", "lol_v1", "--data_root", str(tmp_path / "data"),
                    "--cache_dir", str(cache), "--results_dir", str(res)])
    assert code == 0
    assert (res / "data_summary.json").exists()
    assert (res / "preview_lol_v1.png").exists()


def test_overlap_check(tmp_path):
    for i in range(3):
        _img(tmp_path / "lol_v1" / "test" / "low" / f"{i}.png", val=i)
        _img(tmp_path / "lol_v1" / "train" / "low" / f"{i}.png", val=50 + i)
    _img(tmp_path / "lol_v2_real" / "train" / "low" / "00001.png", val=1)  # same pixels as v1 test 1
    _img(tmp_path / "lol_v2_real" / "test" / "low" / "00700.png", val=99)
    res = dl.check_overlap(tmp_path)
    assert res["lol_v1/test in lol_v2_real/train"] == 1
    assert res["lol_v1/train in lol_v2_real/test"] == 0

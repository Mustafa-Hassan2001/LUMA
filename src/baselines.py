"""Baseline methods run with pretrained weights, through the SAME metric code as ours.

| key           | weights used                                                        |
|---------------|---------------------------------------------------------------------|
| zero_dce      | official (Li-Chongyi/Zero-DCE, Epoch99.pth); architecture copied here |
| zero_dce_pp   | official (Li-Chongyi/Zero-DCE_extension, Epoch99.pth), scale_factor=1 |
| enlightengan  | official weights converted to ONNX (arsenyinfo/EnlightenGAN-inference)|
| retinexnet    | UNOFFICIAL PyTorch port + its checkpoint (aasharma90/RetinexNet_PyTorch); |
|               | the official release is TensorFlow 1                                |
| retinexformer | official code + per-dataset weights (caiyuanhao1998/Retinexformer)  |

Each ``load_<key>(dataset, cache, device)`` returns ``(fn, module_or_None, info)``
where ``fn`` maps a (1,3,H,W) tensor in [0,1] to the enhanced (1,3,H,W) tensor.
Anything that cannot be downloaded or loaded raises; ``run_baselines.py`` records
the failure and falls back to *reported* numbers (clearly marked).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ZERO_DCE_URL = ("https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/"
                "Zero-DCE_code/snapshots/Epoch99.pth")
ZERO_DCE_PP_URL = ("https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/"
                   "Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth")
RETINEXNET_REPO = "https://github.com/aasharma90/RetinexNet_PyTorch.git"
RETINEXFORMER_REPO = "https://github.com/caiyuanhao1998/Retinexformer.git"
RETINEXFORMER_WEIGHTS_FOLDER = "1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV"  # README "pretrained_weights"
ENLIGHTENGAN_PIP = "git+https://github.com/arsenyinfo/EnlightenGAN-inference"
RETINEXFORMER_WEIGHT_FILE = {"lol_v1": "LOL_v1.pth", "lol_v2_real": "LOL_v2_real.pth",
                             "lol_v2_synthetic": "LOL_v2_synthetic.pth"}


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[baseline] downloading {url}")
        urllib.request.urlretrieve(url, dest.with_suffix(".part"))
        dest.with_suffix(".part").rename(dest)
    return dest


def _clone(url: str, dest: Path) -> Path:
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "-q", url, str(dest)], check=True)
    return dest


def _import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _strip_module(sd: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def _pad_to(x, factor, mode="reflect"):
    h, w = x.shape[-2:]
    ph, pw = (-h) % factor, (-w) % factor
    return (F.pad(x, (0, pw, 0, ph), mode=mode) if (ph or pw) else x), h, w


# ----------------------------------------------------------------------------- Zero-DCE
class ZeroDCE(nn.Module):
    """enhance_net_nopool from Zero-DCE_code/model.py (same layer names -> same weights)."""

    def __init__(self):
        super().__init__()
        nf = 32
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        for r in torch.split(x_r, 3, dim=1):
            x = x + r * (torch.pow(x, 2) - x)
        return x


class CSDN_Tem(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.depth_conv = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
        self.point_conv = nn.Conv2d(cin, cout, 1, 1, 0, groups=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ZeroDCEpp(nn.Module):
    """enhance_net_nopool from Zero-DCE++/model.py. scale_factor=1 = full resolution."""

    def __init__(self, scale_factor: int = 1):
        super().__init__()
        nf = 32
        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = CSDN_Tem(3, nf)
        self.e_conv2 = CSDN_Tem(nf, nf)
        self.e_conv3 = CSDN_Tem(nf, nf)
        self.e_conv4 = CSDN_Tem(nf, nf)
        self.e_conv5 = CSDN_Tem(nf * 2, nf)
        self.e_conv6 = CSDN_Tem(nf * 2, nf)
        self.e_conv7 = CSDN_Tem(nf * 2, 3)

    def forward(self, x):
        xd = x if self.scale_factor == 1 else F.interpolate(
            x, scale_factor=1 / self.scale_factor, mode="bilinear")
        x1 = self.relu(self.e_conv1(xd))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor != 1:
            x_r = F.interpolate(x_r, size=x.shape[-2:], mode="bilinear", align_corners=True)
        for _ in range(8):
            x = x + x_r * (torch.pow(x, 2) - x)
        return x


def _module_fn(m: nn.Module, device, pad_factor: int = 1):
    m = m.to(device).eval()

    @torch.no_grad()
    def fn(x):
        x = x.to(device)
        xp, h, w = _pad_to(x, pad_factor) if pad_factor > 1 else (x, *x.shape[-2:])
        return m(xp)[..., :h, :w].float().clamp(0, 1)
    return fn


def load_zero_dce(dataset, cache: Path, device):
    m = ZeroDCE()
    sd = torch.load(_download(ZERO_DCE_URL, cache / "zero_dce" / "Epoch99.pth"), map_location="cpu")
    m.load_state_dict(_strip_module(sd), strict=True)
    return _module_fn(m, device), m, {"weights": "official", "source": ZERO_DCE_URL,
                                      "trained_on": "SICE part1 (unpaired, no LOL)"}


def load_zero_dce_pp(dataset, cache: Path, device):
    m = ZeroDCEpp(scale_factor=1)
    sd = torch.load(_download(ZERO_DCE_PP_URL, cache / "zero_dce_pp" / "Epoch99.pth"),
                    map_location="cpu")
    m.load_state_dict(_strip_module(sd), strict=True)
    return _module_fn(m, device), m, {
        "weights": "official", "source": ZERO_DCE_PP_URL, "trained_on": "SICE part1 (no LOL)",
        "notes": "run at full resolution (scale_factor=1); the official test script uses 12"}


# ----------------------------------------------------------------------------- EnlightenGAN
def load_enlightengan(dataset, cache: Path, device):
    try:
        from enlighten_inference import EnlightenOnnxModel
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", ENLIGHTENGAN_PIP], check=True)
        from enlighten_inference import EnlightenOnnxModel
    import onnxruntime as ort
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    model = EnlightenOnnxModel(providers=providers)

    def fn(x):
        rgb = (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        bgr = np.ascontiguousarray(rgb[..., ::-1])       # the wrapper expects cv2 (BGR) images
        out = model.predict(bgr)[..., ::-1].astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1)[None]

    n_params = None
    try:
        import onnx
        from enlighten_inference import __file__ as ef
        g = onnx.load(str(Path(ef).parent / "enlighten.onnx")).graph
        n_params = int(sum(np.prod(t.dims) for t in g.initializer))
    except Exception:
        pass
    return fn, None, {"weights": "official (converted to ONNX by arsenyinfo)",
                      "source": ENLIGHTENGAN_PIP, "trained_on": "unpaired EnlightenGAN data",
                      "params": n_params, "provider": providers[0],
                      "notes": "ONNX Runtime; includes uint8 round-trip and CPU<->GPU copies"}


# ----------------------------------------------------------------------------- RetinexNet
class _RetinexNetWrap(nn.Module):
    def __init__(self, decom, relight):
        super().__init__()
        self.decom, self.relight = decom, relight

    def forward(self, x):
        r, i = self.decom(x)
        d = self.relight(i, r)
        return r * torch.cat([d, d, d], 1)


def load_retinexnet(dataset, cache: Path, device):
    repo = _clone(RETINEXNET_REPO, cache / "src" / "RetinexNet_PyTorch")
    mod = _import_file("retinexnet_port_model", repo / "model.py")
    decom, relight = mod.DecomNet(), mod.RelightNet()

    def latest(d):
        files = sorted((repo / "ckpts" / d).glob("*.tar"), key=lambda p: (len(p.name), p.name))
        if not files:
            raise FileNotFoundError(f"no checkpoint in {repo / 'ckpts' / d}")
        return files[-1]
    decom.load_state_dict(torch.load(latest("Decom"), map_location="cpu"))
    relight.load_state_dict(torch.load(latest("Relight"), map_location="cpu"))
    m = _RetinexNetWrap(decom, relight)
    return _module_fn(m, device), m, {
        "weights": "UNOFFICIAL PyTorch port checkpoint", "source": RETINEXNET_REPO,
        "trained_on": "LOL our485 + synthetic pairs (per the port's README)",
        "notes": "official RetinexNet is TensorFlow 1; numbers may differ from the paper"}


# ----------------------------------------------------------------------------- Retinexformer
def _retinexformer_weights(dataset: str, cache: Path) -> Path:
    import gdown
    fname = RETINEXFORMER_WEIGHT_FILE.get(dataset, "LOL_v1.pth")
    dest = cache / "retinexformer" / fname
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:  # list the folder and fetch just the file we need
        files = gdown.download_folder(id=RETINEXFORMER_WEIGHTS_FOLDER, skip_download=True,
                                      quiet=True)
        match = [f for f in files if Path(f.path).name == fname]
        if not match:
            raise FileNotFoundError(f"{fname} not in folder: {[Path(f.path).name for f in files]}")
        gdown.download(id=match[0].id, output=str(dest), quiet=False)
    except TypeError:  # old gdown without skip_download
        tmp = cache / "retinexformer" / "_folder"
        gdown.download_folder(id=RETINEXFORMER_WEIGHTS_FOLDER, output=str(tmp), quiet=False)
        found = next(tmp.rglob(fname))
        found.rename(dest)
    if not dest.exists():
        raise FileNotFoundError(
            f"could not download {fname}; download it manually from "
            f"https://drive.google.com/drive/folders/{RETINEXFORMER_WEIGHTS_FOLDER} into {dest.parent}")
    return dest


def load_retinexformer(dataset, cache: Path, device):
    try:
        import einops  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops"], check=True)
    repo = _clone(RETINEXFORMER_REPO, cache / "src" / "Retinexformer")
    mod = _import_file("retinexformer_arch", repo / "basicsr" / "models" / "archs" /
                       "RetinexFormer_arch.py")
    # Options/RetinexFormer_LOL_v1.yml (same for v2 real/synthetic)
    m = mod.RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    wpath = _retinexformer_weights(dataset, cache)
    ck = torch.load(wpath, map_location="cpu", weights_only=False)
    m.load_state_dict(_strip_module(ck.get("params", ck)), strict=True)
    return _module_fn(m, device, pad_factor=4), m, {
        "weights": f"official ({wpath.name})", "source": RETINEXFORMER_REPO,
        "trained_on": RETINEXFORMER_WEIGHT_FILE.get(dataset, "lol_v1 (LOL_v1.pth)"),
        "notes": "reflect padding to a multiple of 4, as in the official test script"}


LOADERS = {"zero_dce": load_zero_dce, "zero_dce_pp": load_zero_dce_pp,
           "retinexnet": load_retinexnet, "enlightengan": load_enlightengan,
           "retinexformer": load_retinexformer}
PER_DATASET_WEIGHTS = {"retinexformer"}

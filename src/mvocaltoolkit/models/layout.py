"""Detection of model files inside a downloaded/extracted folder.

Model authors pack their models differently (a zip with a folder inside, a folder with renamed files,
a .ckpt next to a dictionary, safetensors with yaml configs...). These functions find what each engine needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CHECKPOINT_EXT = (".ckpt", ".pt", ".pth", ".safetensors")


def _files(root: Path, *patterns: str) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found += [p for p in root.glob(pattern) if p.is_file()]
    return sorted(set(found), key=lambda p: (len(p.relative_to(root).parts), p.name))


def _first(root: Path, *names: str) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def detect_sofa(root: Path) -> dict[str, Any] | None:
    """A SOFA model folder: checkpoint + dictionary (+ optional OpenUtau G2P model in g2p/)."""
    ckpt = _first(root, "model.ckpt", "model.safetensors") or next(
        iter(_files(root, "*.ckpt", "*.safetensors")), None
    )
    if ckpt is None:
        return None
    dictionary = _first(root, "dict.txt", "dictionary.txt") or next(
        iter([p for p in _files(root, "*.txt") if p.name.lower() not in ("readme.txt", "license.txt")]), None
    )
    layout: dict[str, Any] = {"checkpoint": ckpt.name, "format": "safetensors" if ckpt.suffix == ".safetensors" else "ckpt"}
    if dictionary is not None:
        layout["dictionary"] = dictionary.name
    for key, name in (("vocab", "vocab.yaml"), ("global_config", "global_config.yaml"), ("train_config", "train_config.yaml")):
        if (root / name).exists():
            layout[key] = name
    if layout["format"] == "safetensors" and "train_config" not in layout:
        configs = [p for p in _files(root, "*.yaml") if p.name not in ("vocab.yaml", "global_config.yaml")]
        if configs:
            layout["train_config"] = configs[0].name
    g2p = root / "g2p"
    if g2p.is_dir() and (g2p / "cfg.yaml").exists():
        weights = _first(g2p, "model.ptsd", "g2p-best.ptsd") or next(iter(_files(g2p, "*.ptsd", "*.pt")), None)
        if weights is not None:
            layout["g2p"] = {"config": "g2p/cfg.yaml", "weights": f"g2p/{weights.name}"}
    readme = _first(root, "readme.txt", "README.txt", "README.md")
    if readme is not None:
        layout["readme"] = readme.name
    return layout


def detect_wfl_asr(root: Path) -> dict[str, Any] | None:
    ckpt = next(iter(_files(root, "*.ckpt", "*.pt", "*.pth", "*.safetensors", "*/*.ckpt", "*/*.pt")), None)
    config = _first(root, "config.yaml") or next(iter(_files(root, "*.yaml", "*/*.yaml")), None)
    if ckpt is None or config is None:
        return None
    return {"checkpoint": str(ckpt.relative_to(root)), "config": str(config.relative_to(root)),
            "generation": wfl_generation(ckpt, config)}


def wfl_generation(ckpt: Path, config: Path) -> int:
    """1 = WFL-ASR main branch (plain state dict), 2 = refactor branch (Lightning checkpoint, new config).

    Tells them apart without torch: a .pt/.ckpt is a zip whose data.pkl names its keys."""
    try:
        cfg = config.read_text(encoding="utf-8", errors="replace")
        if "forced_alignment_args" in cfg or "unfreeze_last_n_layers" in cfg or "\nfinetune:" in "\n" + cfg:
            return 2
    except OSError:
        pass
    try:
        import zipfile  # noqa: PLC0415

        with zipfile.ZipFile(ckpt) as z:
            pkl = next((n for n in z.namelist() if n.endswith("data.pkl")), None)
            if pkl is not None:
                data = z.read(pkl)
                if b"pytorch-lightning_version" in data or (b"state_dict" in data and b"model.encoder" in data):
                    return 2
    except (OSError, zipfile.BadZipFile):
        pass
    return 1


def detect_game(root: Path) -> dict[str, Any] | None:
    """A GAME ONNX model (official releases): config.json + encoder/segmenter/estimator/dur2bd/bd2dur .onnx."""
    names = ("encoder", "segmenter", "estimator", "dur2bd", "bd2dur")
    if not (root / "config.json").exists() or not all((root / f"{n}.onnx").exists() for n in names):
        return None
    layout: dict[str, Any] = {"format": "onnx", "config": "config.json", **{n: f"{n}.onnx" for n in names}}
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if config.get("languages"):
            layout["languages"] = [k for k in config["languages"] if config["languages"][k]]
    except (OSError, ValueError):
        pass
    return layout


def detect_hubertfa(root: Path) -> dict[str, Any] | None:
    """A HubertFA ONNX model: model.onnx + vocab.json + config.json (+ VERSION, dictionaries per language)."""
    model = _first(root, "model.onnx") or next(iter(_files(root, "*.onnx")), None)
    if model is None or not (root / "vocab.json").exists() or not (root / "config.json").exists():
        return None
    layout: dict[str, Any] = {"model": model.name, "vocab": "vocab.json", "config": "config.json"}
    if (root / "VERSION").exists():
        layout["version_file"] = "VERSION"
    try:
        vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
        dictionaries = {k: v for k, v in (vocab.get("dictionaries") or {}).items() if v and (root / v).exists()}
        layout["dictionaries"] = dictionaries
        layout["languages"] = list(dictionaries)
        layout["non_lexical_phonemes"] = list(vocab.get("non_lexical_phonemes") or [])
    except (OSError, ValueError):
        pass
    return layout


def detect_generic(root: Path) -> dict[str, Any] | None:
    ckpt = next(iter(_files(root, *(f"*{ext}" for ext in CHECKPOINT_EXT))), None)
    return {"checkpoint": ckpt.name} if ckpt is not None else {}


def detect_vocoder(root: Path) -> dict[str, Any] | None:
    """An ONNX vocoder (an OpenUtau .oudep package or a folder with the .onnx and vocoder.yaml)."""
    onnx = next(iter(_files(root, "*.onnx")), None)
    if onnx is None:
        return None
    layout: dict[str, Any] = {"onnx": onnx.name}
    if (root / "vocoder.yaml").exists():
        layout["config"] = "vocoder.yaml"
    return layout


DETECTORS = {
    "vocoder": detect_vocoder,
    "sofa": detect_sofa,
    "wfl_asr": detect_wfl_asr,
    "game": detect_game,
    "hubertfa": detect_hubertfa,
}


def detect(engine: str, root: Path) -> dict[str, Any] | None:
    return DETECTORS.get(engine, detect_generic)(root)


def find_model_dirs(engine: str, root: Path, max_depth: int = 4) -> list[Path]:
    """All folders below root (including root) that look like a model of the engine."""
    detector = DETECTORS.get(engine, detect_generic)
    found: list[Path] = []

    def walk(path: Path, depth: int) -> None:
        if detector(path):
            found.append(path)
            return  # don't look for models inside a model folder
        if depth >= max_depth:
            return
        for child in sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith(("__", "."))):
            walk(child, depth + 1)

    walk(root, 0)
    return found

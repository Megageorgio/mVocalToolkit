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
    return {"checkpoint": str(ckpt.relative_to(root)), "config": str(config.relative_to(root))}


def detect_game(root: Path) -> dict[str, Any] | None:
    # GAME loads a checkpoint file with config.yaml (and lang_map.json) in the same folder
    if not (root / "config.yaml").exists():
        return None
    ckpt = next(iter(_files(root, "*.ckpt", "*.pt", "*.pth", "*.safetensors")), None)
    if ckpt is None:
        return None
    layout: dict[str, Any] = {"model": ckpt.name, "config": "config.yaml"}
    if (root / "lang_map.json").exists():
        layout["lang_map"] = "lang_map.json"
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


DETECTORS = {
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

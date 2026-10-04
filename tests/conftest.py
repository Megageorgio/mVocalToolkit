from __future__ import annotations

import math
import struct
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FAKE_ENGINES = Path(__file__).parent / "fake_engines"


def make_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(i / 20))) for i in range(frames)))
    return path


def make_sofa_model(folder: Path, words: dict[str, str] | None = None, safetensors: bool = True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    words = words or {"hello": "hh ah l ow", "world": "w er l d", "привет": "p rj i vj e t"}
    (folder / "dict.txt").write_text("\n".join(f"{w}\t{p}" for w, p in words.items()) + "\n", encoding="utf-8")
    if safetensors:
        (folder / "model.safetensors").write_bytes(b"\0" * 16)
        (folder / "vocab.yaml").write_text("<vocab_size>: 3\n", encoding="utf-8")
        (folder / "global_config.yaml").write_text("x: 1\n", encoding="utf-8")
        (folder / "train_config.yaml").write_text("model: {}\n", encoding="utf-8")
    else:
        (folder / "model.ckpt").write_bytes(b"\0" * 16)
    g2p = folder / "g2p"
    g2p.mkdir(exist_ok=True)
    (g2p / "cfg.yaml").write_text("_target_: models.g2p_model.G2p\n", encoding="utf-8")
    (g2p / "model.ptsd").write_bytes(b"\0")
    return folder


def make_hubertfa_model(folder: Path) -> Path:
    import json  # noqa: PLC0415

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.onnx").write_bytes(b"\0" * 16)
    (folder / "VERSION").write_text("5\n", encoding="utf-8")
    (folder / "config.json").write_text(json.dumps({"mel_spec_config": {"sample_rate": 16000, "hop_size": 320}}),
                                        encoding="utf-8")
    (folder / "dictionaries").mkdir(exist_ok=True)
    (folder / "dictionaries" / "en.txt").write_text("hello\thh ah l ow\nworld\tw er l d\n", encoding="utf-8")
    (folder / "dictionaries" / "ja.txt").write_text("ka\tk a\n", encoding="utf-8")
    vocab = {"dictionaries": {"en": "dictionaries/en.txt", "ja": "dictionaries/ja.txt"}, "language_prefix": True,
             "non_lexical_phonemes": ["AP", "EP"], "vocab": {"SP": 0}, "vocab_size": 1}
    (folder / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    return folder

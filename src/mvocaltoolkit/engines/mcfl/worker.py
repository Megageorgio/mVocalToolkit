"""MCFL engine worker (github.com/Megageorgio/MCFL, MIT): a release model (model.json + model.safetensors) runs in
numpy; an exported package (model.onnx, labels.txt, langs.txt, ...) with its own mcfl_infer.py when it has one,
else with the copy next to this worker."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import mvt_engine as rt

_models: dict[str, Any] = {}


def _runtime(root: Path):
    """The mcfl_infer module of the package (it matches the model), else the bundled one."""
    own = root / "mcfl_infer.py"
    if own.is_file():
        spec = importlib.util.spec_from_file_location(f"mcfl_infer_{abs(hash(str(root)))}", own)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    import mcfl_infer  # noqa: PLC0415

    return mcfl_infer


def _load(path: str) -> Any:
    if path not in _models:
        root = Path(path)
        if (root / "model.json").is_file() and (root / "model.safetensors").is_file():
            _models[path] = _plain_model(root)
        else:
            threads = max(1, min(4, os.cpu_count() or 1))
            _models[path] = _runtime(root).MCFL(str(root), threads=threads)
    return _models[path]


def _plain_model(root: Path) -> Any:
    """A model of the release format "mcfl-v1" (model.json + model.safetensors, fp16, batch norms folded in): the
    network in numpy, the decoding of mcfl_infer."""
    import json  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    import mcfl_infer as mi  # noqa: PLC0415

    def read_safetensors(path: Path) -> dict[str, Any]:
        b = path.read_bytes()
        n = int.from_bytes(b[:8], "little")
        head = json.loads(b[8:8 + n])
        types = {"F16": np.float16, "F32": np.float32}
        out = {}
        for k, d in head.items():
            if k == "__metadata__":
                continue
            s, e = d["data_offsets"]
            out[k] = np.frombuffer(b[8 + n + s:8 + n + e], dtype=types[d["dtype"]]).astype(np.float32).reshape(d["shape"])
        return out

    def erf(x):  # Abramowitz and Stegun 7.1.26, error below 1.5e-7
        a = np.abs(x)
        t = 1 / (1 + 0.3275911 * a)
        y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
        return np.sign(x) * y

    def gelu(x):
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))

    class Plain(mi.MCFL):
        def __init__(self) -> None:  # noqa: D107  (no onnxruntime session)
            h = json.loads((root / "model.json").read_text(encoding="utf-8"))
            if h.get("format") != "mcfl-v1":
                raise ValueError(f"{root}: unknown MCFL format {h.get('format')!r}")
            self.h = h
            self.cfg = {**h.get("decode", {}), "frame_sec": h.get("frame_sec", 0.02)}
            self.labels = list(h["labels"])
            self.langs = dict(h["langs"])
            self.merge = h.get("phoneme_merge_map", {})
            self.decoders = mi.LanguageDecoders(self.labels, self.langs, h["lang_phonemes"], h.get("shared_labels", mi.SHARED))
            self.fd = self.cfg["frame_sec"]
            self.model_file = "model.safetensors"
            self.w = read_safetensors(root / "model.safetensors")
            n_fft = h["n_fft"]
            self.win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)

        def _run(self, audio, lid):
            h, w = self.h, self.w
            n_fft, hop = h["n_fft"], h["hop"]
            x = np.pad(audio.astype(np.float32), (n_fft // 2, n_fft // 2), mode="reflect")
            n = 1 + (len(x) - n_fft) // hop
            idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
            power = np.abs(np.fft.rfft(x[idx] * self.win, axis=1)) ** 2
            mel = (np.log(np.maximum(power.astype(np.float32) @ w["mel_fb"], 1e-5)) - w["mel_mean"]) / w["mel_std"]  # [F, mels]
            sw, k = w["stem.w"], w["stem.w"].shape[2]
            mp = np.pad(mel, ((k // 2, k // 2), (0, 0)))
            t = (len(mel) + 2 * (k // 2) - k) // 2 + 1
            y = sum(mp[j:j + 2 * (t - 1) + 1:2] @ sw[:, :, j].T for j in range(k)) + w["stem.b"]
            y = gelu(y) + w["lang"][lid]
            for i, d in enumerate(h["dilations"]):
                dw = w[f"blocks.{i}.dw.w"]
                kk = dw.shape[1]
                p = d * (kk - 1) // 2
                yp = np.pad(y, ((p, p), (0, 0)))
                z = sum(yp[j * d:j * d + t] * dw[:, j] for j in range(kk)) + w[f"blocks.{i}.dw.b"]
                z = gelu(z @ w[f"blocks.{i}.pw1.w"].T + w[f"blocks.{i}.pw1.b"])
                y = y + z @ w[f"blocks.{i}.pw2.w"].T + w[f"blocks.{i}.pw2.b"]
            lg = y @ w["cls.w"].T + w["cls.b"]
            off = 1 / (1 + np.exp(-(y @ w["off.w"].T + w["off.b"])))
            return mi.log_softmax(lg), off

    return Plain()


@rt.method()
def segment(
    model: dict[str, Any],
    items: list[dict[str, Any]],
    language: str | None = None,
    silence_threshold: float | None = None,
    min_silence_duration: float | None = None,
    **_ignored: Any,  # options of the WFL-ASR engines
) -> list[dict[str, Any]]:
    m = _load(model["path"])
    langs = list(m.langs)
    code = (language or "").lower().split("-")[0].split("_")[0]
    if code not in m.langs:
        code = langs[0]
        rt.log(f"MCFL: language {language!r} is not in the model ({', '.join(langs)}), using {code}")
    # the request's silence options over the package's own (config.json); the Viterbi bias stays the package's,
    # picked for this model when it was exported (the WFL-ASR default of the request doesn't suit it)
    for key, value in (("silence_threshold", silence_threshold), ("min_silence_sec", min_silence_duration)):
        if value is not None:
            m.cfg[key] = value
    results = []
    for index, item in enumerate(items):
        name = item.get("name") or Path(item["audio"]).stem
        rt.progress(index / len(items), f"Segmenting {name}")
        try:
            segments = m.label_file(item["audio"], lang=code)
            results.append({"ok": True, "phones": [[float(s), float(e), str(p)] for s, e, p in segments]})
        except Exception as e:  # noqa: BLE001
            results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
    rt.progress(1.0, "Segmented")
    return results


if __name__ == "__main__":
    rt.run()

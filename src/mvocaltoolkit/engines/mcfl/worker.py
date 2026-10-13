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


def _load(path: str, layout: dict[str, Any] | None = None) -> Any:
    if path not in _models:
        root = Path(path)
        config = (layout or {}).get("config") or ("mcfl.json" if (root / "mcfl.json").is_file() else "model.json")
        if (root / "model.safetensors").is_file():
            _models[path] = _plain_model(root, config)
        else:
            threads = max(1, min(4, os.cpu_count() or 1))
            _models[path] = _runtime(root).MCFL(str(root), threads=threads)
        _models[path]._dir = str(root)
    return _models[path]


def _plain_model(root: Path, config: str = "model.json") -> Any:
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
            h = json.loads((root / config).read_text(encoding="utf-8"))
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


def _forced(m: Any, path: str, lang: str, phones: list[str], confidence: bool) -> list[list[Any]]:
    """Places known [phones] (in order, the language's own names): the best path through their B-/I- tags, each
    at least one frame; a phoneme the model doesn't know may take any label of the language. Continuous segments
    with exactly the given names (as mLabeler's built-in model does)."""
    import numpy as np  # noqa: PLC0415

    import mcfl_infer as mi  # noqa: PLC0415

    a = mi.load_audio(path)
    a = a / (np.abs(a).max() + 1e-8)
    lid = m.langs[lang]
    lp, off = m.frames(a, lid)
    lang_phonemes = m.h.get("lang_phonemes") if hasattr(m, "h") else None
    if lang_phonemes is None:
        import json  # noqa: PLC0415
        lang_phonemes = json.loads((Path(m._dir) / "lang_phonemes.json").read_text(encoding="utf-8"))
    keep = set(lang_phonemes.get(lang, [])) | set(mi.SHARED)
    cols = [i for i, lab in enumerate(m.labels) if lab != "O" and lab[2:] in keep]
    sub = lp[:, cols].astype(np.float64)
    mx = sub.max(1, keepdims=True)
    sub = sub - mx - np.log(np.exp(sub - mx).sum(1, keepdims=True))
    col = {m.labels[c]: j for j, c in enumerate(cols)}
    to_canon = {native: canon for canon, per in m.merge.items() for lg, native in per.items() if lg == lang}
    canon = [to_canon.get(p, p) for p in phones]
    t, k = len(sub), len(phones)
    if k > t:
        raise ValueError(f"{k} phonemes don't fit in {t * m.fd:.2f} s")
    best = sub.max(1)
    b = np.stack([sub[:, col[f"B-{c}"]] if f"B-{c}" in col and f"I-{c}" in col else best for c in canon], 1)
    i_ = np.stack([sub[:, col[f"I-{c}"]] if f"B-{c}" in col and f"I-{c}" in col else best for c in canon], 1)
    logb = np.log(m.cfg.get("viterbi_bias", 1.0))
    neg = -1e18
    sb = np.full(k, neg); si = np.full(k, neg)
    sb[0] = b[0, 0]
    from_b = np.zeros((t, k), bool)
    start_from_b = np.zeros((t, k), bool)
    for f in range(1, t):
        start_from_b[f, 1:] = sb[:-1] >= si[:-1]
        prev_end = np.concatenate([[neg], np.maximum(sb[:-1], si[:-1])])
        from_b[f] = sb >= si
        sb, si = prev_end + b[f], np.maximum(sb, si) + logb + i_[f]
    j = k - 1
    in_b = sb[j] >= si[j]
    starts = [0] * k
    for f in range(t - 1, 0, -1):
        if in_b:
            starts[j] = f
            in_b = bool(start_from_b[f, j])
            j -= 1
            if j < 0:
                break
        else:
            in_b = bool(from_b[f, j])
    fd = m.fd
    p = np.exp(lp)
    index = {lab: n for n, lab in enumerate(m.labels)}
    dur = len(a) / mi.SR
    out, prev = [], -1.0
    for n in range(k):
        f0 = starts[n]
        f1 = starts[n + 1] - 1 if n + 1 < k else t - 1
        s = 0.0 if n == 0 else min(dur, max(prev + 1e-4, (f0 + float(off[f0, 0])) * fd))
        prev = s
        row = [s, None, phones[n]]
        if confidence:
            cs = [index[x] for x in (f"B-{canon[n]}", f"I-{canon[n]}") if x in index]
            row.append(round(float(p[f0:f1 + 1, cs].sum(1).mean()), 4) if cs else None)
        out.append(row)
    for n in range(k):
        out[n][1] = out[n + 1][0] if n + 1 < k else dur
    return out


def _with_confidence(m: Any, path: str, lang: str) -> list[list[Any]]:
    """As MCFL.label, each segment with the mean probability (B- + I- of its label) over its frames."""
    import numpy as np  # noqa: PLC0415

    import mcfl_infer as mi  # noqa: PLC0415

    a = mi.load_audio(path)
    a = a / (np.abs(a).max() + 1e-8)
    lid = m.langs[lang]
    lp, off = m.frames(a, lid)
    segs = m.decoders(lp, off, a, lid, m.cfg)
    p = np.exp(lp)
    index = {lab: i for i, lab in enumerate(m.labels)}
    fd = m.fd
    out = []
    for s, e, ph in segs:
        cols = [index[c] for c in (f"B-{ph}", f"I-{ph}") if c in index]
        f0 = min(len(p) - 1, max(0, int(s / fd)))
        f1 = min(len(p), max(f0 + 1, int(round(e / fd))))
        conf = round(float(p[f0:f1, cols].sum(1).mean()), 4) if cols else None
        out.append([float(s), float(e), str(m.merge.get(ph, {}).get(lang, ph)), conf])
    return out


@rt.method()
def segment(
    model: dict[str, Any],
    items: list[dict[str, Any]],
    language: str | None = None,
    silence_threshold: float | None = None,
    min_silence_duration: float | None = None,
    confidence: bool = False,
    **_ignored: Any,  # options of the WFL-ASR engines
) -> list[dict[str, Any]]:
    m = _load(model["path"], model.get("layout"))
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
            known = [x for x in (item.get("phonemes") or []) if x]
            if known:  # the phonemes are given: only their boundaries are found
                results.append({"ok": True, "phones": _forced(m, item["audio"], code, known, confidence)})
            elif confidence:
                results.append({"ok": True, "phones": _with_confidence(m, item["audio"], code)})
            else:
                segments = m.label_file(item["audio"], lang=code)
                results.append({"ok": True, "phones": [[float(s), float(e), str(p)] for s, e, p in segments]})
        except Exception as e:  # noqa: BLE001
            results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
    rt.progress(1.0, "Segmented")
    return results


if __name__ == "__main__":
    rt.run()

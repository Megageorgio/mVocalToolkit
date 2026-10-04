"""SOFA engine worker. Runs inside the SOFA environment with the SOFA source code on sys.path."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import mvt_engine as rt

warnings.filterwarnings("ignore")

_models: dict[str, "LoadedModel"] = {}


class LoadedModel:
    def __init__(self, path: str, layout: dict[str, Any], device: str):
        import numpy as np  # noqa: F401,PLC0415
        import torch  # noqa: PLC0415
        import yaml  # noqa: PLC0415

        import modules.utils.get_melspec as melspec_module  # noqa: PLC0415
        from modules.task.forced_alignment import LitForcedAlignmentTask  # noqa: PLC0415

        self.root = Path(path)
        self.layout = layout
        self.device = device
        ckpt = self.root / layout["checkpoint"]
        if layout.get("format") == "safetensors":
            from safetensors.torch import load_file  # noqa: PLC0415

            def read_yaml(key: str) -> dict:
                name = layout.get(key)
                if not name:
                    raise FileNotFoundError(f"{key} config is required for a .safetensors SOFA model")
                return yaml.safe_load((self.root / name).read_text(encoding="utf-8")) or {}

            config = read_yaml("train_config")
            if layout.get("global_config"):
                config.update(read_yaml("global_config"))
            vocab_text = (self.root / layout["vocab"]).read_text(encoding="utf-8") if layout.get("vocab") else None
            if vocab_text is None:
                raise FileNotFoundError("vocab.yaml is required for a .safetensors SOFA model")
            task = LitForcedAlignmentTask(
                vocab_text,
                config["model"],
                config["melspec_config"],
                config.get("optimizer_config", {}),
                config.get("loss_config", _DEFAULT_LOSS),
                False,
            )
            state = load_file(str(ckpt))
            state = _strip_prefix(state, task.state_dict().keys())
            missing, unexpected = task.load_state_dict(state, strict=False)
            important = [k for k in missing if k.startswith(("backbone", "head"))]
            if important:
                raise RuntimeError(f"Model weights don't match the config: missing {important[:5]}")
        else:
            task = LitForcedAlignmentTask.load_from_checkpoint(str(ckpt), map_location="cpu", strict=False)
        task.to(device).eval()
        # MelSpecExtractor keeps a global transform; reset it so models with other mel configs work
        melspec_module.melspec_transform = None
        cfg = dict(task.melspec_config)
        cfg.pop("device", None)
        task.get_melspec = melspec_module.MelSpecExtractor(**cfg, device=device)
        self.task = task
        self.vocab: dict[str, Any] = task.vocab
        self.dictionary: dict[str, list[str]] = {}
        if layout.get("dictionary"):
            self.dictionary = _load_dictionary(self.root / layout["dictionary"])
        self.g2p = None
        g2p = layout.get("g2p")
        if g2p:
            try:
                from g2p_model import load_g2p  # noqa: PLC0415

                self.g2p = load_g2p(self.root / g2p["config"], self.root / g2p["weights"], device="cpu")
            except Exception as e:  # noqa: BLE001
                rt.log(f"G2P model could not be loaded: {e}")
        self._melspec_module = melspec_module

    def activate(self) -> None:
        # the mel transform is global in SOFA: make sure it belongs to this model
        self._melspec_module.melspec_transform = None
        cfg = dict(self.task.melspec_config)
        cfg.pop("device", None)
        self.task.get_melspec = self._melspec_module.MelSpecExtractor(**cfg, device=self.device)

    def lookup(self, word: str) -> list[str] | None:
        for variant in (word, word.lower(), word.lower().replace("ё", "е")):
            if variant in self.dictionary:
                return self.dictionary[variant]
        return None

    def guess(self, word: str) -> list[str] | None:
        if self.g2p is None:
            return None
        phones = [p for p in self.g2p.predict_str(word.lower()) if p in self.vocab]
        return phones or None


_DEFAULT_LOSS = {
    "function": {"num_bins": 10, "alpha": 0.999, "label_smoothing": 0.08, "pseudo_label_ratio": 0.3},
    "losses": {
        "weights": [10.0, 0.1, 0.01, 0.1, 1.0, 1.0, 5.0],
        "enable_RampUpScheduler": [False, False, False, True, True, True, True],
    },
}


def _strip_prefix(state: dict, expected_keys) -> dict:
    expected = set(expected_keys)
    if any(k in expected for k in state):
        return state
    for prefix in ("model.", "module.", "state_dict."):
        stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if any(k in expected for k in stripped):
            return stripped
    return state


def _load_dictionary(path: Path) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            if "\t" in line:
                word, phones = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue
                word, phones = parts
            word = word.strip()
            if word and word not in entries:
                entries[word] = phones.split()
    return entries


def _get_model(model: dict[str, Any]) -> LoadedModel:
    key = model["path"]
    loaded = _models.get(key)
    if loaded is None:
        if len(_models) >= 2:  # keep at most two aligners in memory
            _models.pop(next(iter(_models)))
            rt.free_memory()
        rt.progress(message="Loading SOFA model")
        loaded = _models[key] = LoadedModel(model["path"], model.get("layout", {}), rt.device())
    loaded.activate()
    return loaded


def build_sequence(model: LoadedModel, item: dict[str, Any], g2p: str, skip_unknown: bool):
    """words/phonemes -> (ph_seq, word_seq, ph_idx_to_word_idx, unknown_words), SOFA conventions."""
    if item.get("phonemes"):
        tokens = [p for p in item["phonemes"] if p not in ("SP",)]
        mode = "phonemes"
    else:
        tokens = list(item.get("words") or [])
        mode = "phonemes" if g2p == "none" else "words"
    ph_seq = ["SP"]
    ph_idx_to_word_idx = [-1]
    word_seq: list[str] = []
    unknown: list[str] = []
    for token in tokens:
        if token == "SP":
            if ph_seq[-1] == "SP":
                ph_idx_to_word_idx[-1] = -2  # explicit phrase boundary
            continue
        if mode == "phonemes":
            phones = [token]
        else:
            phones = model.lookup(token)
            if phones is None and g2p == "auto":
                phones = model.guess(token)
            if phones is None:
                if token not in unknown:
                    unknown.append(token)
                continue
        phones = [p for i, p in enumerate(phones) if not ((i == 0 or i == len(phones) - 1) and p == "SP")]
        missing_ph = [p for p in phones if p not in model.vocab]
        if missing_ph:
            if token not in unknown:
                unknown.append(token)
            continue
        word_index = len(word_seq)
        word_seq.append(token)
        for ph in phones:
            ph_seq.append(ph)
            ph_idx_to_word_idx.append(word_index)
        if ph_seq[-1] != "SP":
            ph_seq.append("SP")
            ph_idx_to_word_idx.append(-1)
    if unknown and not skip_unknown:
        return None, None, None, unknown
    return ph_seq, word_seq, ph_idx_to_word_idx, unknown


@rt.method()
def align(model: dict[str, Any], items: list[dict[str, Any]], mode: str = "force", g2p: str = "auto",
          ap_detector: str = "loudness_spectral_centroid", skip_unknown_words: bool = False,
          dictionary: str | None = None) -> list[dict[str, Any]]:
    loaded = _get_model(model)
    original = loaded.dictionary
    if dictionary:
        loaded.dictionary = _load_dictionary(Path(dictionary))
    try:
        return _align(loaded, items, mode, g2p, ap_detector, skip_unknown_words)
    finally:
        loaded.dictionary = original


def _align(loaded: "LoadedModel", items: list[dict[str, Any]], mode: str, g2p: str, ap_detector: str,
           skip_unknown_words: bool) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    from modules.utils.post_processing import post_processing  # noqa: PLC0415

    loaded.task.set_inference_mode(mode)
    detector = None
    if ap_detector != "none":
        import modules.AP_detector  # noqa: PLC0415

        detector = modules.AP_detector.LoudnessSpectralcentroidAPDetector()
    results: list[dict[str, Any]] = []
    total = len(items)
    for index, item in enumerate(items):
        name = item.get("name") or Path(item["audio"]).stem
        rt.progress(index / total, f"Aligning {name}")
        try:
            ph_seq, word_seq, ph_map, unknown = build_sequence(loaded, item, g2p, skip_unknown_words)
            if ph_seq is None:
                results.append({"ok": False, "error": "Unknown words: " + " ".join(unknown), "unknown_words": unknown})
                continue
            if len(word_seq) == 0:
                results.append({"ok": False, "error": "Nothing to align (no known words)", "unknown_words": unknown})
                continue
            prediction = _predict(loaded, Path(item["audio"]), ph_seq, word_seq, ph_map)
            if detector is not None:
                prediction = detector.process([prediction])[0]
            processed, errors = post_processing([prediction])
            if errors:
                raise errors[0][1]
            _, wav_length, confidence, ph_seq_o, ph_iv, word_seq_o, word_iv = processed[0]
            results.append(
                {
                    "ok": True,
                    "duration": float(wav_length),
                    "confidence": float(confidence),
                    "phones": [[float(s), float(e), p] for p, (s, e) in zip(ph_seq_o, ph_iv)],
                    "words": [[float(s), float(e), w] for w, (s, e) in zip(word_seq_o, word_iv)],
                    "unknown_words": unknown,
                }
            )
        except Exception as e:  # noqa: BLE001
            if rt.is_oom(e):
                rt.free_memory()
            results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    rt.progress(1.0, "Aligned")
    return results


def _predict(loaded: LoadedModel, wav_path: Path, ph_seq, word_seq, ph_map):
    """Same as LitForcedAlignmentTask.predict_step, without Lightning's Trainer."""
    import torch  # noqa: PLC0415
    import librosa  # noqa: PLC0415
    from einops import repeat  # noqa: PLC0415

    task = loaded.task
    sr = task.melspec_config["sample_rate"]
    with torch.no_grad():
        # SOFA's load_wav prefers torchaudio.load, which needs torchcodec (and ffmpeg) since torchaudio 2.9
        audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
        waveform = torch.from_numpy(audio).to(loaded.device)
        wav_length = waveform.shape[0] / sr
        melspec = task.get_melspec(waveform).detach().unsqueeze(0)
        melspec = (melspec - melspec.mean()) / melspec.std()
        melspec = repeat(melspec, "B C T -> B C (T N)", N=task.melspec_config["scale_factor"])
        ph_seq_o, ph_iv, word_seq_o, word_iv, confidence, _, _ = task._infer_once(
            melspec, wav_length, ph_seq, word_seq, ph_map, False, False
        )
    return (wav_path, wav_length, confidence, ph_seq_o, ph_iv, word_seq_o, word_iv)


@rt.method()
def g2p(model: dict[str, Any], words: list[str]) -> dict[str, list[str] | None]:
    loaded = _get_model(model)
    return {w: loaded.lookup(w) or loaded.guess(w) for w in words}


@rt.method()
def model_info(model: dict[str, Any]) -> dict[str, Any]:
    loaded = _get_model(model)
    return {
        "phonemes": sorted(k for k in loaded.vocab if not str(k).startswith("<")),
        "dictionary_size": len(loaded.dictionary),
        "has_g2p": loaded.g2p is not None,
        "melspec": dict(loaded.task.melspec_config),
    }


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    rt.run()

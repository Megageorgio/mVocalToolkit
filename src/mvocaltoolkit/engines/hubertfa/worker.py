"""HubertFA engine worker (ONNX inference). Runs with the HubertFA source code on sys.path."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import mvt_engine as rt

warnings.filterwarnings("ignore")

_models: dict[str, "LoadedModel"] = {}


def _providers() -> list[str]:
    import onnxruntime as ort  # noqa: PLC0415

    available = ort.get_available_providers()
    requested = (os.environ.get("MVT_DEVICE") or "auto").lower()
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()  # CUDA / cuDNN from the nvidia-* wheels
        except Exception as e:  # noqa: BLE001
            rt.log(f"Could not preload CUDA libraries: {e}")
    order = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in available] or ["CPUExecutionProvider"]


class LoadedModel:
    def __init__(self, path: str, layout: dict[str, Any]):
        import onnxruntime as ort  # noqa: PLC0415

        from tools.decoder import AlignmentDecoder, NonLexicalDecoder  # noqa: PLC0415

        self.root = Path(path)
        self.layout = layout
        version_file = self.root / layout.get("version_file", "VERSION")
        if version_file.exists():
            version = int(version_file.read_text(encoding="utf-8").split()[0])
            if version != 5:
                raise RuntimeError(f"Unsupported HubertFA ONNX model version {version} (5 is required)")
        self.vocab = json.loads((self.root / layout.get("vocab", "vocab.json")).read_text(encoding="utf-8"))
        config = json.loads((self.root / layout.get("config", "config.json")).read_text(encoding="utf-8"))
        self.mel_cfg = config["mel_spec_config"]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = _providers()
        self.session = ort.InferenceSession(str(self.root / layout.get("model", "model.onnx")), options,
                                            providers=providers)
        rt.log(f"HubertFA providers: {self.session.get_providers()}")
        self.fa_decoder = AlignmentDecoder(vocab=self.vocab, sample_rate=self.mel_cfg["sample_rate"],
                                           hop_size=self.mel_cfg["hop_size"])
        self.nll_decoder = NonLexicalDecoder(vocab=self.vocab,
                                             class_names=["None", *self.vocab["non_lexical_phonemes"]],
                                             sample_rate=self.mel_cfg["sample_rate"], hop_size=self.mel_cfg["hop_size"])
        self._dictionaries: dict[str, dict[str, list[str]]] = {}

    # ----- dictionaries -----

    @property
    def languages(self) -> list[str]:
        return [k for k, v in (self.vocab.get("dictionaries") or {}).items() if v]

    def resolve_language(self, language: str | None) -> str | None:
        languages = self.languages
        if language in languages:
            return language
        if language:
            base = language.lower().replace("-", "_").split("_")[0]
            for lang in languages:
                if lang.lower().split("_")[0] == base:
                    return lang
        if len(languages) == 1:
            return languages[0]
        if not languages:
            return None
        raise ValueError(f"Language {language!r} is not supported by this model, available: {', '.join(languages)}")

    def dictionary(self, language: str | None, custom: str | None = None) -> dict[str, list[str]]:
        key = custom or language or ""
        if key in self._dictionaries:
            return self._dictionaries[key]
        if custom:
            path = Path(custom)
        else:
            name = (self.vocab.get("dictionaries") or {}).get(language or "")
            if not name:
                raise FileNotFoundError(f"The model has no dictionary for {language!r}")
            path = self.root / name
        table: dict[str, list[str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            word, phones = line.split("\t", 1)
            table.setdefault(word.strip(), phones.strip().split())
        self._dictionaries[key] = table
        return table

    def phone_id(self, language: str | None, phone: str) -> str | None:
        vocab = self.vocab["vocab"]
        if self.vocab.get("language_prefix") and language and phone != "SP":
            prefixed = f"{language}/{phone}"
            if prefixed in vocab:
                return prefixed
        return phone if phone in vocab else None

    def lookup(self, table: dict[str, list[str]], word: str) -> list[str] | None:
        for candidate in (word, word.lower(), word.replace("ё", "е"), word.lower().replace("ё", "е")):
            if candidate in table:
                return table[candidate]
        return None


def _get_model(model: dict[str, Any]) -> LoadedModel:
    key = model["path"]
    if key not in _models:
        _models.clear()
        rt.progress(message="Loading HubertFA model")
        _models[key] = LoadedModel(model["path"], model.get("layout", {}))
    return _models[key]


def build_sequence(model: LoadedModel, item: dict[str, Any], language: str | None, g2p: str,
                   skip_unknown: bool, dictionary: str | None, extra_words: dict[str, list[str]] | None = None):
    """words/phonemes -> (ph_seq, word_seq, ph_idx_to_word_idx, unknown) in HubertFA conventions."""
    if item.get("phonemes"):
        tokens, mode = [p for p in item["phonemes"] if p != "SP"], "phonemes"
    else:
        tokens = list(item.get("words") or [])
        mode = "phonemes" if g2p == "none" else "words"
    table = model.dictionary(language, dictionary) if mode == "words" else {}
    if extra_words and mode == "words":
        table = {**table, **{w: list(p) for w, p in extra_words.items() if w and p}}
    ph_seq, ph_map, word_seq, unknown = ["SP"], [-1], [], []
    for token in tokens:
        if token == "SP":
            continue
        phones = [token] if mode == "phonemes" else model.lookup(table, token)
        if phones is None and model.phone_id(language, token) is not None:
            phones = [token]  # the text frontend already produced phonemes (e.g. Japanese via pyopenjtalk)
        if phones is None:
            if token not in unknown:
                unknown.append(token)
            continue
        phones = [p for i, p in enumerate(phones) if not ((i == 0 or i == len(phones) - 1) and p == "SP")]
        ids = [model.phone_id(language, p) for p in phones]
        if not ids or any(i is None for i in ids):
            if token not in unknown:
                unknown.append(token)
            continue
        word_index = len(word_seq)
        word_seq.append(token)
        for ph in ids:
            ph_seq.append(ph)
            ph_map.append(word_index if ph != "SP" else -1)
        if ph_seq[-1] != "SP":
            ph_seq.append("SP")
            ph_map.append(-1)
    if unknown and not skip_unknown:
        return None, None, None, unknown
    return ph_seq, word_seq, ph_map, unknown


def _infer_item(model: LoadedModel, wav_path: Path, ph_seq, word_seq, ph_map, non_lexical: list[str],
                pad_times: int, pad_length: float):
    """InferenceBase.infer for one file, without the folder-based dataset."""
    from tools.infer_base import InferenceBase  # noqa: PLC0415

    confidences: list[float] = []

    class _Inference(InferenceBase):
        def _infer(self, padded_wav, padded_frames, word_seq, ph_seq, ph_idx_to_word_idx, wav_length,
                   non_lexical_phonemes):
            names = [o.name for o in model.session.get_outputs()]
            out = dict(zip(names, model.session.run(names, {"waveform": [padded_wav]})))
            words, confidence = model.fa_decoder.decode(
                ph_frame_logits=out["ph_frame_logits"][:, :, padded_frames:],
                ph_edge_logits=out["ph_edge_logits"][:, padded_frames:],
                wav_length=wav_length, ph_seq=ph_seq, word_seq=word_seq, ph_idx_to_word_idx=ph_idx_to_word_idx,
            )
            try:
                confidences.append(float(confidence))
            except (TypeError, ValueError):
                pass
            non_lexical_words = model.nll_decoder.decode(cvnt_logits=out["cvnt_logits"][:, :, padded_frames:],
                                                         wav_length=wav_length,
                                                         non_lexical_phonemes=non_lexical_phonemes)
            return words, non_lexical_words

    inference = _Inference()
    inference.vocab = model.vocab
    inference.mel_cfg = model.mel_cfg
    inference.vocab_folder = model.root
    inference.dataset = [(wav_path, ph_seq, word_seq, ph_map)]
    inference.infer(non_lexical_phonemes=",".join(non_lexical), pad_times=pad_times, pad_length=pad_length)
    _, wav_length, words = inference.predictions[0]
    confidence = sum(confidences) / len(confidences) if confidences else None
    return wav_length, words, confidence


@rt.method()
def align(model: dict[str, Any], items: list[dict[str, Any]], language: str | None = None, g2p: str = "auto",
          non_lexical_phonemes: list[str] | None = None, pad_times: int = 1, pad_length: float = 5.0,
          skip_unknown_words: bool = False, dictionary: str | None = None,
          extra_words: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
    loaded = _get_model(model)
    lang = loaded.resolve_language(language)
    allowed = set(loaded.vocab.get("non_lexical_phonemes") or [])
    non_lexical = [p for p in (non_lexical_phonemes if non_lexical_phonemes is not None else ["AP"]) if p in allowed]
    results: list[dict[str, Any]] = []
    total = len(items)
    for index, item in enumerate(items):
        name = item.get("name") or Path(item["audio"]).stem
        rt.progress(index / total, f"Aligning {name}")
        try:
            ph_seq, word_seq, ph_map, unknown = build_sequence(loaded, item, lang, g2p, skip_unknown_words,
                                                               dictionary, extra_words)
            if ph_seq is None:
                results.append({"ok": False, "error": "Unknown words: " + " ".join(unknown), "unknown_words": unknown})
                continue
            if not word_seq:
                results.append({"ok": False, "error": "Nothing to align (no known words)", "unknown_words": unknown})
                continue
            wav_length, words, confidence = _infer_item(loaded, Path(item["audio"]), ph_seq, word_seq, ph_map,
                                                        non_lexical, max(1, pad_times), pad_length)
            phones, word_rows = [], []
            for word in words:
                for ph in word.phonemes:
                    phones.append([float(ph.start), float(ph.end), ph.text])
                word_rows.append([float(word.start), float(word.end), word.text])
            results.append({"ok": True, "duration": float(wav_length), "confidence": confidence, "phones": phones,
                            "words": word_rows, "unknown_words": unknown, "language": lang})
        except Exception as e:  # noqa: BLE001
            results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
    rt.progress(1.0, "Aligned")
    return results


@rt.method()
def g2p(model: dict[str, Any], words: list[str], language: str | None = None) -> dict[str, list[str] | None]:
    loaded = _get_model(model)
    lang = loaded.resolve_language(language)
    table = loaded.dictionary(lang)
    return {w: loaded.lookup(table, w) for w in words}


@rt.method()
def model_info(model: dict[str, Any]) -> dict[str, Any]:
    loaded = _get_model(model)
    return {
        "languages": loaded.languages,
        "language_prefix": bool(loaded.vocab.get("language_prefix")),
        "non_lexical_phonemes": loaded.vocab.get("non_lexical_phonemes", []),
        "providers": loaded.session.get_providers(),
    }


if __name__ == "__main__":
    rt.run()

"""TIFA engine worker (PyTorch inference). Runs with the TIFA source code on sys.path.

TIFA does its own G2P (g2pflow pipeline from the model's config.yaml): plain text goes in as it is,
fixed words and known phonemes go in as PFML.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import mvt_engine as rt

warnings.filterwarnings("ignore")

_models: dict[str, "LoadedModel"] = {}
SILENCE = "SP"


def _japanese_dictionary() -> str | None:
    """unidic-lite instead of g2pflow downloading the full UniDic (770 MB), unless the full one is already there."""
    try:
        import unidic  # noqa: PLC0415

        if (Path(unidic.DICDIR) / "sys.dic").exists():
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        import unidic_lite  # noqa: PLC0415

        return unidic_lite.DICDIR
    except Exception:  # noqa: BLE001
        return None


class LoadedModel:
    def __init__(self, path: str, layout: dict[str, Any]):
        from inference.api import load_inference_model  # noqa: PLC0415
        from lib.config.schema import ConfigurationScope  # noqa: PLC0415

        self.root = Path(path)
        checkpoint = self.root / layout.get("checkpoint", "model.pt")
        self.backend, self.vocabulary, inference_config = load_inference_model(checkpoint, scope=ConfigurationScope.FA)
        if inference_config.g2p is None:
            raise RuntimeError("The model has no G2P configuration (inference.g2p in config.yaml)")
        self.g2p_config = inference_config.g2p
        mecab_dir = _japanese_dictionary()
        for converter in self.g2p_config.converters:
            if converter.id == "japanese-mecab" and mecab_dir and "unidic_dir" not in (converter.kwargs or {}):
                converter.kwargs = {**(converter.kwargs or {}), "unidic_dir": mecab_dir}
        self._pipeline = None
        symbols = self.vocabulary.symbol_to_id if hasattr(self.vocabulary, "symbol_to_id") else {}
        self.languages = sorted({s.split("/", 1)[0] for s in symbols if "/" in s})

    @property
    def pipeline(self):
        if self._pipeline is None:
            from lib.g2p import build_pipeline_from_config  # noqa: PLC0415

            self._pipeline = build_pipeline_from_config(self.g2p_config, root_path=self.root)
        return self._pipeline

    def resolve_language(self, language: str | None) -> str | None:
        if not language:
            return self.languages[0] if len(self.languages) == 1 else None
        if language in self.languages:
            return language
        base = language.lower().replace("_", "-").split("-")[0]
        aliases = {"cmn": "zh", "jp": "ja", "jpn": "ja", "eng": "en", "can": "yue", "zh-yue": "yue"}
        base = aliases.get(base, base)
        for lang in self.languages:
            if lang.lower() == base:
                return lang
        raise ValueError(f"Language {language!r} is not supported by this model, available: {', '.join(self.languages)}")

    def languages_for(self, language: str | None, extra: list[str] | None) -> list[str] | None:
        """Default language first (its prefix is dropped from the labels), then the extra ones in priority order."""
        result = [language] if language else []
        for lang in extra or []:
            resolved = self.resolve_language(lang)
            if resolved and resolved not in result:
                result.append(resolved)
        return result or None


def _get_model(model: dict[str, Any]) -> LoadedModel:
    key = model["path"]
    if key not in _models:
        _models.clear()
        rt.progress(message="Loading TIFA model")
        _models[key] = LoadedModel(model["path"], model.get("layout", {}))
    return _models[key]


def _pfml_word(text: str, phonemes: list[str] | None = None, language: str | None = None) -> str:
    attrs = f"text={quoteattr(text)}"
    if language:
        attrs += f" language={quoteattr(language)}"  # bare phoneme names are looked up with this prefix
    if phonemes is not None:
        attrs += f" phonemes={quoteattr(' '.join(phonemes))}"
    return f"<word {attrs}/>"


def _source(item: dict[str, Any], g2p: str, extra_words: dict[str, list[str]] | None,
            language: str | None) -> tuple[str, bool]:
    """(text, is_pfml) for one item: plain text, fixed words, or known phonemes."""
    if item.get("phonemes"):
        phones = [p for p in item["phonemes"] if p and p != SILENCE]
        return "".join(_pfml_word(p, [p], language) for p in phones), True
    if item.get("words"):
        words = [w for w in item["words"] if w and w != SILENCE]
        if g2p == "none":
            return "".join(_pfml_word(w, [w], language) for w in words), True
        extra = extra_words or {}
        return "".join(_pfml_word(w, extra.get(w), language) for w in words), True
    text = (item.get("text") or "").strip()
    if extra_words and text:
        # user words in a plain text: fixed as words with the given phonemes, the rest goes through G2P
        tokens = text.split()
        if any(t in extra_words for t in tokens):
            parts = []
            for t in tokens:
                parts.append(_pfml_word(t, extra_words[t], language) if t in extra_words else _escape(t) + " ")
            return "".join(parts), True
    return text, False


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _make_dataset(model: LoadedModel, items: list[dict[str, Any]], sources: list[tuple[str, bool]],
                  languages: list[str] | None, skip_unknown: bool):
    import librosa  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from inference.data import AudioTextDataset, _skip  # noqa: PLC0415
    from lib.audio import load_audio  # noqa: PLC0415
    from lib.g2p_encoding import G2PEncodingError, encode_paths  # noqa: PLC0415

    class ItemsDataset(AudioTextDataset):
        """AudioTextDataset with the texts given in memory instead of files next to the audio."""

        def __init__(self):  # noqa: D107  (the parent's file scan is not needed)
            self.vocabulary = model.vocabulary
            self.sample_rate = model.backend.sample_rate
            self.language = languages
            self.oov_handling = "force" if skip_unknown else "discard"
            self.errors: dict[str, str] = {}

        def __len__(self):
            return len(items)

        def __getitem__(self, idx):
            identifier = f"{idx:05d}"
            text, is_pfml = sources[idx]
            if not text:
                self.errors[identifier] = "Empty text"
                return _skip(identifier, "Empty text")
            try:
                if is_pfml:
                    g2p_words = model.pipeline.convert_pfml(text, languages=self.language)
                else:
                    g2p_words = model.pipeline.convert(text, languages=self.language)
            except Exception as e:  # noqa: BLE001
                self.errors[identifier] = f"G2P failed: {e}"
                return _skip(identifier, self.errors[identifier])
            try:
                data, lexicon, texts = encode_paths(g2p_words, self.vocabulary, self.oov_handling,
                                                    languages=self.language)
            except G2PEncodingError as e:
                self.errors[identifier] = f"Unknown phonemes: {e}"
                return _skip(identifier, self.errors[identifier])
            if not data["paths"].any():
                self.errors[identifier] = "Nothing to align (no known words)"
                return _skip(identifier, self.errors[identifier])
            audio, sr = load_audio(Path(items[idx]["audio"]))
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            return {
                "skip": False,
                "identifier": identifier,
                "warning": "",
                "waveform": torch.from_numpy(audio).float(),
                "duration": len(audio) / self.sample_rate,
                **{key: torch.from_numpy(value) for key, value in data.items()},
                "lexicon": lexicon,
                "texts": texts,
            }

    return ItemsDataset()


def _read_textgrid(path: Path) -> tuple[float, dict[str, list[list[Any]]]]:
    import textgrid  # noqa: PLC0415

    tg = textgrid.TextGrid.fromFile(str(path))
    tiers = {}
    for tier in tg:
        tiers[tier.name] = [[float(iv.minTime), float(iv.maxTime), iv.mark] for iv in tier]
    return float(tg.maxTime), tiers


def _with_silence(rows: list[list[Any]], duration: float, min_gap: float = 0.015) -> list[list[Any]]:
    """Gaps (empty intervals and uncovered time) become SP, as in the other aligners' labels.
    A gap of a frame or so (left by an omitted zero-width phoneme) goes to the interval before it."""
    out: list[list[Any]] = []
    t = 0.0
    for start, end, text in rows:
        if out and t + 1e-6 < start < t + min_gap:
            out[-1][1] = start
        elif start > t + 1e-6:
            out.append([t, start, SILENCE])
        out.append([start, end, text or SILENCE])
        t = end
    if duration > t + 1e-6:
        out.append([t, duration, SILENCE])
    merged: list[list[Any]] = []
    for row in out:
        if merged and row[2] == SILENCE and merged[-1][2] == SILENCE:
            merged[-1][1] = row[1]
        else:
            merged.append(row)
    return merged


@rt.method()
def align(model: dict[str, Any], items: list[dict[str, Any]], language: str | None = None,
          extra_languages: list[str] | None = None, g2p: str = "auto", skip_unknown_words: bool = False,
          extra_words: dict[str, list[str]] | None = None, batch_size: int = 4,
          score_unit: str = "levenshtein", skip_penalty: float = 0.5) -> list[dict[str, Any]]:
    import lightning.pytorch as pl  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from inference.callbacks import SaveTextGridCallback, StatisticsCallback  # noqa: PLC0415
    from inference.module import ForcedAlignmentInferenceModule  # noqa: PLC0415

    loaded = _get_model(model)
    lang = loaded.resolve_language(language)
    languages = loaded.languages_for(lang, extra_languages)
    sources = [_source(item, g2p, extra_words, lang) for item in items]
    dataset = _make_dataset(loaded, items, sources, languages, skip_unknown_words)
    total = len(items)

    class Progress(pl.callbacks.ProgressBar):
        """Progress events instead of a console bar; TIFA's warnings (skipped phonemes...) go to the log."""

        done = 0

        def print(self, *args, **kwargs):
            rt.log(" ".join(str(a) for a in args))

        def on_predict_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
            self.done += len(batch.get("identifier", [])) + len(batch.get("warning", []))
            rt.progress(min(1.0, self.done / max(1, total)), "Aligning")

    class Diagnosis(StatisticsCallback):
        def _save(self):  # keep the records in memory, no files and plots
            pass

    device = (os.environ.get("MVT_DEVICE") or "auto").lower()
    accelerator = "cpu" if device == "cpu" or not torch.cuda.is_available() else "gpu"
    with tempfile.TemporaryDirectory(prefix="mvt-tifa-") as tmp:
        diagnosis = Diagnosis(save_dir=Path(tmp) / "statistics")
        trainer = pl.Trainer(
            accelerator=accelerator, devices=1, precision="32-true", logger=False, enable_checkpointing=False,
            enable_progress_bar=True, enable_model_summary=False,
            callbacks=[SaveTextGridCallback(output_dir=tmp, language=lang, timestep=loaded.backend.timestep,
                                            skip_handling="preserve"), diagnosis, Progress()],
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=max(1, batch_size), num_workers=0, shuffle=False,
                                             collate_fn=dataset.collate)
        rt.progress(0.0, "Aligning")
        trainer.predict(ForcedAlignmentInferenceModule(loaded.backend, score_unit=score_unit,
                                                       skip_penalty=skip_penalty), loader)
        records = {r["identifier"]: r for r in diagnosis._metric_records}
        results: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            identifier = f"{index:05d}"
            path = Path(tmp) / f"{identifier}.TextGrid"
            if not path.exists():
                results.append({"ok": False, "error": dataset.errors.get(identifier, "Alignment failed")})
                continue
            duration, tiers = _read_textgrid(path)
            record = records.get(identifier, {})
            diag = {k: float(record[k]) for k in ("agreement", "confidence", "determinacy", "monotonicity")
                    if record.get(k) is not None}
            if record.get("num_skipped_tokens"):
                diag["skipped_phonemes"] = int(record["num_skipped_tokens"])
            result = {
                "ok": True,
                "duration": duration,
                "confidence": diag.get("confidence"),
                "phones": _with_silence([r for r in tiers.get("phones", []) if r[2]], duration),
                "words": _with_silence([r for r in tiers.get("words", []) if r[2]], duration),
                "language": lang,
                "diagnosis": diag,
            }
            texts = [r for r in tiers.get("texts", []) if r[2]]
            if [r[2] for r in texts] != [r[2] for r in tiers.get("words", []) if r[2]]:
                result["texts"] = texts  # written words (e.g. 猫) when they differ from the readings (ne, ko)
            results.append(result)
    rt.progress(1.0, "Aligned")
    return results


@rt.method()
def g2p(model: dict[str, Any], words: list[str], language: str | None = None) -> dict[str, list[str] | None]:
    """Phonemes of each word (first pronunciation; None when the G2P cannot convert it)."""
    loaded = _get_model(model)
    lang = loaded.resolve_language(language)
    out: dict[str, list[str] | None] = {}
    for word in words:
        try:
            converted = loaded.pipeline.convert_pfml(_pfml_word(word), languages=[lang] if lang else None)
            out[word] = [_strip(p, lang) for w in converted for g in w.readings[0].paths[0] for p in g.phonemes]
        except Exception:  # noqa: BLE001
            out[word] = None
    return out


@rt.method()
def phonemize(model: dict[str, Any], texts: list[str], language: str | None = None,
              extra_languages: list[str] | None = None) -> list[dict[str, Any]]:
    """Text -> words with their pronunciation candidates, as the aligner will see them."""
    loaded = _get_model(model)
    lang = loaded.resolve_language(language)
    languages = loaded.languages_for(lang, extra_languages)
    out = []
    for text in texts:
        try:
            words = loaded.pipeline.convert(text, languages=languages)
        except Exception as e:  # noqa: BLE001
            out.append({"ok": False, "error": str(e)})
            continue
        out.append({"ok": True, "words": [
            {"text": w.text, "candidates": [
                {"scripts": [g.script for g in path], "phonemes": [_strip(p, lang) for g in path for p in g.phonemes]}
                for reading in w.readings for path in reading.paths
            ]} for w in words
        ]})
    return out


def _strip(phone: str, lang: str | None) -> str:
    prefix = f"{lang}/" if lang else None
    return phone[len(prefix):] if prefix and phone.startswith(prefix) else phone


@rt.method()
def model_info(model: dict[str, Any]) -> dict[str, Any]:
    loaded = _get_model(model)
    return {"languages": loaded.languages, "sample_rate": loaded.backend.sample_rate,
            "timestep": loaded.backend.timestep}


if __name__ == "__main__":
    rt.run()

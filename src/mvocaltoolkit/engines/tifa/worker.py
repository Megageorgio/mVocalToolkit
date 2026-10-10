"""TIFA engine worker (PyTorch inference). Runs with the TIFA source code on sys.path.

TIFA does its own G2P (g2pflow pipeline from the model's config.yaml): plain text goes in as it is,
fixed words and known phonemes go in as PFML.
"""

from __future__ import annotations

import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import mvt_engine as rt
from mvt_pieces import cut_points as _cut_points
from mvt_pieces import silences as _silences
from mvt_pieces import with_silence as _with_silence

warnings.filterwarnings("ignore")

_models: dict[str, "LoadedModel"] = {}
SILENCE = "SP"
BREATH = "AP"
# an optional breath the aligner did not hear is squeezed to a frame or so; shorter ones are not kept
BREATH_MIN = 0.06
BREATH_PUNCTUATION = set(",.!?;:…—–-")


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


def _breath_word() -> str:
    """A breath the aligner may place or leave out: two pronunciations, AP or nothing."""
    return (f'<word text="{BREATH}"><path><phoneme>{BREATH}</phoneme></path>'
            '<path><group phonemes=""/></path></word>')


def _source(item: dict[str, Any], g2p: str, extra_words: dict[str, list[str]] | None,
            language: str | None, breaths: bool = False) -> tuple[str, bool]:
    """(text, is_pfml) for one item: plain text, fixed words, or known phonemes.
    breaths: an optional breath at the start, at SP marks and after punctuation."""
    lead = _breath_word() if breaths else ""
    if item.get("phonemes") or item.get("words"):
        phonemes = bool(item.get("phonemes"))
        extra = extra_words or {}
        parts = [lead]
        for token in item["phonemes"] if phonemes else item["words"]:
            if not token:
                continue
            if token == SILENCE:
                if breaths and parts[-1] != lead:
                    parts.append(lead)
                continue
            fixed = [token] if phonemes or g2p == "none" else extra.get(token)
            parts.append(_pfml_word(token, fixed, language))
        return "".join(parts), True
    text = (item.get("text") or "").strip()
    tokens = text.split()
    extra = extra_words or {}
    if any(_known(t, extra) for t in tokens) or (breaths and text):
        # user words and G2P guesses in a plain text: fixed as words with the given phonemes, the rest goes
        # through the model's G2P
        parts = [lead]
        for t in tokens:
            fixed = _known(t, extra)
            parts.append(_pfml_word(_bare(t), fixed, language) if fixed else _escape(t) + " ")
            if breaths and t[-1] in BREATH_PUNCTUATION:
                parts.append(lead)
        return "".join(parts), True
    return text, False


def _bare(token: str) -> str:
    """A word of a plain text as it is looked up: lower case, without the punctuation around it."""
    return token.lower().strip(".,!?;:…—–\"'«»()[]-")


def _known(token: str, extra: dict[str, list[str]]) -> list[str] | None:
    return extra.get(token) or extra.get(_bare(token))


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
            self.unknown: dict[str, list[str]] = {}

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
                # a word the dictionary lacks goes through as its own spelling
                word = re.search(r"in word '(.+?)'", str(e))
                if word:
                    self.unknown[identifier] = [word.group(1)]
                    self.errors[identifier] = f"Unknown words: {word.group(1)}"
                else:
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


@rt.method()
def align(model: dict[str, Any], items: list[dict[str, Any]], language: str | None = None,
          extra_languages: list[str] | None = None, g2p: str = "auto", skip_unknown_words: bool = False,
          extra_words: dict[str, list[str]] | None = None, batch_size: int = 4,
          score_unit: str = "levenshtein", skip_penalty: float = 0.5, optional_breaths: bool = False,
          split_silence: bool = False, split_max_length: float = 25.0,
          split_min_silence: float = 0.3) -> list[dict[str, Any]]:
    loaded = _get_model(model)
    lang = loaded.resolve_language(language)
    languages = loaded.languages_for(lang, extra_languages)
    if optional_breaths and BREATH not in getattr(loaded.vocabulary, "symbol_to_id", {}):
        rt.log(f"optional breaths: the model has no {BREATH} phoneme, the option is ignored")
        optional_breaths = False
    sources = [_source(item, g2p, extra_words, lang, optional_breaths) for item in items]
    run = dict(loaded=loaded, languages=languages, lang=lang, skip_unknown=skip_unknown_words,
               batch_size=batch_size, score_unit=score_unit, skip_penalty=skip_penalty, breaths=optional_breaths)
    results = _predict(items=items, sources=sources, progress=(0.0, 0.5 if split_silence else 1.0), **run)
    if split_silence:
        results = _realign_in_pieces(items, results, run, split_max_length, split_min_silence)
    rt.progress(1.0, "Aligned")
    return results


def _predict(loaded: LoadedModel, items: list[dict[str, Any]], sources: list[tuple[str, bool]],
             languages: list[str] | None, lang: str | None, skip_unknown: bool, batch_size: int, score_unit: str,
             skip_penalty: float, breaths: bool, progress: tuple[float, float]) -> list[dict[str, Any]]:
    import lightning.pytorch as pl  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from inference.callbacks import SaveTextGridCallback, StatisticsCallback  # noqa: PLC0415
    from inference.module import ForcedAlignmentInferenceModule  # noqa: PLC0415

    dataset = _make_dataset(loaded, items, sources, languages, skip_unknown)
    total = len(items)
    low, high = progress

    class Progress(pl.callbacks.ProgressBar):
        """Progress events instead of a console bar; TIFA's warnings (skipped phonemes...) go to the log."""

        done = 0

        def print(self, *args, **kwargs):
            rt.log(" ".join(str(a) for a in args))

        def on_predict_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
            self.done += len(batch.get("identifier", [])) + len(batch.get("warning", []))
            rt.progress(low + (high - low) * min(1.0, self.done / max(1, total)), "Aligning")

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
        rt.progress(low, "Aligning")
        trainer.predict(ForcedAlignmentInferenceModule(loaded.backend, score_unit=score_unit,
                                                       skip_penalty=skip_penalty), loader)
        records = {r["identifier"]: r for r in diagnosis._metric_records}
        results: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            identifier = f"{index:05d}"
            path = Path(tmp) / f"{identifier}.TextGrid"
            if not path.exists():
                failed: dict[str, Any] = {"ok": False, "error": dataset.errors.get(identifier, "Alignment failed")}
                if identifier in dataset.unknown:
                    failed["unknown_words"] = dataset.unknown[identifier]
                results.append(failed)
                continue
            duration, tiers = _read_textgrid(path)
            record = records.get(identifier, {})
            diag = {k: float(record[k]) for k in ("agreement", "confidence", "determinacy", "monotonicity")
                    if record.get(k) is not None}
            if record.get("num_skipped_tokens"):
                diag["skipped_phonemes"] = int(record["num_skipped_tokens"])
            phones = [r for r in tiers.get("phones", []) if r[2]]
            words = [r for r in tiers.get("words", []) if r[2]]
            if breaths:
                phones, words = _drop_unheard_breaths(phones), _drop_unheard_breaths(words, join=False)
            result = {
                "ok": True,
                "duration": duration,
                "confidence": diag.get("confidence"),
                "phones": _with_silence(phones, duration),
                "words": _with_silence(words, duration),
                "language": lang,
                "diagnosis": diag,
            }
            texts = [r for r in tiers.get("texts", []) if r[2]]
            if [r[2] for r in texts] != [r[2] for r in tiers.get("words", []) if r[2]]:
                result["texts"] = texts  # written words (e.g. 猫) when they differ from the readings (ne, ko)
            results.append(result)
    return results


def _drop_unheard_breaths(rows: list[list[Any]], join: bool = True) -> list[list[Any]]:
    """Optional breaths the aligner squeezed to almost nothing: the time goes to the interval before
    (join) or becomes a gap (words tier, filled with SP later)."""
    out: list[list[Any]] = []
    for row in rows:
        if row[2] == BREATH and row[1] - row[0] < BREATH_MIN:
            if join and out and abs(out[-1][1] - row[0]) < 1e-6:
                out[-1] = [out[-1][0], row[1], out[-1][2]]
            continue
        out.append(list(row))
    return out


def _piece_source(result: dict[str, Any], start: float, end: float, lang: str | None,
                  breaths: bool) -> tuple[str, bool] | None:
    """The words of the first pass inside [start, end), with the pronunciation it chose (None: no words)."""
    phones = [r for r in result["phones"] if r[2] != SILENCE]
    breath = _breath_word()
    parts = [breath] if breaths else []
    words = 0
    for ws, we, word in result["words"]:
        if word == SILENCE or not start <= (ws + we) / 2 < end:
            continue
        if breaths and word == BREATH:
            if parts and parts[-1] != breath:
                parts.append(breath)
            continue
        symbols = [p for s, e, p in phones if ws - 1e-6 <= (s + e) / 2 <= we + 1e-6]
        if not symbols:
            continue
        inner = "".join(
            f"<phoneme symbol={quoteattr(p)}/>" if "/" in p or not lang or p in (BREATH, SILENCE)
            else f"<phoneme language={quoteattr(lang)} symbol={quoteattr(p)}/>" for p in symbols)
        parts.append(f"<word text={quoteattr(word)}>{inner}</word>")
        words += 1
    return ("".join(parts), True) if words else None


def _move_texts(texts: list[list[Any]], old_words: list[list[Any]],
                new_words: list[list[Any]]) -> list[list[Any]] | None:
    """The written-words tier follows the words it covers to their new times (None: the words differ)."""
    def spoken(rows):
        return [r for r in rows if r[2] not in (SILENCE, BREATH)]

    old, new = spoken(old_words), spoken(new_words)
    if len(old) != len(new):  # same words in the same order; only the labels of readings may be spelled differently
        return None
    out = []
    for start, end, text in texts:
        covered = [i for i, (s, e, _) in enumerate(old) if start - 1e-6 <= (s + e) / 2 <= end + 1e-6]
        if covered:
            out.append([new[covered[0]][0], new[covered[-1]][1], text])
    return out


def _realign_in_pieces(items: list[dict[str, Any]], results: list[dict[str, Any]], run: dict[str, Any],
                       max_length: float, min_silence: float) -> list[dict[str, Any]]:
    """Long files: cut at clear silences into pieces of up to max_length seconds and align each piece again
    with its own words (TIFA is most accurate on phrase-long audio). The first pass decides which words go
    to which piece, so a cut never falls inside a word."""
    import librosa  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="mvt-tifa-split-") as tmp:
        plan = []  # (item index, [(start, end, piece index or None)])
        pieces: list[dict[str, Any]] = []
        sources: list[tuple[str, bool]] = []
        for index, (item, result) in enumerate(zip(items, results)):
            if not result.get("ok") or result["duration"] <= max_length:
                continue
            audio, sr = librosa.load(item["audio"], sr=None, mono=True)
            duration = len(audio) / sr
            cuts = _cut_points(duration, _silences(audio, sr, min_silence), result["phones"], max_length)
            if not cuts:
                continue
            bounds = [0.0, *cuts, duration]
            spans = []
            for k in range(len(bounds) - 1):
                a, b = bounds[k], bounds[k + 1]
                source = _piece_source(result, a, b, run["lang"], run["breaths"])
                piece = None
                if source is not None:
                    path = Path(tmp) / f"{index:05d}_{k:03d}.wav"
                    sf.write(str(path), audio[int(round(a * sr)):int(round(b * sr))], sr)
                    piece = len(pieces)
                    pieces.append({"audio": str(path), "name": path.stem})
                    sources.append(source)
                spans.append((a, b, piece))
            plan.append((index, spans))
        if not pieces:
            return results
        realigned = _predict(items=pieces, sources=sources, progress=(0.5, 1.0), **run)
    for index, spans in plan:
        failed = [realigned[p] for _, _, p in spans if p is not None and not realigned[p].get("ok")]
        if failed:
            rt.log(f"{items[index].get('name') or index}: a piece failed ({failed[0].get('error')}), "
                   "the whole-file alignment is kept")
            continue
        phones, words, scores = [], [], []
        for a, b, p in spans:
            if p is None:
                continue
            res = realigned[p]
            for tier, rows in (("phones", phones), ("words", words)):
                rows += [[a + s, min(a + e, b), text] for s, e, text in res[tier] if a + s < b - 1e-6]
            if res.get("confidence") is not None:
                scores.append(res["confidence"])
        result = results[index]
        words = [r for r in words if r[2] != SILENCE]
        if result.get("texts"):
            texts = _move_texts(result["texts"], result["words"], words)
            if texts is None:
                rt.log(f"{items[index].get('name') or index}: the written words could not be matched after the "
                       "split, the whole-file alignment is kept")
                continue
            result["texts"] = texts
        result["phones"] = _with_silence([r for r in phones if r[2] != SILENCE], result["duration"])
        result["words"] = _with_silence(words, result["duration"])
        result["diagnosis"] = {**result.get("diagnosis", {}), "pieces": len(spans)}
        if scores:
            result["confidence"] = result["diagnosis"]["confidence"] = sum(scores) / len(scores)
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

"""Transcription and forced alignment.

transcribe:  audio -> text (WhisperX: batched, VAD) -> tokens (text frontend)
align:       audio + text|words|phonemes (or transcribed) -> phoneme/word label (SOFA, HubertFA or TIFA) -> rules -> files
"""

from __future__ import annotations

from typing import Any

from ..api_models import AlignRequest, ItemResult, TranscribeOptions, TranscribeRequest
from ..jobs import Job
from ..labels import Interval, Label
from ..text import normalize as text_normalize
from ..text.g2p import fill_unknown, load_user_words
from ..text.rules import apply_rule_sets
from ..toolkit import Toolkit
from .io import Item, OutputWriter, resolve_inputs

CHUNK = 32  # items per engine request (progress & cancellation granularity)
ALIGN_ENGINES = ("sofa", "hubertfa", "tifa")
# aligners that take the text as it is and do their own G2P (no text frontend, no dictionary of ours)
OWN_G2P_ENGINES = ("tifa",)


def _sub_progress(job: Job, start: float, end: float, stage: str):
    def report(value: float | None, message: str = "", detail: dict[str, Any] | None = None) -> None:
        if value is None:
            job.progress(None, stage=stage, message=message, detail=detail)
        else:
            job.progress(start + (end - start) * max(0.0, min(1.0, value)), stage=stage, message=message, detail=detail)

    return report


async def text_frontend(tk: Toolkit, job: Job, language: str | None, texts: list[str]) -> list[list[str]]:
    name = text_normalize.frontend_name(language)
    if name in text_normalize.ENGINE_FRONTENDS:
        result = await tk.engines.call("textfront", "normalize", {"language": name, "texts": texts}, log=job.log)
        return [list(tokens) for tokens in result]
    return [text_normalize.normalize(text, name) for text in texts]


def _whisper_model(tk: Toolkit, model: str) -> tuple[str, dict[str, Any]]:
    """Catalog id (engine-managed whisper model) or a raw faster-whisper model name / path."""
    entry = tk.catalog.get(model)
    if entry is not None and entry.engine == "whisperx":
        return str(entry.params.get("model", model)), dict(entry.params)
    return model, {}


async def transcribe_items(
    tk: Toolkit, job: Job, items: list[Item], options: TranscribeOptions, language: str | None, report
) -> None:
    model_name, params = _whisper_model(tk, options.model)
    lang = options.language or language
    lang = text_normalize.frontend_name(lang) if lang else None
    total = len(items)
    for start in range(0, total, CHUNK):
        job.check_cancelled()
        chunk = items[start : start + CHUNK]

        def on_progress(data: dict[str, Any], offset=start, size=len(chunk)) -> None:

            if data.get("stage") == "install":  # first use of the engine

                job.progress(None, stage="install", message=data.get("message"))

                return
            inner = float(data.get("progress", 0.0))
            report((offset + inner * size) / total, data.get("message", "Transcribing"),
                   {"items_done": int(offset + inner * size), "items_total": total})

        result = await tk.engines.call(
            "whisperx",
            "transcribe",
            {
                "items": [{"audio": str(i.audio), "name": i.name} for i in chunk],
                "model": model_name,
                "language": lang,
                "batch_size": options.batch_size,
                "compute_type": options.compute_type,
                "vad": options.vad.model_dump(),
                "initial_prompt": options.initial_prompt,
                "hotwords": options.hotwords,
                "suppress_numerals": options.suppress_numerals,
                "beam_size": options.beam_size,
                "temperature": options.temperature,
                "model_params": params,
            },
            on_progress=on_progress,
            log=job.log,
        )
        for item, res in zip(chunk, result):
            if not res.get("ok", True):
                item.data["error"] = res.get("error", "transcription failed")
                continue
            item.text = res.get("text", "").strip()
            item.data["transcription"] = {
                "language": res.get("language"),
                "segments": res.get("segments", []),
                "low_confidence": res.get("low_confidence", False),
            }
            if not item.language and res.get("language"):
                item.language = res["language"]
        report((start + len(chunk)) / total, "Transcribing")


async def run_transcribe(tk: Toolkit, job: Job, req: TranscribeRequest) -> dict[str, Any]:
    items = resolve_inputs(req.input, tk.home)
    job.progress(0.0, stage="transcribe", message=f"{len(items)} files")
    await transcribe_items(tk, job, items, req, req.language, _sub_progress(job, 0.0, 0.9, "transcribe"))
    ok_items = [i for i in items if "error" not in i.data and i.text is not None]
    if req.frontend and ok_items:
        job.progress(0.9, stage="frontend")
        by_lang: dict[str | None, list[Item]] = {}
        for item in ok_items:
            by_lang.setdefault(item.language or req.language, []).append(item)
        for lang, group in by_lang.items():
            tokens = await text_frontend(tk, job, lang, [i.text or "" for i in group])
            for item, toks in zip(group, tokens):
                item.tokens = toks
    results = []
    for item in items:
        error = item.data.get("error")
        if error:
            job.item_error(item.name, error)
        elif req.save_txt:
            from pathlib import Path  # noqa: PLC0415

            folder = Path(req.output_dir).expanduser() / item.rel_dir if req.output_dir else item.audio.parent
            folder.mkdir(parents=True, exist_ok=True)
            txt = folder / f"{item.name}.txt"
            txt.write_text(" ".join(item.tokens) if item.tokens else (item.text or ""), encoding="utf-8")
            item.data["txt"] = str(txt)
        results.append(
            ItemResult(
                name=item.name,
                audio=str(item.audio),
                ok=not error,
                error=error,
                text=item.text,
                tokens=item.tokens,
                data={k: v for k, v in item.data.items() if k != "error"},
            )
        )
    return {"items": [r.model_dump(exclude_none=True) for r in results]}


def _label_from_engine(res: dict[str, Any]) -> Label:
    label = Label(duration=res.get("duration"), confidence=res.get("confidence"))
    phones = []
    for row in res.get("phones", []):
        start, end, text = row[0], row[1], row[2]
        confidence = row[3] if len(row) > 3 and row[3] is not None else None
        phones.append(Interval(start=float(start), end=float(end), text=str(text),
                               confidence=float(confidence) if confidence is not None else None))
    label.tiers["phones"] = phones
    for tier in ("words", "texts"):
        if res.get(tier):
            label.tiers[tier] = [Interval(start=float(s), end=float(e), text=str(t)) for s, e, t, *_ in res[tier]]
    return label


def _align_params(model, req: AlignRequest, chunk: list[Item], language: str | None,
                  extra_words: dict[str, list[str]] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": {"path": model.path, "layout": model.layout},
        "items": [{"audio": str(i.audio), "name": i.name, "words": i.words, "phonemes": i.phonemes, "text": i.text}
                  for i in chunk],
        "g2p": req.g2p,
        "skip_unknown_words": req.skip_unknown_words,
    }
    words = req.extra_words if extra_words is None else extra_words
    if words:
        params["extra_words"] = words
    if model.engine == "tifa":
        params.update({"language": language, "extra_languages": req.extra_languages})
    elif model.engine == "hubertfa":
        params.update({
            "language": language,
            "non_lexical_phonemes": req.non_lexical_phonemes if req.ap_detector != "none" else [],
            "pad_times": req.pad_times,
            "pad_length": req.pad_length,
            "dictionary": req.dictionary,
        })
    else:
        params.update({"mode": req.mode, "ap_detector": req.ap_detector})
        if req.dictionary:
            params["dictionary"] = req.dictionary
    return params


async def run_align(tk: Toolkit, job: Job, req: AlignRequest) -> dict[str, Any]:
    items = resolve_inputs(req.input, tk.home)
    job.progress(0.0, stage="model", message=f"{len(items)} files")
    model = await tk.models.require(ALIGN_ENGINES, req.model, _sub_progress(job, 0.0, 0.1, "download"))
    language = req.language or model.text_frontend or (model.languages[0] if model.languages else None)

    # 1. transcription of items without text
    need_text = [i for i in items if i.text is None and i.words is None and i.phonemes is None]
    if need_text:
        if req.transcribe is None:
            for item in need_text:
                item.data["error"] = "No text for this audio (transcription is disabled)"
        else:
            await transcribe_items(
                tk, job, need_text, req.transcribe, language, _sub_progress(job, 0.1, 0.4, "transcribe")
            )

    # 2. optional review of the transcriptions by the user
    if req.review_transcription:
        review = [
            {"name": i.name, "audio": str(i.audio), "text": i.text, "words": i.words, "phonemes": i.phonemes}
            for i in items
            if "error" not in i.data
        ]
        edited = await job.pause({"stage": "review_transcription", "items": review})
        by_name = {e.get("name"): e for e in (edited or {}).get("items", [])}
        for item in items:
            change = by_name.get(item.name)
            if not change:
                continue
            if change.get("skip"):
                item.data["error"] = "skipped"
                continue
            for key in ("text", "words", "phonemes"):
                if key in change:
                    setattr(item, key, change[key])

    # 3. text frontend
    job.progress(0.4, stage="frontend")
    pending = [i for i in items if "error" not in i.data and i.words is None and i.phonemes is None
               and model.engine not in OWN_G2P_ENGINES]
    by_lang: dict[str | None, list[Item]] = {}
    for item in pending:
        by_lang.setdefault(item.language or language, []).append(item)
    for lang, group in by_lang.items():
        tokens = await text_frontend(tk, job, lang, [i.text or "" for i in group])
        for item, toks in zip(group, tokens):
            item.words = toks
            if not toks:
                item.data["error"] = "Empty text after normalization"

    # 4. alignment, with the user's own words and G2P guesses for the words the dictionary lacks
    to_align = [i for i in items if "error" not in i.data]
    extra_words = dict(req.extra_words)
    if req.g2p == "auto" or model.engine in OWN_G2P_ENGINES:
        try:
            if model.engine in OWN_G2P_ENGINES:
                extra_words = {**load_user_words(tk.home, model.id), **extra_words}
            else:
                extra_words = await fill_unknown(tk, model, language, [i.words or [] for i in to_align if i.phonemes is None],
                                                 extra_words)
        except Exception as e:  # noqa: BLE001
            job.log(f"own words / G2P: {e}")
    report = _sub_progress(job, 0.45, 0.95, "align")
    writer = OutputWriter(req.output, tk.home, job.id)
    total = max(1, len(to_align))
    for start in range(0, len(to_align), CHUNK):
        job.check_cancelled()
        chunk = to_align[start : start + CHUNK]

        def on_progress(data: dict[str, Any], offset=start, size=len(chunk)) -> None:

            if data.get("stage") == "install":  # first use of the engine

                job.progress(None, stage="install", message=data.get("message"))

                return
            done = offset + float(data.get("progress", 0.0)) * size
            report(done / total, data.get("message", "Aligning"), {"items_done": int(done), "items_total": len(to_align)})

        results = await tk.engines.call(
            model.engine, "align", _align_params(model, req, chunk, language, extra_words), on_progress=on_progress, log=job.log
        )
        for item, res in zip(chunk, results):
            if res.get("unknown_words"):
                item.data["unknown_words"] = res["unknown_words"]
            if not res.get("ok", True):
                item.data["error"] = res.get("error", "alignment failed")
                continue
            if res.get("diagnosis"):
                item.data["diagnosis"] = res["diagnosis"]
            item.data["label"] = _label_from_engine(res)
        report((start + len(chunk)) / total, "Aligning")

    # 5. optional refinement of the boundaries (before the rules: the refiner knows the model's phoneme names)
    if req.refine is not None:
        from .refine import refine_items  # noqa: PLC0415

        await refine_items(tk, job, to_align, req.refine, _sub_progress(job, 0.95, 0.99, "refine"))

    # 6. rules and files
    rule_sets = list(req.postprocess.rule_sets)
    if not rule_sets and req.postprocess.use_model_defaults:
        rule_sets = list(model.defaults.get("rule_sets", []))
    for item in to_align:
        label = item.data.get("label")
        if "error" in item.data or label is None:
            continue
        if rule_sets or req.postprocess.rules:
            label = apply_rule_sets(label, rule_sets, req.postprocess.rules)
        item.data["label"] = label
        item.data["files"] = writer.write(item, label)

    csv_files = writer.finalize()
    results_out = []
    for item in items:
        error = item.data.get("error")
        if error:
            job.item_error(item.name, error)
        results_out.append(
            ItemResult(
                name=item.name,
                audio=str(item.audio),
                ok=not error,
                error=error,
                text=item.text,
                tokens=item.words,
                unknown_words=item.data.get("unknown_words"),
                label=item.data.get("label") if req.output.return_labels else None,
                files=item.data.get("files", {}),
                data={k: item.data[k] for k in ("transcription", "diagnosis", "refine", "refine_error") if k in item.data},
            ).model_dump(exclude_none=True)
        )
    return {"model": model.id, "engine": model.engine, "language": language, "items": results_out,
            "csv": csv_files}

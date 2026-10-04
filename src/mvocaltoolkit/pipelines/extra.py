"""Segmentation without text (WFL-ASR), MIDI extraction (GAME), tempo estimation, text tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..api_models import ItemResult, MidiRequest, SegmentRequest, TempoRequest, TextRequest
from ..jobs import Job
from ..labels import Interval, Label
from ..text.dictionary import Dictionary
from ..text.rules import apply_rule_sets
from ..toolkit import Toolkit
from .io import OutputWriter, resolve_inputs
from .label import CHUNK, _sub_progress, text_frontend


async def run_segment(tk: Toolkit, job: Job, req: SegmentRequest) -> dict[str, Any]:
    items = resolve_inputs(req.input, tk.home)
    model = await tk.models.require("wfl_asr", req.model, _sub_progress(job, 0.0, 0.1, "download"))
    writer = OutputWriter(req.output, tk.home, job.id)
    report = _sub_progress(job, 0.1, 0.95, "segment")
    total = len(items)
    for start in range(0, total, CHUNK):
        job.check_cancelled()
        chunk = items[start : start + CHUNK]

        def on_progress(data: dict[str, Any], offset=start, size=len(chunk)) -> None:
            report((offset + float(data.get("progress", 0.0)) * size) / total, data.get("message", "Segmenting"))

        results = await tk.engines.call(
            "wfl_asr",
            "segment",
            {
                "model": {"path": model.path, "layout": model.layout},
                "items": [{"audio": str(i.audio), "name": i.name, "phonemes": i.phonemes} for i in chunk],
                "lang_id": req.lang_id if req.lang_id is not None else model.params.get("lang_id"),
                "sample": req.sample,
                "top_k": req.top_k,
                "top_p": req.top_p,
                "temperature": req.temperature,
                "confidence_threshold": req.confidence_threshold,
            },
            on_progress=on_progress,
            log=job.log,
        )
        for item, res in zip(chunk, results):
            if not res.get("ok", True):
                item.data["error"] = res.get("error", "segmentation failed")
                continue
            label = Label(tiers={"phones": [Interval(start=s, end=e, text=t) for s, e, t, *_ in res["phones"]]})
            if req.postprocess.rule_sets or req.postprocess.rules:
                label = apply_rule_sets(label, req.postprocess.rule_sets, req.postprocess.rules)
            item.data["label"] = label
            item.data["files"] = writer.write(item, label)
    csv_files = writer.finalize()
    return {"model": model.id, "items": _results(job, items, req.output.return_labels), "csv": csv_files}


async def run_midi(tk: Toolkit, job: Job, req: MidiRequest) -> dict[str, Any]:
    items = resolve_inputs(req.input, tk.home)
    model = await tk.models.require("game", req.model, _sub_progress(job, 0.0, 0.1, "download"))
    tempo_by_item: dict[str, float] = {}
    if req.tempo == "auto":
        tempo = await tk.engines.call(
            "tempo", "tempo", {"items": [{"audio": str(i.audio), "name": i.name} for i in items]}, log=job.log
        )
        tempo_by_item = {r["name"]: r["bpm"] for r in tempo if r.get("ok", True) and r.get("bpm")}
    out_root = Path(req.output_dir).expanduser() if req.output_dir else None
    report = _sub_progress(job, 0.15, 0.95, "midi")
    total = len(items)
    for index, item in enumerate(items):
        job.check_cancelled()
        out_dir = (out_root / item.rel_dir) if out_root else item.audio.parent
        res = await tk.engines.call(
            "game",
            "extract",
            {
                "model": {"path": model.path, "layout": model.layout},
                "audio": str(item.audio),
                "output_dir": str(out_dir),
                "language": req.language or (model.languages[0] if model.languages else None),
                "tempo": tempo_by_item.get(item.name, 120.0) if req.tempo == "auto" else req.tempo,
                "options": {
                    k: v
                    for k, v in {
                        "seg_threshold": req.seg_threshold,
                        "seg_radius": req.seg_radius,
                        "t0": req.t0,
                        "nsteps": req.nsteps,
                        "est_threshold": req.est_threshold,
                        "batch_size": req.batch_size,
                    }.items()
                    if v is not None
                },
                "output_formats": req.output_formats,
                "pitch_format": req.pitch_format,
                "round_pitch": req.round_pitch,
            },
            log=job.log,
        )
        if not res.get("ok", True):
            item.data["error"] = res.get("error", "MIDI extraction failed")
        else:
            notes = [
                Interval(start=n["start"], end=n["end"], text=str(n["note"]), data={"pitch": n.get("pitch")})
                for n in res.get("notes", [])
            ]
            item.data["label"] = Label(tiers={"notes": notes})
            item.data["files"] = res.get("files", {})
            if item.name in tempo_by_item:
                item.data["tempo"] = tempo_by_item[item.name]
        report((index + 1) / total, "Extracting notes")
    return {"model": model.id, "items": _results(job, items, True)}


async def run_tempo(tk: Toolkit, job: Job, req: TempoRequest) -> dict[str, Any]:
    items = resolve_inputs(req.input, tk.home)
    results = await tk.engines.call(
        "tempo", "tempo", {"items": [{"audio": str(i.audio), "name": i.name} for i in items], "model": req.model},
        log=job.log,
    )
    out = []
    for item, res in zip(items, results):
        if not res.get("ok", True):
            job.item_error(item.name, res.get("error", "failed"))
        out.append({"name": item.name, "audio": str(item.audio), **res})
    return {"items": out}


async def run_text(tk: Toolkit, job: Job | None, req: TextRequest, validate_only: bool = False) -> dict[str, Any]:
    """Normalizes texts and converts them to phonemes with the dictionary (and G2P) of a SOFA model."""
    model = None
    if req.model:
        model = await tk.models.require("sofa", req.model)
    language = req.language or (model.text_frontend if model else None) or (
        model.languages[0] if model and model.languages else None
    )
    dummy = job if job is not None else _NullJob()
    tokens = await text_frontend(tk, dummy, language, req.texts)  # type: ignore[arg-type]
    items = []
    dictionary = None
    if model is not None and model.file("dictionary") is not None:
        dictionary = Dictionary.load(model.file("dictionary"))  # type: ignore[arg-type]
    for text, toks in zip(req.texts, tokens):
        entry: dict[str, Any] = {"text": text, "tokens": toks}
        if dictionary is not None:
            entry["unknown_words"] = dictionary.missing(toks)
            if not validate_only:
                entry["phonemes"] = [dictionary.lookup(t) for t in toks]
        items.append(entry)
    unknown = sorted({w for e in items for w in e.get("unknown_words", [])})
    if unknown and req.g2p == "auto" and model is not None and model.layout.get("g2p") and not validate_only:
        guessed = await tk.engines.call(
            "sofa", "g2p", {"model": {"path": model.path, "layout": model.layout}, "words": unknown}
        )
        for entry in items:
            entry["phonemes"] = [
                p if p is not None else guessed.get(t) for p, t in zip(entry["phonemes"], entry["tokens"])
            ]
            entry["guessed"] = {w: guessed.get(w) for w in entry.get("unknown_words", [])}
    return {"language": language, "model": model.id if model else None, "items": items}


class _NullJob:
    def log(self, _message: str) -> None:
        pass


def _results(job: Job, items, return_labels: bool) -> list[dict[str, Any]]:
    out = []
    for item in items:
        error = item.data.get("error")
        if error:
            job.item_error(item.name, error)
        out.append(
            ItemResult(
                name=item.name,
                audio=str(item.audio),
                ok=not error,
                error=error,
                label=item.data.get("label") if return_labels else None,
                files=item.data.get("files", {}),
                data={k: v for k, v in item.data.items() if k not in ("label", "files", "error")},
            ).model_dump(exclude_none=True)
        )
    return out

"""Segmentation without text (WFL-ASR), MIDI extraction (GAME), tempo estimation, vocal separation, pitch
extraction, text tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..api_models import (
    ItemResult,
    MidiRequest,
    PitchRequest,
    SegmentRequest,
    SeparateRequest,
    ResynthRequest,
    TempoRequest,
    TextRequest,
)
from ..jobs import Job
from ..labels import Interval, Label
from ..models.store import InstalledModel, ModelNotFound
from ..text.dictionary import Dictionary
from ..text.rules import apply_rule_sets
from ..toolkit import Toolkit
from .io import OutputWriter, output_folder, resolve_inputs
from .label import ALIGN_ENGINES, CHUNK, _sub_progress, text_frontend


def _lang_id(model: InstalledModel, language: str) -> int | None:
    """The model's number for a language code, from its langs.txt ("ru,0" per line)."""
    root = Path(model.path)
    langs = next(iter(sorted(root.rglob("langs.txt"))), None)
    if langs is None:
        return None
    code = language.lower().split("-")[0].split("_")[0]
    try:
        for line in langs.read_text(encoding="utf-8").splitlines():
            name, _, num = line.strip().partition(",")
            if name.strip().lower() == code and num.strip().isdigit():
                return int(num)
    except OSError:
        return None
    return None


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

            if data.get("stage") == "install":  # first use of the engine

                job.progress(None, stage="install", message=data.get("message"))

                return
            report((offset + float(data.get("progress", 0.0)) * size) / total, data.get("message", "Segmenting"))

        results = await tk.engines.call(
            # models trained with WFL-ASR's refactor branch need its code
            "wfl_asr_v2" if int(model.layout.get("generation", 1)) >= 2 else "wfl_asr",
            "segment",
            {
                "model": {"path": model.path, "layout": model.layout},
                "items": [{"audio": str(i.audio), "name": i.name, "phonemes": i.phonemes} for i in chunk],
                "lang_id": req.lang_id if req.lang_id is not None else (
                    _lang_id(model, req.language) if req.language else model.params.get("lang_id")),
                "decoder": req.decoder,
                "viterbi_bias": req.viterbi_bias,
                "silence_threshold": req.silence_threshold,
                "min_silence_duration": req.min_silence_duration,
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
            item.data["label"] = Label(tiers={"phones": [Interval(start=s, end=e, text=t) for s, e, t, *_ in res["phones"]]})
    if req.refine is not None:
        from .refine import refine_items  # noqa: PLC0415

        await refine_items(tk, job, items, req.refine, _sub_progress(job, 0.95, 0.99, "refine"))
    for item in items:
        label = item.data.get("label")
        if "error" in item.data or label is None:
            continue
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
    report = _sub_progress(job, 0.15, 0.95, "midi")
    total = len(items)
    for index, item in enumerate(items):
        job.check_cancelled()
        out_dir = output_folder(item, req.output_dir, tk.home, job.id)
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


async def _separation_model(tk: Toolkit, job: Job, model_id: str) -> tuple[str, str, str]:
    """-> (audio-separator model file name, folder for its files, id)"""
    if model_id == "auto":
        # the Roformers take minutes per song on a processor; a lighter model is the better default there
        from ..engines.env import _cpu_only  # noqa: PLC0415

        model_id = "separation-vocals-mdx-fast" if _cpu_only(tk.settings.torch_backend) else "separation-vocals-bs-roformer"
    if model_id.startswith("path:") or Path(model_id).expanduser().is_absolute():
        path = Path(model_id.removeprefix("path:")).expanduser()
        return path.name, str(path.parent), str(path)
    entry = tk.catalog.get(model_id)
    if entry is None and tk.models.get_installed(model_id) is None:
        # any model name known to audio-separator (it downloads it itself)
        return model_id, str(tk.home.dir("models/_audio-separator")), model_id
    model = await tk.models.require("separation", model_id, _sub_progress(job, 0.0, 0.05, "download"))
    file_name = model.params.get("model_file") or model.layout.get("checkpoint")
    if not file_name:
        raise ModelNotFound(f"Model {model_id} has no model_file parameter")
    return str(file_name), model.path, model.id


async def run_separate(tk: Toolkit, job: Job, req: SeparateRequest) -> dict[str, Any]:
    items = resolve_inputs(req.input, tk.home)
    model_file, model_dir, model_id = await _separation_model(tk, job, req.model)
    report = _sub_progress(job, 0.05, 0.98, "separate")
    total = len(items)
    for index, item in enumerate(items):
        job.check_cancelled()

        def on_progress(data: dict[str, Any], offset=index) -> None:

            if data.get("stage") == "install":  # first use of the engine

                job.progress(None, stage="install", message=data.get("message"))

                return
            report((offset + float(data.get("progress", 0.0))) / total, data.get("message", "Separating"))

        res = await tk.engines.call(
            "separation",
            "separate",
            {
                "items": [{"audio": str(item.audio), "name": item.name,
                           "output_dir": str(output_folder(item, req.output_dir, tk.home, job.id))}],
                "model_file": model_file,
                "model_dir": model_dir,
                "stems": req.stems,
                "output_format": req.output_format,
                "sample_rate": req.sample_rate,
                "options": req.options,
            },
            on_progress=on_progress,
            log=job.log,
        )
        result = res[0] if res else {"ok": False, "error": "no result"}
        if not result.get("ok", True):
            item.data["error"] = result.get("error", "separation failed")
        else:
            item.data["files"] = result.get("files", {})
        report((index + 1) / total, "Separating")
    return {"model": model_id, "items": _results(job, items, False)}


async def _pitch_model(tk: Toolkit, job: Job, model_id: str) -> tuple[str, str | None, str]:
    """-> (method, RMVPE model path, id)"""
    if model_id in ("fcpe", "parselmouth") and tk.catalog.get(model_id) is None:
        return model_id, None, model_id
    model = await tk.models.require("pitch", model_id, _sub_progress(job, 0.0, 0.05, "download"))
    method = str(model.params.get("method") or "rmvpe")
    checkpoint = model.file("checkpoint") if method == "rmvpe" else None
    if method == "rmvpe" and checkpoint is None:
        raise ModelNotFound(f"No RMVPE checkpoint found in {model.path}")
    return method, str(checkpoint) if checkpoint else None, model.id


def _write_f0(folder: Path, name: str, hop: float, f0: list[float], fmts: list[str]) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for fmt in fmts:
        fmt = fmt.lower().lstrip(".")
        if fmt == "csv":
            path = folder / f"{name}.f0.csv"
            lines = ["time,f0"] + [f"{i * hop:.4f},{v:.3f}" for i, v in enumerate(f0)]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif fmt == "json":
            path = folder / f"{name}.f0.json"
            path.write_text(json.dumps({"hop": hop, "f0": f0}), encoding="utf-8")
        elif fmt == "txt":
            path = folder / f"{name}.f0.txt"
            path.write_text("\n".join(f"{v:.3f}" for v in f0) + "\n", encoding="utf-8")
        else:
            raise ValueError(f"Unknown f0 format {fmt!r} (csv, json, txt)")
        files[fmt] = str(path)
    return files


async def run_pitch(tk: Toolkit, job: Job, req: PitchRequest) -> dict[str, Any]:
    items = resolve_inputs(req.input, tk.home)
    method, model_path, model_id = await _pitch_model(tk, job, req.model)
    report = _sub_progress(job, 0.05, 0.98, "pitch")
    total = len(items)
    for start in range(0, total, CHUNK):
        job.check_cancelled()
        chunk = items[start : start + CHUNK]

        def on_progress(data: dict[str, Any], offset=start, size=len(chunk)) -> None:

            if data.get("stage") == "install":  # first use of the engine

                job.progress(None, stage="install", message=data.get("message"))

                return
            report((offset + float(data.get("progress", 0.0)) * size) / total, data.get("message", "Pitch"))

        results = await tk.engines.call(
            "pitch",
            "pitch",
            {
                "items": [{"audio": str(i.audio), "name": i.name} for i in chunk],
                "method": method,
                "model_path": model_path,
                "hop": req.hop,
                "f0_min": req.f0_min,
                "f0_max": req.f0_max,
                "threshold": req.threshold,
            },
            on_progress=on_progress,
            log=job.log,
        )
        for item, res in zip(chunk, results):
            if not res.get("ok", True):
                item.data["error"] = res.get("error", "pitch extraction failed")
                continue
            f0 = res.get("f0", [])
            if req.output_formats:
                item.data["files"] = _write_f0(output_folder(item, req.output_dir, tk.home, job.id), item.name,
                                               req.hop, f0, req.output_formats)
            if req.return_curve:
                item.data["f0"] = f0
            item.data["hop"] = req.hop
            item.data["voiced_ratio"] = round(sum(1 for v in f0 if v > 0) / len(f0), 3) if f0 else 0.0
    return {"model": model_id, "method": method, "items": _results(job, items, False)}


async def run_resynth(tk: Toolkit, job: Job, req: ResynthRequest) -> dict[str, Any]:
    """The recording (or [start, end] of it) with the given f0: WORLD, or NSF-HiFiGAN (its model downloaded on
    first use). Result: a WAV in the job's output folder."""
    items = resolve_inputs(req.input, tk.home)
    if len(items) != 1:
        raise ValueError("resynthesis works on one recording")
    item = items[0]
    model: dict[str, Any] | None = None
    if req.method == "nsf":
        installed = await tk.models.require("vocoder", req.model, _sub_progress(job, 0.0, 0.3, "download"))
        onnx = installed.file("onnx")
        if onnx is None:
            raise ModelNotFound(f"No ONNX vocoder found in {installed.path}")
        model = {"onnx": str(onnx)}
    folder = tk.home.outputs / job.id
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{item.name}.resynth.{req.method}.wav"
    report = _sub_progress(job, 0.3, 0.98, "resynth")

    def on_progress(data: dict[str, Any]) -> None:
        if data.get("stage") == "install":  # first use of the engine
            job.progress(None, stage="install", message=data.get("message"))
            return
        report(float(data.get("progress", 0.0)), data.get("message", "Resynthesis"))

    res = await tk.engines.call(
        "vocoder", "resynth",
        {"audio": str(item.audio), "out": str(out), "f0": req.f0, "hop": req.hop, "method": req.method,
         "start": req.start, "end": req.end, "model": model},
        on_progress=on_progress, log=job.log,
    )
    return {"method": req.method, "file": res.get("file", str(out)), "sample_rate": res.get("sample_rate"),
            "seconds": res.get("seconds")}


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
    """Normalizes texts and converts them to phonemes with the dictionary (and G2P) of an aligner model."""
    model: InstalledModel | None = None
    if req.model:
        model = await tk.models.require(ALIGN_ENGINES, req.model)
    language = req.language or (model.text_frontend if model else None) or (
        model.languages[0] if model and model.languages else None
    )
    dummy = job if job is not None else _NullJob()
    tokens = await text_frontend(tk, dummy, language, req.texts)  # type: ignore[arg-type]
    items = []
    dictionary = None
    dictionary_path = _dictionary_path(model, language) if model is not None else None
    if dictionary_path is not None:
        dictionary = Dictionary.load(dictionary_path)
    if req.extra_words:
        dictionary = (dictionary or Dictionary({})).with_extra(req.extra_words)
    for text, toks in zip(req.texts, tokens):
        entry: dict[str, Any] = {"text": text, "tokens": toks}
        if dictionary is not None:
            entry["unknown_words"] = dictionary.missing(toks)
            if not validate_only:
                entry["phonemes"] = [dictionary.lookup(t) for t in toks]
        items.append(entry)
    unknown = sorted({w for e in items for w in e.get("unknown_words", [])})
    if (unknown and req.g2p == "auto" and model is not None and model.engine == "sofa" and model.layout.get("g2p")
            and not validate_only):
        guessed = await tk.engines.call(
            "sofa", "g2p", {"model": {"path": model.path, "layout": model.layout}, "words": unknown}
        )
        for entry in items:
            entry["phonemes"] = [
                p if p is not None else guessed.get(t) for p, t in zip(entry["phonemes"], entry["tokens"])
            ]
            entry["guessed"] = {w: guessed.get(w) for w in entry.get("unknown_words", [])}
    return {"language": language, "model": model.id if model else None, "items": items}


def _dictionary_path(model: InstalledModel, language: str | None) -> Path | None:
    if model.engine == "hubertfa":
        dictionaries: dict[str, str] = model.layout.get("dictionaries") or {}
        if not dictionaries:
            return None
        name = dictionaries.get(language or "")
        if name is None and language:
            base = language.lower().split("_")[0].split("-")[0]
            name = next((v for k, v in dictionaries.items() if k.lower().split("_")[0] == base), None)
        if name is None and len(dictionaries) == 1:
            name = next(iter(dictionaries.values()))
        return Path(model.path) / name if name else None
    return model.file("dictionary")


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

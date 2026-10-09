"""Refinement of phoneme boundaries by a boundary refiner model (mRefinerModel): after alignment or segmentation
(the "refine" option of /align and /segment), or of ready labels (/refine)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import formats
from ..api_models import RefineOptions, RefineRequest
from ..jobs import Job
from ..labels import Interval, Label
from ..toolkit import Toolkit
from .io import Item, OutputWriter, resolve_inputs
from .label import CHUNK, _sub_progress

LABEL_SIDECARS = (".lab", ".TextGrid", ".textgrid", ".json")


def apply_refined(label: Label, phones: list[list[Any]]) -> Label:
    """The label with the refiner's phone boundaries; word boundaries that sat on a moved phone boundary move too."""
    old = label.phones
    if len(phones) != len(old):
        return label  # the refiner never adds or removes phonemes; be safe anyway
    moved: dict[float, float] = {}
    new_phones = []
    for iv, (start, end, *_rest) in zip(old, phones):
        if abs(iv.start - start) > 1e-9:
            moved[iv.start] = float(start)
        new_phones.append(iv.model_copy(update={"start": float(start), "end": float(end)}))

    def shift(t: float) -> float:
        for before, after in moved.items():
            if abs(t - before) < 1e-6:
                return after
        return t

    tiers = dict(label.tiers)
    tiers["phones"] = new_phones
    for name, tier in label.tiers.items():
        if name == "phones" or not moved:
            continue
        tiers[name] = [iv.model_copy(update={"start": shift(iv.start), "end": shift(iv.end)}) for iv in tier]
    return label.model_copy(update={"tiers": tiers})


async def refine_items(tk: Toolkit, job: Job, items: list[Item], opts: RefineOptions, report) -> str:
    """Refines item.data["label"] of the items that have one; returns the refiner model id."""
    model = await tk.models.require("refiner", opts.model, _sub_progress(job, 0.0, 0.0, "download"))
    todo = [i for i in items if "error" not in i.data and i.data.get("label") is not None and len(i.data["label"].phones) > 1]
    total = max(1, len(todo))
    for start in range(0, len(todo), CHUNK):
        job.check_cancelled()
        chunk = todo[start : start + CHUNK]

        def on_progress(data: dict[str, Any], offset=start, size=len(chunk)) -> None:
            if data.get("stage") == "install":  # first use of the engine
                job.progress(None, stage="install", message=data.get("message"))
                return
            report((offset + float(data.get("progress", 0.0)) * size) / total, data.get("message", "Refining"))

        results = await tk.engines.call(
            "refiner",
            "refine",
            {
                "model": {"path": model.path, "layout": model.layout},
                "items": [{"audio": str(i.audio), "name": i.name,
                           "phones": [[iv.start, iv.end, iv.text] for iv in i.data["label"].phones]} for i in chunk],
                "mode": opts.mode,
                "phone_map": opts.phone_map,
                "min_confidence": opts.min_confidence,
                "max_shift_ms": opts.max_shift_ms,
            },
            on_progress=on_progress,
            log=job.log,
        )
        for item, res in zip(chunk, results):
            if not res.get("ok", True):
                # the first stage's labels stay; the problem is only reported
                job.log(f"{item.name}: refinement failed: {res.get('error')}")
                item.data["refine_error"] = res.get("error", "refinement failed")
                continue
            item.data["label"] = apply_refined(item.data["label"], res["phones"])
            item.data["refine"] = res.get("info", {})
        report((start + len(chunk)) / total, "Refining")
    return model.id


def _sidecar_label(audio: Path) -> Label | None:
    for ext in LABEL_SIDECARS:
        path = audio.with_suffix(ext)
        if path.is_file():
            return formats.read(path)
    return None


async def run_refine(tk: Toolkit, job: Job, req: RefineRequest) -> dict[str, Any]:
    from .extra import _results  # noqa: PLC0415

    items = resolve_inputs(req.input, tk.home)
    job.progress(0.0, stage="model", message=f"{len(items)} files")
    for item in items:
        try:
            label = Label.from_segments(list(item.segments)) if item.segments else _sidecar_label(item.audio)
        except Exception as e:  # noqa: BLE001
            item.data["error"] = f"Can't read the labels: {e}"
            continue
        if label is None or not label.phones:
            item.data["error"] = "No labels to refine next to this audio"
            continue
        item.data["label"] = label
    model_id = await refine_items(tk, job, items, req, _sub_progress(job, 0.05, 0.95, "refine"))
    writer = OutputWriter(req.output, tk.home, job.id)
    for item in items:
        if "error" not in item.data and item.data.get("label") is not None:
            item.data["files"] = writer.write(item, item.data["label"])
    csv_files = writer.finalize()
    return {"model": model_id, "items": _results(job, items, req.output.return_labels), "csv": csv_files}

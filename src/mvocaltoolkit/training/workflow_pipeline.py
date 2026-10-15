"""Optional source transforms, always in separate folders, with original labels retained."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import hashlib

from .. import formats
from ..api_models import AlignRequest, InputSpec, OutputSpec, SeparateRequest
from .dataset import AUDIO, _csv_rows, _label_for, _label_index, _ds_for
from .preparation_cache import digest


async def preprocess(tk, job, source, out: Path, options: dict, exclusions: list[str], call) -> str:
    from . import preparation_cache as cache
    if not (options.get("separation") or options.get("alignment")) and not exclusions:
        return source.folder
    from ..pipelines.extra import run_separate
    from ..pipelines.label import run_align
    class CancellableEngines:
        async def call(self, engine, method, params, **kwargs):
            return await call(tk, job, engine, method, params)
        def __getattr__(self, name):
            return getattr(tk.engines, name)
    class PipelineToolkit:
        engines = CancellableEngines()
        def __getattr__(self, name):
            return getattr(tk, name)
    pipeline_tk = PipelineToolkit()
    root = Path(source.folder).expanduser().resolve()
    excluded = {Path(p).expanduser().resolve() for p in exclusions}
    index = _label_index(root)
    out.mkdir(parents=True, exist_ok=False)
    for audio in sorted(root.rglob("*") if source.recursive else root.glob("*")):
        if not audio.is_file() or audio.suffix.lower() not in AUDIO or audio.resolve() in excluded:
            continue
        job.check_cancelled()
        name = audio.stem + "-" + digest_name(audio.relative_to(root))
        label = _label_for(audio, _csv_rows(audio.parent) or _csv_rows(root), index)
        ds = _ds_for(audio, index)
        text = audio.with_suffix(".txt")
        stamp = dict(audio=digest(audio), label=label.model_dump() if label else None, options=options,
            name=str(audio.relative_to(root)), ds=digest(ds) if ds else None, text=digest(text) if text.is_file() else None,
            implementation=digest(Path(__file__)))
        identifiers = [p["model"] for p in options.get("separation", [])] + ([options["alignment"]["model"]] if options.get("alignment") else [])
        engine_names = {"separation", "sofa", "tifa", "hubertfa", "whisperx"}
        stamp.update(cache.model_stamp(tk, identifiers, engine_names))
        key = hashlib.sha256(json.dumps(stamp, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cache_root = tk.home.cache / "source-pipeline"
        if options.get("cache", "use") == "use" and cache.restore(cache_root, key, out, root):
            continue
        target = out / (name + audio.suffix.lower())
        shutil.copy2(audio, target)
        if label:
            formats.write(label, target.with_suffix(".json"), "json")
        if ds:
            shutil.copy2(ds, target.with_suffix(".ds"))
        if text.is_file():
            shutil.copy2(text, target.with_suffix(".txt"))
        elif label is None and audio.with_suffix(".lab").is_file():
            shutil.copy2(audio.with_suffix(".lab"), target.with_suffix(".txt"))
        for n, separation in enumerate(options.get("separation", [])):
            req = SeparateRequest(input=InputSpec(items=[{"path": str(target)}]), model=separation["model"],
                output_dir=str(out / f"stems-{name}-{n}"), stems=separation.get("stems"), output_format="wav",
                options=separation.get("options", {}))
            result = await run_separate(pipeline_tk, job, req)
            item = result["items"][0]
            if not item.get("ok", True):
                raise ValueError(f"Separation failed: {item.get('error')}")
            files = item.get("files", {})
            wanted = separation.get("stem", "vocals").lower()
            selected = next((v for k, v in files.items() if k.lower() == wanted), None)
            if selected is None:
                raise ValueError(f"Choose one of the separation stems {list(files)}, requested {wanted}")
            replaced = out / (name + f"-pass{n}.wav")
            shutil.copy2(selected, replaced)
            for suffix in (".json", ".txt", ".ds"):
                if target.with_suffix(suffix).is_file():
                    shutil.copy2(target.with_suffix(suffix), replaced.with_suffix(suffix))
                    target.with_suffix(suffix).unlink()
            target.unlink()
            target = replaced
        alignment = options.get("alignment")
        if alignment and (not label or alignment.get("replace", False)):
            params = {k: v for k, v in alignment.items() if k not in {"replace", "input", "output", "language"}}
            req = AlignRequest(input=InputSpec(items=[{"path": str(target)}]), language=source.language,
                output=OutputSpec(formats=["json"], dir=str(out), layout="beside", return_labels=True), **params)
            result = await run_align(pipeline_tk, job, req)
            item = result["items"][0]
            if not item.get("ok", True):
                raise ValueError(f"Alignment failed: {item.get('error')}")
        if options.get("cache", "use") != "off":
            stamp.update(cache.model_stamp(tk, identifiers, engine_names))
            key = hashlib.sha256(json.dumps(stamp, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            paths = [target] + [p for ext in (".txt", ".ds", ".json") if (p := target.with_suffix(ext)).is_file()]
            cache.store(cache_root, key, out, paths, dict(audio=target.name), root)
    # Stem output trees contain audio too. Keep only final audio in the import root.
    # Move intermediate stems out of the recursively scanned source tree.
    for stem_dir in list(out.glob("stems-*")):
        storage = out.parent / (out.name + "-intermediates")
        storage.mkdir(exist_ok=True)
        shutil.move(str(stem_dir), str(storage / stem_dir.name))
    return str(out)


def digest_name(path):
    import hashlib
    return hashlib.sha256(str(path).encode()).hexdigest()[:10]


async def audio_checks(tk, job, result, options, call):
    for record in result.recordings:
        job.check_cancelled()
        report = await call(tk, job, "audio", "inspect_audio", {"audio": record["prepared"], "options": options})
        record["signal"] = report["metrics"]
        result.problems.extend(dict(file=record["file"], speaker=record["speaker"], language=record["language"], **p) for p in report["problems"])
    path = Path(result.out_dir) / "dataset.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report.update(recordings=result.recordings, problems=result.problems)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

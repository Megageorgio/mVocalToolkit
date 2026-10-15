"""Persisted, optional bank stages shared by the wizard and one-click prototype."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field

from ..api_models import DatasetSource, RecordingPreparationRequest
from .dataset import SourceSpec, build_dataset, spec_from_result, validate_report
from .diffsinger import build_overlay, resolve_assets
from .runs import RunSpec, ACTIVE
from .openutau import VoicebankSpec, assemble
from .phonemizers import default_profile


class BankWorkflowRequest(BaseModel):
    model_config = {"extra": "forbid"}
    project_dir: str
    name: str
    sources: list[DatasetSource] = Field(min_length=1)
    prototype: bool = True
    prototype_steps: int = Field(200, ge=1, le=100000)
    prepare: bool = True
    preparation: dict = Field(default_factory=dict)
    # Optional separation passes, then alignment/transcription, then refinement.
    pipeline: dict = Field(default_factory=dict)
    dictionaries: dict[str, str] = Field(default_factory=dict)
    phonemizer_profiles: dict[str, str] = Field(default_factory=dict)
    extra_phonemes: list[str] = Field(default_factory=list)
    merged_phoneme_groups: list[list[str]] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    checks: dict = Field(default_factory=dict)
    acoustic: bool = True
    variance: bool = True
    acoustic_export: str = ""
    variance_export: str = ""
    training: dict = Field(default_factory=dict)
    acoustic_settings: dict = Field(default_factory=dict)
    variance_settings: dict = Field(default_factory=dict)
    device: str = "auto"
    source: str = "main"
    export: bool = True
    render_ds: str = ""
    vocoder: str = ""
    review_before_training: bool = False
    render_options: dict = Field(default_factory=dict)
    auto_test_phrase: bool = True
    render_test_phrase: bool = False
    test_phrase: dict = Field(default_factory=dict)
    vocoder_training: dict | None = None
    g2p_training: list[dict] = Field(default_factory=list)


def training_config(req, kind, data):
    settings = req.acoustic_settings if kind == "acoustic" else req.variance_settings
    config = {**data, "preset": {"goal": "check" if req.prototype else "full", "size": "small", "gpu_gb": 8},
        "variance": {"dur": True, "pitch": True} if kind == "variance" else {}, **req.training, **settings, "source": req.source}
    config["preset"] = {"goal": "check" if req.prototype else "full", "size": "small", "gpu_gb": 8, **config.get("preset", {})}
    if req.prototype:
        config["preset"]["goal"] = "check"
        config["overrides"] = {**config.get("overrides", {}), "max_updates": req.prototype_steps,
            "val_check_interval": min(req.prototype_steps, int(config.get("overrides", {}).get("val_check_interval", 100)))}
    return config


def check_workflow(req: BankWorkflowRequest) -> dict:
    root = Path(req.project_dir).expanduser().resolve()
    if not req.name.strip() or any(c in req.name for c in '/\\:*?"<>|') or req.name in {".", ".."}:
        raise ValueError("Choose a bank name without path separators")
    if req.device not in {"auto", "cpu", "cuda", "mps"} and not (req.device.startswith("cuda:") and req.device[5:].isdigit()):
        raise ValueError("Choose auto, cpu, cuda:N or mps")
    if req.source not in {"main", "benchmark"}:
        raise ValueError("Source must be main or benchmark")
    from ..api_models import AlignRequest, InputSpec, SeparateRequest
    sample = InputSpec(items=[{"path": req.sources[0].folder}])
    if set(req.pipeline) - {"alignment", "separation", "cache"}:
        raise ValueError("Unknown pipeline option")
    if req.pipeline.get("cache", "use") not in {"use", "refresh", "off"}:
        raise ValueError("Pipeline cache must be use, refresh or off")
    for separation in req.pipeline.get("separation", []):
        SeparateRequest(input=sample, output_dir=str(root / "_check"), **{k: v for k, v in separation.items() if k != "stem"})
    if alignment := req.pipeline.get("alignment"):
        if alignment.get("review_transcription"):
            raise ValueError("Review transcription in the labelling tool first; the bank workflow supports review_before_training")
        AlignRequest(input=sample, **{k: v for k, v in alignment.items() if k not in {"replace", "input", "output", "language"}})
    for src in req.sources:
        path = Path(src.folder).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Recording folder does not exist: {path}")
        if root == path or path in root.parents:
            raise ValueError("The project output must be outside the recording source folders")
        if not src.language or not src.speaker:
            raise ValueError("Every source needs a primary language and singer")
        if req.prepare:
            RecordingPreparationRequest(**{**req.preparation, "folder": str(path), "out_dir": str(root / "workflows" / "_check"), "language": src.language})
    if req.export:
        for kind, enabled, folder in [("acoustic", req.acoustic, req.acoustic_export), ("variance", req.variance, req.variance_export)]:
            if not enabled and (kind == "acoustic" or folder):
                if not folder.strip() or not (Path(folder) / "dsconfig.yaml").is_file():
                    raise ValueError(f"Select an existing {kind} ONNX export with dsconfig.yaml when its training is disabled")
    if req.render_ds and not Path(req.render_ds).is_file():
        raise ValueError("Test DS does not exist")
    if (req.render_ds or req.render_test_phrase) and not req.export:
        raise ValueError("Enable bank export/assembly before requesting a test render")
    if req.render_test_phrase and not req.auto_test_phrase and not req.render_ds:
        raise ValueError("Enable test phrase generation or select a test DS before requesting a render")
    if req.export and req.auto_test_phrase and not req.render_ds:
        from .test_phrases import PhraseRequest, note_midi
        phrase = PhraseRequest.model_validate({**req.test_phrase, "model_dir": "", "run_id": "", "config": "", "language": req.sources[0].language})
        if phrase.mode not in {"cv", "vc", "v", "cvc", "manual"}:
            raise ValueError("Phrase mode must be cv, vc, v, cvc or manual")
        if not phrase.notes.split(): raise ValueError("Specify at least one test note")
        for note in phrase.notes.split(): note_midi(note)
        if phrase.mode == "manual" and not 1 <= len(phrase.phonemes.replace("|", " ").split()) <= 512:
            raise ValueError("Enter 1–512 test phonemes; use | or new lines between syllables")
    if req.export and req.vocoder_training is None:
        if req.vocoder and not (Path(req.vocoder) / "vocoder.yaml").is_file():
            raise ValueError("Select a local ONNX vocoder folder with vocoder.yaml")
        if (req.render_ds or req.render_test_phrase) and not req.vocoder:
            raise ValueError("Test rendering needs a local ONNX vocoder or vocoder training in this chain")
    if req.vocoder_training is not None:
        from .vocoder import build_overlay as vocoder_overlay
        vocoder_overlay({**req.vocoder_training, "device": req.device})
    for settings in req.g2p_training:
        steps = settings.get("max_steps", 0)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 10_000_000:
            raise ValueError("G2P max_steps must be a whole number between 0 and 10000000")
    forbidden = {"datasets", "dictionaries", "data_report", "overlay", "binary_data_dir"}
    if any(forbidden & set(v) for v in (req.training, req.acoustic_settings, req.variance_settings)):
        raise ValueError("Workflow data paths are managed by its immutable dataset snapshot")
    protected = forbidden | {"work_dir", "base_config"}
    if any(protected & set(v.get("overrides", {})) for v in (req.training, req.acoustic_settings, req.variance_settings)):
        raise ValueError("Dataset/workspace paths cannot be replaced through overrides")
    data = dict(datasets=[dict(raw_data_dir=s.folder, speaker=s.speaker, language=s.language) for s in req.sources],
        dictionaries=req.dictionaries or {s.language: "_generated_dictionary" for s in req.sources})
    overlays = {}
    for kind, enabled in [("acoustic", req.acoustic), ("variance", req.variance)]:
        if enabled:
            overlays[kind] = build_overlay("diffsinger_" + kind, {**training_config(req, kind, data), "device": req.device})
    from .workflow_plan import describe_workflow
    return dict(ok=True, stages=[s for s, on in [("prepare", req.prepare), ("dataset", True), ("acoustic", req.acoustic),
        ("variance", req.variance), ("export", req.export), ("bank_render", bool(req.render_ds or req.render_test_phrase))] if on],
        **describe_workflow(req, overlays),
        prototype=req.prototype, warning="A prototype checks the pipeline; short training does not establish voice quality" if req.prototype else "")


class WorkflowJob:
    def __init__(self, manager, state):
        self.manager, self.state = manager, state
        self.id = state["id"]
    @property
    def cancelled(self):
        return self.state["status"] in {"stopped", "stopping"}
    def check_cancelled(self):
        if self.state["status"] in {"stopped", "stopping"}:
            raise asyncio.CancelledError()
    def progress(self, fraction, stage=None, message=None, **kwargs):
        self.state.update(progress=fraction, message=message or "", detail=stage or "")
        self.manager.save(self.state)
    def log(self, message):
        with (Path(self.state["dir"]) / "workflow.log").open("a", encoding="utf-8") as f:
            f.write(str(message) + "\n")
    def item_error(self, item, error):
        self.log(f"{item}: {error}")


class BankWorkflows:
    def __init__(self, tk):
        self.tk = tk
        self.states, self.tasks = {}, {}
        self.index = tk.home.dir("training") / "bank-workflows.json"
    def save(self, state):
        state["updated"] = time.time()
        path = Path(state["dir"]) / "workflow.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    def load(self):
        if self.states or not self.index.is_file():
            return
        for folder in json.loads(self.index.read_text(encoding="utf-8")):
            path = Path(folder) / "workflow.json"
            if path.is_file():
                state = json.loads(path.read_text(encoding="utf-8"))
                if state["status"] in {"running", "stopping"}:
                    state.update(status="interrupted", message="Toolkit restarted; resume to continue from completed stages")
                self.states[state["id"]] = state
    def start(self, req, call):
        check_workflow(req)
        identifier = uuid.uuid4().hex[:12]
        folder = Path(req.project_dir).expanduser().resolve() / "workflows" / identifier
        folder.mkdir(parents=True, exist_ok=False)
        state = dict(id=identifier, dir=str(folder), request=req.model_dump(), status="running", stage="queued", progress=None,
            message="", created=time.time(), completed={}, runs={}, results={}, error="")
        self.states[identifier] = state
        self.save(state)
        tmp = self.index.with_suffix(".tmp")
        tmp.write_text(json.dumps([s["dir"] for s in self.states.values()]), encoding="utf-8")
        os.replace(tmp, self.index)
        self.tasks[identifier] = asyncio.create_task(self.execute(state, call))
        return state
    def resume(self, identifier, call):
        state = self.states[identifier]
        if state["status"] in {"running", "stopping", "done"}:
            raise ValueError("Only interrupted, failed or stopped workflows can resume")
        if state["status"] == "review":
            state["approved"] = True
            state["review_sources"] = state["results"]["dataset"]["spec"]["datasets"]
            state["completed"].pop("dataset", None)
        state.update(status="running", error="")
        self.tasks[identifier] = asyncio.create_task(self.execute(state, call))
        self.save(state)
        return state
    async def stop(self, identifier):
        state = self.states[identifier]
        if state["status"] != "running":
            return state
        state["status"] = "stopping"
        self.save(state)
        for run_id in state["runs"].values():
            run = self.tk.training.get(run_id)
            if run and run.info.status in ACTIVE:
                await self.tk.training.stop(run_id)
        task = self.tasks.get(identifier)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        state["status"] = "stopped"
        self.save(state)
        return state
    async def close(self):
        for task in self.tasks.values():
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    @asynccontextmanager
    async def slot(self, job):
        while self.tk.training._busy():
            job.check_cancelled()
            await asyncio.sleep(1)
        async with self.tk.training.inference_slot():
            yield

    async def execute(self, state, call):
        from .preparation import prepare_recordings
        from .workflow_pipeline import preprocess
        req = BankWorkflowRequest.model_validate(state["request"])
        folder = Path(state["dir"])
        job = WorkflowJob(self, state)
        variance_config = training_config(req, "variance", dict(datasets=[dict(raw_data_dir=s.folder, speaker=s.speaker, language=s.language) for s in req.sources],
            dictionaries=req.dictionaries or {s.language: "_generated_dictionary" for s in req.sources}))
        variance_overlay = build_overlay("diffsinger_variance", {**variance_config, "device": req.device}) if req.variance else {}
        needs_notes, needs_groups = bool(variance_overlay.get("predict_pitch")), bool(variance_overlay.get("predict_dur"))
        async def stage(name, action):
            job.check_cancelled()
            if name in state["completed"]:
                return state["completed"][name]
            state.update(stage=name, progress=None, message="")
            self.save(state)
            value = await action()
            state["completed"][name] = value
            self.save(state)
            return value
        try:
            prepared, profiles, extra = [], {k: v for k, v in req.phonemizer_profiles.items() if v}, set(req.extra_phonemes)
            for i, src in enumerate(req.sources):
                async def prep(src=src, i=i):
                    async with self.slot(job):
                        working = await preprocess(self.tk, job, src, folder / f"pipeline-{i}-{uuid.uuid4().hex[:6]}", req.pipeline, req.exclusions, call)
                        if not req.prepare:
                            return dict(out_dir=working, count=0, conversions=[])
                        languages = req.preparation.get("extra_languages", [])
                        targets = {code: req.phonemizer_profiles[code] if code in req.phonemizer_profiles else profiles.get(code) or default_profile(code) for code in [src.language, *languages]}
                        targets = {k: v for k, v in targets.items() if v}
                        available = getattr(self.tk.models, "installed", lambda: {})().values()
                        candidates = [m for m in available if m.engine == "wfl_asr" and set([src.language, *languages]) <= set(m.languages)
                            and (not languages or int(m.layout.get("generation", 1)) >= 2)]
                        candidates.sort(key=lambda m: ("multi13" not in m.id, m.id))
                        automatic_model = candidates[0].id if candidates else "wfl-multi13-v0.0.1"
                        params = dict(segment=True, max_length=20, label_mode="missing", label_model=automatic_model,
                            phonemizer_profiles=targets, estimate_notes=needs_notes, f0_method="auto", edge_silence=.25, fade_ms=5)
                        params.update(req.preparation)
                        if params["label_model"] in {"", "auto"}:
                            params["label_model"] = automatic_model
                        params.update(folder=working, out_dir=str(folder / f"prepared-{i}-{uuid.uuid4().hex[:6]}"), language=src.language,
                            recursive=src.recursive if Path(working).resolve() == Path(src.folder).resolve() else True)
                        return await prepare_recordings(self.tk, job, RecordingPreparationRequest(**params), call)
                result = await stage(f"prepare_{i}", prep)
                prepared.append(SourceSpec(**{**src.model_dump(), "folder": result["out_dir"],
                    "recursive": True if req.prepare or Path(result["out_dir"]).resolve() != Path(src.folder).resolve() else src.recursive}))
                profiles.update(result.get("phonemizer_profiles") or {})
                for conversion in result.get("conversions", []):
                    extra.update(conversion.get("extensions", {}))
            async def dataset():
                reviewed = [SourceSpec(**{**next((s.model_dump(exclude={"recursive"}) for s in req.sources if s.speaker == row["speaker"] and s.language == row["language"]), {}), "folder": row["raw_data_dir"]})
                    for row in state.get("review_sources", [])]
                result = await asyncio.to_thread(build_dataset, reviewed or prepared, folder / ("dataset-" + uuid.uuid4().hex[:6]),
                    req.dictionaries or None, 3, None, sorted(extra), profiles)
                if req.checks.get("detailed_audio"):
                    from .workflow_pipeline import audio_checks
                    async with self.slot(job):
                        await audio_checks(self.tk, job, result, req.checks, call)
                validate_report(str(Path(result.out_dir) / "dataset.json"), needs_notes, needs_groups)
                return dict(spec=spec_from_result(result), out_dir=result.out_dir, problems=result.problems,
                    coverage=result.coverage, speaker_coverage=result.speaker_coverage)
            built = await stage("dataset", dataset)
            state["results"]["dataset"] = built
            if req.review_before_training and not state.get("approved"):
                state.update(status="review", message="Data ready; review findings and continue when ready")
                return
            async def auxiliary(key, kind, config):
                if key in state["runs"]:
                    run = self.tk.training.get(state["runs"][key])
                    if run is None: raise ValueError("A workflow run was removed; start a new chain")
                    if run.info.status.value in {"stopped", "failed"}: await self.tk.training.resume(run.id)
                else:
                    run = self.tk.training.start(RunSpec(kind=kind, name=req.name + " · " + key,
                        dir=str(folder / (key + "-" + uuid.uuid4().hex[:6])), config=config, device=req.device))
                    state["runs"][key] = run.id; self.save(state)
                while run.info.status in ACTIVE:
                    job.check_cancelled(); job.progress(run.info.progress, message=run.info.message or f"{key}: {run.info.step}")
                    await asyncio.sleep(2)
                if run.info.status.value != "done": raise ValueError(run.info.error or f"{key} stopped")
                return dict(run=run.id, step=run.info.step, dir=str(run.dir))
            trained_vocoder = None
            if req.vocoder_training is not None:
                async def vocoder_train():
                    from .vocoder import build_overlay as vocoder_overlay, scan_sources, resolve_assets as vocoder_assets
                    config = {**req.vocoder_training, "sources": [s.__dict__ for s in prepared if s.role == "main"]}
                    if req.prototype:
                        config["steps"] = min(req.prototype_steps, int(config.get("steps", req.prototype_steps)))
                        config["validation_interval"] = min(config["steps"], int(config.get("validation_interval", 100)))
                    report = await asyncio.to_thread(scan_sources, config)
                    errors = [p["problem"] for p in report["problems"] if p["severity"] == "error"]
                    if errors: raise ValueError("Vocoder data: " + "; ".join(errors[:5]))
                    config.update(overlay=vocoder_overlay({**config, "device": req.device}), data_report=report, assets=vocoder_assets(config))
                    return await auxiliary("vocoder", "singingvocoders_vocoder", config)
                trained_vocoder = await stage("vocoder", vocoder_train)
                async def vocoder_export():
                    run = self.tk.training.get(trained_vocoder["run"])
                    out = folder / ("vocoder-export-" + uuid.uuid4().hex[:6])
                    async with self.slot(job):
                        native = await call(self.tk, job, "singingvocoders", "export", dict(run_dir=str(run.dir), kind=run.info.spec.kind, out_dir=str(out), step=trained_vocoder["step"]))
                        onnx = await call(self.tk, job, "diffsinger", "export_vocoder", dict(checkpoint=native["checkpoint"], out_dir=str(out)))
                    return dict(**onnx, checkpoint=native["checkpoint"])
                trained_vocoder = await stage("export_vocoder", vocoder_export)
                state["results"]["vocoder"] = trained_vocoder
            for n, settings in enumerate(req.g2p_training):
                async def g2p(n=n, settings=settings):
                    from .g2p import snapshot
                    language = settings.get("language", req.sources[0].language)
                    config = {**settings, "dictionary": settings.get("dictionary") or built["spec"]["dictionaries"][language], "language": language}
                    if req.prototype: config["max_steps"] = min(req.prototype_steps, int(config.get("max_steps") or req.prototype_steps))
                    config.update(await asyncio.to_thread(snapshot, config, folder / "g2p-data"))
                    return await auxiliary(f"g2p_{n}", "g2p_rnnt", config)
                trained = await stage(f"g2p_{n}", g2p)
                async def g2p_export(n=n, trained=trained):
                    run = self.tk.training.get(trained["run"])
                    async with self.slot(job):
                        return await call(self.tk, job, "g2p", "export", dict(run_dir=str(run.dir), kind=run.info.spec.kind, out_dir=str(folder / (f"g2p-export-{n}-" + uuid.uuid4().hex[:6])), step=trained["step"]))
                state["results"][f"g2p_{n}"] = await stage(f"export_g2p_{n}", g2p_export)
            async def train(kind, settings):
                if kind in state["runs"]:
                    run = self.tk.training.get(state["runs"][kind])
                    if run is None:
                        raise ValueError("A workflow run was removed; create a new workflow")
                    if run.info.status.value in {"failed", "stopped"}:
                        await self.tk.training.resume(run.id)
                else:
                    config = training_config(req, kind, {**built["spec"], "extra_phonemes": sorted(extra), "merged_phoneme_groups": req.merged_phoneme_groups})
                    if trained_vocoder:
                        config["assets"] = {**config.get("assets", {}), "vocoder": trained_vocoder["checkpoint"]}
                    config["overlay"] = build_overlay("diffsinger_" + kind, {**config, "device": req.device})
                    config["assets"] = resolve_assets(config)
                    run = self.tk.training.start(RunSpec(kind="diffsinger_" + kind, name=req.name + " · " + kind,
                        dir=str(folder / (kind + "-" + uuid.uuid4().hex[:6])), device=req.device, config=config))
                    state["runs"][kind] = run.id
                    self.save(state)
                while run.info.status in ACTIVE or (run.info.status.value == "failed" and run.info.error_kind == "oom" and run.task is not None and not run.task.done()):
                    job.check_cancelled()
                    job.progress(run.info.progress, message=run.info.message or f"{kind}: {run.info.step} steps")
                    await asyncio.sleep(2)
                if run.info.status.value != "done":
                    raise ValueError(run.info.error or f"{kind} stopped; resume the workflow to continue")
                return dict(run=run.id, dir=str(run.dir), step=run.info.step)
            trained = {}
            for kind, enabled, settings in [("acoustic", req.acoustic, req.acoustic_settings), ("variance", req.variance, req.variance_settings)]:
                if enabled:
                    trained[kind] = await stage(kind, lambda kind=kind, settings=settings: train(kind, settings))
            exports = {"acoustic": req.acoustic_export, "variance": req.variance_export}
            if req.export:
                for kind, record in trained.items():
                    async def export(kind=kind, record=record):
                        run = self.tk.training.get(record["run"])
                        out = folder / ("onnx-" + kind + "-" + uuid.uuid4().hex[:6])
                        async with self.slot(job):
                            value = await call(self.tk, job, run.info.spec.engine_name(), "export", dict(run_dir=str(run.dir),
                                kind=run.info.spec.kind, out_dir=str(out), step=record["step"] or None,
                                export_spk=sorted({s.speaker for s in req.sources if s.role == "main"})))
                        shutil.copy2(Path(built["out_dir"]) / "dataset.json", out / "dataset-report.json")
                        return value
                    value = await stage("export_" + kind, export)
                    exports[kind] = value["out_dir"]
                async def bank():
                    result = await asyncio.to_thread(assemble, VoicebankSpec(name=req.name + "-" + state["id"], out_dir=str(Path(req.project_dir) / "exports"),
                        acoustic=exports["acoustic"], variance=[exports["variance"]] if exports["variance"] else [], vocoder=trained_vocoder["out_dir"] if trained_vocoder else req.vocoder,
                        phonemizer_profiles=profiles, default_phonemizer_profile=profiles.get(req.sources[0].language, ""),
                        notes="Prototype: short training for pipeline verification." if req.prototype else ""))
                    return dict(dir=result.dir, files=result.files, problems=result.problems)
                state["results"]["bank"] = await stage("bank", bank)
                phrase_path = req.render_ds
                if req.auto_test_phrase and not phrase_path:
                    async def phrase():
                        from .test_phrases import PhraseRequest, inventory, build_phrase
                        bank_dir = Path(state["results"]["bank"]["dir"])
                        info = inventory(bank_dir, req.sources[0].language)
                        params = {"mode": "cv" if info["consonants"] else "v", **req.test_phrase,
                            "model_dir": str(bank_dir), "run_id": "", "config": "", "language": req.sources[0].language}
                        return await asyncio.to_thread(build_phrase, PhraseRequest.model_validate(params), folder / ("phrase-" + uuid.uuid4().hex[:6]))
                    state["results"]["phrase"] = await stage("test_phrase", phrase)
                    phrase_path = state["results"]["phrase"]["path"]
                if phrase_path and (req.render_ds or req.render_test_phrase):
                    async def render():
                        async with self.slot(job):
                            return await call(self.tk, job, "bank_render", "render", dict(**{k: v for k, v in req.render_options.items() if k not in {"bank", "ds", "out_dir", "vocoder"}}, bank=state["results"]["bank"]["dir"],
                                ds=phrase_path, out_dir=str(folder / ("bank-render-" + uuid.uuid4().hex[:6])), vocoder=trained_vocoder["out_dir"] if trained_vocoder else req.vocoder or None))
                    state["results"]["render"] = await stage("bank_render", render)
            state.update(status="done", progress=1.0, message="Bank workflow completed")
        except asyncio.CancelledError:
            if state["status"] == "running":
                state.update(status="interrupted", message="Workflow interrupted; completed stages are retained")
        except Exception as error:
            state.update(status="failed", error=str(error), message=str(error))
            job.log(str(error))
        finally:
            self.save(state)

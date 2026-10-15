"""HTTP API (FastAPI). Interactive documentation: http://<host>:<port>/docs"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import sys
import time
import uuid
import json
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .. import __version__, formats
from ..api_models import (
    AlignRequest,
    CatalogAddRequest,
    ConvertRequest,
    DatasetBuildRequest,
    RecordingPreparationRequest,
    DatasetDownloadRequest,
    FixLabelsRequest,
    MidiRequest,
    ModelImportRequest,
    PitchRequest,
    RefineRequest,
    SegmentRequest,
    SeparateRequest,
    ResynthRequest,
    OtoNoteRequest,
    TempoRequest,
    TextRequest,
    TrainingExportRequest,
    TrainingQueueRequest,
    CheckpointRetentionRequest,
    CheckpointCopyRequest,
    VoicebankRequest,
    TrainingRenderRequest,
    TranscribeRequest,
)
from ..datasets import license_text
from ..engines.manager import EngineError
from ..jobs import FINISHED, Job, JobInfo
from ..models.catalog import ENGINE_TASKS, TASKS
from ..models.store import ModelNotFound, model_listing
from ..pipelines.refine import run_refine
from ..pipelines.extra import run_midi, run_oto_note, run_pitch, run_resynth, run_segment, run_separate, run_tempo, run_text
from ..pipelines.label import run_align, run_transcribe
from ..text.g2p import load_user_words, save_user_words
from ..text.languages import language_info
from ..text.rules import RULE_SETS
from ..training.runs import FINISHED as RUN_FINISHED, RunSpec
from ..clients import STALE_SECONDS, Clients
from .. import storage
from ..toolkit import Toolkit
from ..update import check_update

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def create_app(toolkit: Toolkit | None = None, exit_when_unused: float = 0.0) -> FastAPI:
    """exit_when_unused: seconds; when > 0 the server stops by itself once no program is attached and no job
    has run for that long (a toolkit started by a program for its own use)."""
    tk_holder: dict[str, Toolkit] = {}
    clients = Clients()

    async def watch_unused(tk: Toolkit) -> None:
        import asyncio  # noqa: PLC0415

        while True:
            await asyncio.sleep(5)
            busy = any(j.info.status.value in ("queued", "running", "paused") for j in tk.jobs.jobs.values())
            workflow_busy = any(w["status"] in {"running", "stopping"} for w in tk.workflows.states.values())
            training_busy = any(r.info.status.value in {"queued", "preparing", "training", "stopping"} for r in tk.training.runs.values())
            if clients.active() or busy or workflow_busy or training_busy:
                clients.last_used = time.monotonic()
                continue
            # a program gets two minutes to attach after starting the toolkit
            grace = max(exit_when_unused, 120.0) if clients.last_used <= clients.started + 1 else exit_when_unused
            if time.monotonic() - clients.last_used > grace:
                print("No program uses the toolkit any more; stopping.", flush=True)
                try:
                    await tk.engines.stop_all()
                except Exception:  # noqa: BLE001
                    pass
                os._exit(0)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        import asyncio  # noqa: PLC0415

        tk = toolkit or Toolkit()
        tk_holder["tk"] = tk
        await tk.start()
        if tk.settings.keep_files_days > 0:
            # leftovers of old jobs, away from the start: a big outputs folder takes a while to walk
            asyncio.get_running_loop().run_in_executor(None, storage.cleanup, tk.home, float(tk.settings.keep_files_days))
        watcher = asyncio.create_task(watch_unused(tk)) if exit_when_unused > 0 else None
        yield
        if watcher is not None:
            watcher.cancel()
        await tk.stop()

    app = FastAPI(
        title="mVocalToolkit",
        version=__version__,
        description="Automatic labeling and singing voice dataset tools as a local API.",
        lifespan=lifespan,
    )

    def get_tk() -> Toolkit:
        return tk_holder["tk"]

    def _authorized(host: str | None, token: str | None, tk: Toolkit) -> bool:
        is_local = host in LOCAL_HOSTS
        if is_local and not tk.settings.require_token_local:
            return True
        return bool(tk.settings.token) and token == tk.settings.token

    async def auth(request: Request, tk: Toolkit = Depends(get_tk)) -> None:
        header = request.headers.get("authorization", "")
        token = header.removeprefix("Bearer ").strip() or request.headers.get("x-mvt-token") or request.query_params.get("token")
        if not _authorized(request.client.host if request.client else None, token, tk):
            raise HTTPException(401, "Invalid or missing token")

    def submit(tk: Toolkit, kind: str, func, params: dict[str, Any]) -> JobInfo:
        job = tk.jobs.submit(kind, func, params)
        return job.info

    # ---------------- system ----------------

    @app.get("/health", tags=["system"])
    async def health(tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Quick check used by GUIs ("Check" button). Doesn't need a token."""
        return {
            "ok": True,
            "name": "mVocalToolkit",
            "version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "home": str(tk.home.root),
            "gpu": _gpu_info(),
            "clients": len(clients.active()),
            "exit_when_unused": exit_when_unused,
            "background_active": any(s["status"] in {"running", "stopping"} for s in tk.workflows.states.values())
                or any(r.info.status in {"queued", "preparing", "training", "stopping"} for r in tk.training.runs.values())
                or any(j.info.status.value in {"queued", "running", "paused"} for j in tk.jobs.jobs.values()),
        }

    # ---------------- programs using the toolkit ----------------

    @app.post("/clients", tags=["system"], dependencies=[Depends(auth)])
    async def attach(body: dict[str, Any]) -> dict[str, Any]:
        """A program starts using the toolkit; it should ping at least every minute and detach when it closes."""
        pid = body.get("pid")
        c = clients.attach(str(body.get("name") or ""), int(pid) if isinstance(pid, (int, float)) else None)
        return {"id": c.id, "stale_seconds": STALE_SECONDS}

    @app.get("/clients", tags=["system"], dependencies=[Depends(auth)])
    async def list_clients() -> list[dict[str, Any]]:
        return clients.describe()

    @app.post("/clients/{client_id}/ping", tags=["system"], dependencies=[Depends(auth)])
    async def ping(client_id: str) -> dict[str, Any]:
        if not clients.ping(client_id):
            raise HTTPException(404, "Unknown client; attach again")
        return {"ok": True}

    @app.delete("/clients/{client_id}", tags=["system"], dependencies=[Depends(auth)])
    async def detach(client_id: str) -> dict[str, Any]:
        clients.detach(client_id)
        return {"ok": True, "clients": len(clients.active())}

    @app.get("/capabilities", tags=["system"], dependencies=[Depends(auth)])
    async def capabilities(tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        return {
            "version": __version__,
            "engines": tk.engines.list(),
            "formats": sorted(set(formats.FORMATS) | {"ds_csv"}),
            "rule_sets": {name: rs["rules"] for name, rs in RULE_SETS.items()},
            "tasks": TASKS,
            "languages": sorted({lang for e in tk.catalog.entries.values() for lang in e.languages}),
        }

    @app.get("/settings", tags=["system"], dependencies=[Depends(auth)])
    async def get_settings(tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        data = tk.settings.to_dict()
        for secret in ("token", "github_token", "hf_token"):
            if data.get(secret):
                data[secret] = "***"
        return data

    # POST too: some HTTP clients (Java's HttpURLConnection) can't send PATCH
    @app.post("/settings", tags=["system"], dependencies=[Depends(auth)])
    @app.patch("/settings", tags=["system"], dependencies=[Depends(auth)])
    async def patch_settings(values: dict[str, Any], tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        device = tk.settings.device
        for key, value in values.items():
            if key in ("host", "port", "token"):
                continue  # need a restart / CLI
            if hasattr(tk.settings, key):
                setattr(tk.settings, key, value)
            else:
                tk.settings.extra[key] = value
        tk.home.save_settings(tk.settings)
        if tk.settings.device != device:
            # engines read the device when they start; busy ones change after their work
            await tk.engines.stop_idle()
        return await get_settings(tk)

    @app.post("/shutdown", tags=["system"], dependencies=[Depends(auth)])
    async def shutdown(tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Stops the server (for programs that started it and are closing), engines first."""
        import threading  # noqa: PLC0415

        try:
            await tk.engines.stop_all()
        except Exception:  # noqa: BLE001
            pass
        threading.Timer(0.3, lambda: os._exit(0)).start()
        return {"ok": True}

    @app.post("/update", tags=["system"], dependencies=[Depends(auth)])
    async def update_now(tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Updates the toolkit now when its source has something newer (instead of waiting for the check at start):
        hands over to the updater, which reinstalls it and starts it again; this server then exits with
        EXIT_UPDATING. {"updating": false} when it is up to date (or wasn't installed as a uv tool)."""
        import threading  # noqa: PLC0415

        from ..self_update import EXIT_UPDATING, hand_over, pending_update, update_log  # noqa: PLC0415

        source = await asyncio.to_thread(pending_update, tk.home.root, tk.settings.github_token, True)
        if not source:
            return {"updating": False}
        if not hand_over(source, tk.home.root, sys.argv[1:]):
            return {"updating": False, "error": "uv was not found"}
        log = update_log(tk.home.root)
        # the same words as an update found at start: a program reading the output follows the log
        print(f"A newer mVocalToolkit is available; updating from {source} and starting again.", flush=True)
        print(f"Update log: {log}", flush=True)
        try:
            await tk.engines.stop_all()
        except Exception:  # noqa: BLE001
            pass
        threading.Timer(0.5, lambda: os._exit(EXIT_UPDATING)).start()
        return {"updating": True, "log": str(log)}

    @app.get("/update/check", tags=["system"], dependencies=[Depends(auth)])
    async def update_check(tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        # the model catalog is checked too: new models show up without restarting
        await asyncio.to_thread(tk.catalog.load, False, True)
        return await check_update()

    # ---------------- engines ----------------

    # ---------------- disk space ----------------

    def _busy(tk: Toolkit) -> bool:
        return any(j.info.status not in FINISHED for j in tk.jobs.jobs.values())

    @app.get("/storage", tags=["system"], dependencies=[Depends(auth)])
    async def storage_usage(tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Bytes used by each model, each engine (environment and sources) and the temporary folders of the home."""
        data = await asyncio.to_thread(storage.usage, tk.home)
        data["keep_files_days"] = tk.settings.keep_files_days
        return data

    @app.post("/storage/cleanup", tags=["system"], dependencies=[Depends(auth)])
    async def storage_cleanup(body: dict[str, Any] | None = None, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Removes uploads, results in <home>/outputs, job history and unfinished downloads.

        Body (all optional): {"older_than_days": 7} (default: all of them), {"parts": ["uploads", "outputs", "jobs",
        "cache/work"]}. While a job runs only entries older than a day are removed."""
        body = body or {}
        days = body.get("older_than_days")
        parts = tuple(body.get("parts") or storage.TEMPORARY)
        busy = _busy(tk)
        result = await asyncio.to_thread(storage.cleanup, tk.home, float(days) if days is not None else None, parts, busy)
        if "jobs" in parts:
            # finished jobs are forgotten too, as if the server had been restarted
            for job_id in [j.id for j in tk.jobs.jobs.values() if j.info.status in FINISHED and not (tk.home.jobs / f"{j.id}.json").exists()]:
                tk.jobs.jobs.pop(job_id, None)
        return {**result, "busy": busy}

    @app.get("/engines", tags=["engines"], dependencies=[Depends(auth)])
    async def engines(tk: Toolkit = Depends(get_tk)) -> list[dict[str, Any]]:
        return tk.engines.list()

    @app.post("/engines/{name}/install", tags=["engines"], dependencies=[Depends(auth)])
    async def install_engine(name: str, force: bool = False, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        spec = _engine(tk, name)

        async def work(job: Job):
            await tk.engines.stop(name)
            job.progress(0.05, stage="install", message=f"Installing {name}")
            await tk.engines.envs.install(spec, log=job.log, force=force)
            return tk.engines.envs.status(spec)

        return submit(tk, "engine_install", work, {"engine": name})

    @app.delete("/engines/{name}", tags=["engines"], dependencies=[Depends(auth)])
    async def uninstall_engine(name: str, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        spec = _engine(tk, name)
        await tk.engines.stop(name)
        tk.engines.envs.uninstall(spec)
        return {"ok": True}

    @app.post("/engines/{name}/stop", tags=["engines"], dependencies=[Depends(auth)])
    async def stop_engine(name: str, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        _engine(tk, name)
        await tk.engines.stop(name)
        return {"ok": True}

    @app.get("/engines/{name}/info", tags=["engines"], dependencies=[Depends(auth)])
    async def engine_info(name: str, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Starts the engine (if installed) and returns its runtime info: torch, CUDA, GPU."""
        spec = _engine(tk, name)
        if not tk.engines.envs.is_ready(spec):
            raise HTTPException(409, f"Engine {name} is not installed")
        info = await tk.engines.call(name, "ping")
        worker = tk.engines.workers.get(name)
        return {**info, "log": list(worker.stderr_tail)[-50:] if worker else []}

    # ---------------- models ----------------

    @app.get("/models", tags=["models"], dependencies=[Depends(auth)])
    async def models(task: str | None = None, language: str | None = None, engine: str | None = None,
                     tk: Toolkit = Depends(get_tk)) -> list[dict[str, Any]]:
        """Catalog + installed models. task: align, transcribe, segment, midi, tempo, pitch, separate.
        language: models for this language (and language-independent ones). engine: comma separated list."""
        return model_listing(tk.models, task=task, engine=engine, language=language)

    @app.get("/tasks", tags=["models"], dependencies=[Depends(auth)])
    async def tasks(tk: Toolkit = Depends(get_tk)) -> list[dict[str, Any]]:
        """Tasks a GUI can offer, with their engines and whether models depend on the language."""
        result = []
        for task in TASKS:
            engines = sorted({name for name, t in ENGINE_TASKS.items() if task in t})
            listed = model_listing(tk.models, task=task)
            result.append({
                "task": task,
                "engines": engines,
                "models": len(listed),
                "language_specific": any(m.get("languages") and m["languages"] != ["*"] for m in listed),
            })
        return result

    @app.get("/languages", tags=["models"], dependencies=[Depends(auth)])
    async def languages(task: str = "align", tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Languages that have models for a task, each with its models: the "language -> model" choice of a GUI.

        Models without a language (or with "*") work for any language: they are listed in every language and
        in the "*" group. Selecting a model and starting the operation downloads it if needed."""
        groups: dict[str, list[dict[str, Any]]] = {}
        universal: list[dict[str, Any]] = []
        for model in model_listing(tk.models, task=task):
            short = {k: model.get(k) for k in ("id", "name", "engine", "type", "version", "description", "author",
                                                "size_hint", "installed", "languages", "pack", "tags",
                                                "defaults", "text_frontend")}
            langs = [lang for lang in model.get("languages") or [] if lang != "*"]
            if not langs:
                universal.append(short)
            for lang in langs:
                groups.setdefault(lang, []).append(short)
        result = [{**language_info(code), "models": models + universal} for code, models in sorted(groups.items())]
        if universal:
            result.append({**language_info("*"), "models": universal})
        return {"task": task, "languages": result}

    @app.get("/models/installed", tags=["models"], dependencies=[Depends(auth)])
    async def installed_models(tk: Toolkit = Depends(get_tk)):
        return list(tk.models.installed().values())

    @app.post("/models/{model_id}/download", tags=["models"], dependencies=[Depends(auth)])
    async def download_model(model_id: str, force: bool = False, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        entry = tk.catalog.get(model_id)
        if entry is None:
            raise HTTPException(404, f"Model {model_id} is not in the catalogs")
        if not force and tk.models.get_installed(model_id) is not None:
            raise HTTPException(409, "Already installed (use force=true to reinstall)")

        async def work(job: Job):
            installed = await tk.models.download(entry, lambda v, m, d=None: job.progress(v, stage="download", message=m, detail=d))
            return {"installed": [m.id for m in installed]}

        return submit(tk, "model_download", work, {"model": model_id})

    @app.delete("/models/{model_id}", tags=["models"], dependencies=[Depends(auth)])
    async def delete_model(model_id: str, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        if not tk.models.remove(model_id):
            raise HTTPException(404, "Not installed")
        return {"ok": True}

    @app.get("/models/{model_id}/words", tags=["models"], dependencies=[Depends(auth)])
    async def model_words(model_id: str, tk: Toolkit = Depends(get_tk)) -> dict[str, list[str]]:
        """The user's own words of an aligner model (used by every request with it)."""
        return load_user_words(tk.home, model_id)

    @app.put("/models/{model_id}/words", tags=["models"], dependencies=[Depends(auth)])
    async def put_model_words(model_id: str, words: dict[str, list[str]], tk: Toolkit = Depends(get_tk)) -> dict[str, list[str]]:
        """Replaces the user's own words of a model: {"word": ["ph", ...]}; {} removes them all."""
        return save_user_words(tk.home, model_id, words)

    @app.post("/models/import", tags=["models"], dependencies=[Depends(auth)])
    async def import_model(req: ModelImportRequest, tk: Toolkit = Depends(get_tk)):
        _engine(tk, req.engine)
        try:
            installed = await asyncio.to_thread(
                tk.models.import_local, req.engine, req.path, req.id, req.name, req.languages, req.text_frontend,
                req.copy_files, req.author,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            raise HTTPException(400, str(e)) from e
        return installed

    @app.get("/catalogs", tags=["models"], dependencies=[Depends(auth)])
    async def catalogs(tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        return {"sources": tk.catalog.sources, "count": len(tk.catalog.entries)}

    @app.post("/catalogs", tags=["models"], dependencies=[Depends(auth)])
    async def add_catalog(req: CatalogAddRequest, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        await asyncio.to_thread(tk.catalog.add_user_catalog, req.ref)
        return await catalogs(tk)

    @app.post("/catalogs/refresh", tags=["models"], dependencies=[Depends(auth)])
    async def refresh_catalogs(tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        await asyncio.to_thread(tk.catalog.load, True)
        return await catalogs(tk)

    # ---------------- datasets ----------------

    @app.get("/datasets", tags=["datasets"], dependencies=[Depends(auth)])
    async def datasets(language: str | None = None, tk: Toolkit = Depends(get_tk)) -> list[dict[str, Any]]:
        """Public singing datasets for training, each with its license (name, link, summary and what it allows:
        training, publishing models, commercial use, redistribution). Only datasets whose license allows training
        are listed. A GUI shows the license before the download."""
        return tk.datasets.listing(language)

    @app.get("/datasets/{dataset_id}/license", tags=["datasets"], dependencies=[Depends(auth)])
    async def dataset_license(dataset_id: str, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        entry = tk.catalog.datasets.get(dataset_id)
        if entry is None:
            raise HTTPException(404, f"Dataset {dataset_id} is not in the catalogs")
        return {**entry.license.model_dump(), "text": license_text(entry)}

    @app.post("/datasets/{dataset_id}/download", tags=["datasets"], dependencies=[Depends(auth)])
    async def download_dataset(dataset_id: str, req: DatasetDownloadRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Downloads a dataset into <home>/datasets/<id> (with LICENSE.txt). accept_license must be true: the user
        has seen and accepted the license."""
        entry = tk.catalog.datasets.get(dataset_id)
        if entry is None:
            raise HTTPException(404, f"Dataset {dataset_id} is not in the catalogs")
        if not req.accept_license:
            raise HTTPException(403, {"error": "license_not_accepted", "license": entry.license.model_dump(),
                                      "text": license_text(entry)})
        if entry.manual_download or not entry.sources:
            raise HTTPException(409, {"error": "manual_download", "url": entry.manual_download or entry.homepage})
        if not req.force and tk.datasets.is_installed(dataset_id):
            raise HTTPException(409, "Already downloaded (use force to download again)")

        async def work(job: Job):
            path = await tk.datasets.download(
                entry, True, lambda v, m, d=None: job.progress(v, stage="download", message=m, detail=d))
            return {"path": str(path)}

        return submit(tk, "dataset_download", work, {"dataset": dataset_id})

    @app.delete("/datasets/{dataset_id}", tags=["datasets"], dependencies=[Depends(auth)])
    async def delete_dataset(dataset_id: str, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        if not tk.datasets.remove(dataset_id):
            raise HTTPException(404, "Not downloaded")
        return {"ok": True}

    # ---------------- training ----------------

    @app.get("/training/environment", tags=["training"], dependencies=[Depends(auth)])
    async def training_environment(engine: str = "diffsinger", tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """torch, CUDA, GPUs with free memory and free disk as the training engine sees them. {"installed": false}
        when the engine isn't installed yet (the first run installs it)."""
        spec = _engine(tk, engine)
        if not tk.engines.envs.is_ready(spec):
            return {"installed": False, "engine": engine, "size_hint": spec.size_hint}
        return {"installed": True, "engine": engine, **await _guard(tk.engines.call(engine, "environment"))}

    @app.post("/training/overlay", tags=["training"], dependencies=[Depends(auth)])
    async def training_overlay(spec: RunSpec) -> dict[str, Any]:
        """The config a DiffSinger run spec resolves to (presets applied), without starting anything: for a
        review step. The engine merges it over openvpi's defaults when the run starts."""
        from ..training.diffsinger import build_overlay, resolve_assets  # noqa: PLC0415

        if spec.kind == "g2p_rnnt":
            from ..training.g2p import report
            try:
                return {"overlay": report(spec.config), "assets": {}}
            except (OSError, ValueError, TypeError) as exc:
                raise HTTPException(400, str(exc)) from exc

        if spec.kind == "singingvocoders_vocoder":
            from ..training.vocoder import build_overlay as vocoder_overlay, resolve_assets as vocoder_assets

            try:
                overlay = vocoder_overlay({**spec.config, "device": spec.device})
                return {"overlay": overlay, "assets": vocoder_assets(spec.config), "real_steps": overlay["max_updates"] // 2}
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, str(exc)) from exc
        if not spec.kind.startswith("diffsinger"):
            raise HTTPException(400, "Only DiffSinger runs have an overlay")
        try:
            return {"overlay": build_overlay(spec.kind, {**spec.config, "device": spec.device}), "assets": resolve_assets(spec.config)}
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/training/runs", tags=["training"], dependencies=[Depends(auth)])
    async def training_start(spec: RunSpec, tk: Toolkit = Depends(get_tk)):
        """Queues a training run. One run trains at a time; the trainer is a process of its own that keeps running
        when the server stops. Progress: GET /training/runs/{id} (numbers), /events (everything), WS /events."""
        from ..training.recipes import FAMILIES, snapshot as recipe_snapshot
        if spec.kind in FAMILIES:
            engine = FAMILIES[spec.kind]["engine"]
            if spec.device == "mps" or spec.engine and spec.engine != engine:
                raise HTTPException(400, "Choose the matching training engine and CPU or CUDA")
            config = {**spec.config, "allow_download": tk.settings.auto_download_models}
            try:
                prepared = await asyncio.to_thread(recipe_snapshot, spec.kind, config, tk.home.dir("training") / "recipe-data")
            except (OSError, ValueError, TypeError) as exc:
                raise HTTPException(400, str(exc)) from exc
            spec = spec.model_copy(update={"engine":engine,"config":{**config,**prepared}})
        _engine(tk, spec.engine_name())
        if spec.kind == "rvc_voice":
            if spec.engine_name() != "rvc" or spec.device == "mps":
                raise HTTPException(400, "RVC training requires the rvc engine and CPU or CUDA")
            from ..training.rvc import snapshot
            try:
                data = await asyncio.to_thread(snapshot, spec.config, tk.home.dir("training") / "rvc-data")
            except (OSError, ValueError, TypeError) as exc:
                raise HTTPException(400, str(exc)) from exc
            spec = spec.model_copy(update={"config": {**spec.config, **data}})
        if spec.kind == "g2p_rnnt":
            from ..training.g2p import snapshot
            try:
                data = await asyncio.to_thread(snapshot, spec.config, tk.home.dir("training") / "g2p-data")
            except (OSError, ValueError, TypeError) as exc:
                raise HTTPException(400, str(exc)) from exc
            spec = spec.model_copy(update={"config": {**{k: v for k, v in spec.config.items() if k != "prepared"}, **data}})
        if spec.kind == "singingvocoders_vocoder":
            from ..training.vocoder import build_overlay as vocoder_overlay, resolve_assets as vocoder_assets, scan_sources

            try:
                overlay = vocoder_overlay({**spec.config, "device": spec.device})
                report = await asyncio.to_thread(scan_sources, spec.config)
                errors = [p["problem"] for p in report["problems"] if p["severity"] == "error"]
                if errors:
                    raise ValueError("; ".join(errors[:5]))
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, str(exc)) from exc
            spec = spec.model_copy(update={"config": {**spec.config, "overlay": overlay, "data_report": report, "assets": vocoder_assets(spec.config)}})
        if spec.kind.startswith("diffsinger"):
            from ..training.diffsinger import build_overlay, resolve_assets  # noqa: PLC0415
            from ..training.dataset import validate_report  # noqa: PLC0415

            try:
                overlay = build_overlay(spec.kind, {**spec.config, "device": spec.device})
                validate_report(str(spec.config.get("data_report") or ""), bool(overlay.get("predict_pitch")), bool(overlay.get("predict_dur")))
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            spec = spec.model_copy(update={"config": {**spec.config, "overlay": overlay, "assets": resolve_assets(spec.config)}})
        try:
            run = tk.training.start(spec)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return run.info

    @app.get("/training/runs", tags=["training"], dependencies=[Depends(auth)])
    async def training_runs(active: bool = False, tk: Toolkit = Depends(get_tk)):
        infos = tk.training.list()
        if active:
            infos = [i for i in infos if i.status not in RUN_FINISHED]
        return infos

    @app.post("/training/vocoder/check", tags=["training"], dependencies=[Depends(auth)])
    async def vocoder_check(config: dict[str, Any]):
        from ..training.vocoder import scan_sources

        try:
            return await asyncio.to_thread(scan_sources, config)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/training/g2p/check", tags=["training"], dependencies=[Depends(auth)])
    async def g2p_check(config: dict[str, Any]):
        from ..training.g2p import report
        try:
            return await asyncio.to_thread(report, config)
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/training/rvc/check", tags=["training"], dependencies=[Depends(auth)])
    async def rvc_check(config: dict[str, Any]):
        from ..training.rvc import check
        try:
            data = await asyncio.to_thread(check, config)
            return {k: v for k, v in data.items() if k != "recordings"}
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/training/recipes", tags=["training"], dependencies=[Depends(auth)])
    async def training_recipes():
        from ..training.recipes import FAMILIES
        return FAMILIES

    @app.post("/training/recipes/{kind}/check", tags=["training"], dependencies=[Depends(auth)])
    async def recipe_check(kind: str, config: dict[str, Any], tk: Toolkit = Depends(get_tk)):
        from ..training.recipes import check
        try:
            return await asyncio.to_thread(check, kind, {**config,"allow_download":tk.settings.auto_download_models})
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/training/config/read", tags=["training"], dependencies=[Depends(auth)])
    async def training_config_read(request: dict[str, str]):
        import yaml
        content = request.get("content", "")
        if len(content) > 1_048_576:
            raise HTTPException(400, "Configuration must be smaller than 1 MB")
        try:
            result = yaml.safe_load(content)
            if not isinstance(result, dict) or not all(isinstance(k, str) for k in result):
                raise ValueError("Configuration must be an object with string keys")
            # Return JSON-compatible values only; YAML timestamps/custom values are not settings.
            json.dumps(result, allow_nan=False)
            return result
        except (yaml.YAMLError, ValueError, TypeError, RecursionError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/training/runs/{run_id}/predict", tags=["training"], dependencies=[Depends(auth)])
    async def g2p_predict(run_id: str, request: dict[str, Any], tk: Toolkit = Depends(get_tk)):
        run = _run(tk, run_id)
        words = request.get("words", [])
        if run.info.spec.kind != "g2p_rnnt" or not isinstance(words, list) or not 1 <= len(words) <= 100 or any(not isinstance(w, str) or not 1 <= len(w) <= 128 for w in words):
            raise HTTPException(400, "Provide 1–100 words of 1–128 characters for a G2P run")
        step = request.get("step")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise HTTPException(400, "Checkpoint step must be a non-negative integer")
        if tk.training._busy():
            raise HTTPException(409, "Wait for active training or rendering to finish before predicting pronunciations")
        async with tk.training.inference_slot():
            return await _guard(tk.engines.call("g2p", "predict", {"run_dir": str(run.dir), "words": words, "step": step}))

    @app.post("/training/dictionary/read", tags=["training"], dependencies=[Depends(auth)])
    async def dictionary_read(request: dict[str, Any]):
        from ..training.dictionaries import describe_entries, read_entries

        try:
            path = Path(str(request.get("path") or "")).expanduser().resolve()
            if not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
                raise ValueError("Choose a dictionary smaller than 20 MB on the toolkit computer")
            entries = await asyncio.to_thread(read_entries, path, request.get("language"))
            return describe_entries(entries)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/training/dictionary/save", tags=["training"], dependencies=[Depends(auth)])
    async def dictionary_save(request: dict[str, Any]):
        from ..training.dictionaries import save_copy

        try:
            if not request.get("out_dir"):
                raise ValueError("Choose a project folder for the dictionary copy")
            return await asyncio.to_thread(save_copy, request.get("entries"), Path(request["out_dir"]).expanduser().resolve())
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/training/queue", tags=["training"], dependencies=[Depends(auth)])
    async def training_queue(req: TrainingQueueRequest, tk: Toolkit = Depends(get_tk)):
        try:
            tk.training.reorder_queue(req.ids)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True}

    def _run(tk: Toolkit, run_id: str):
        run = tk.training.get(run_id)
        if run is None:
            raise HTTPException(404, f"Unknown run {run_id}")
        return run

    async def training_call(tk: Toolkit, job: Job, engine: str, method: str, params: dict):
        task = asyncio.create_task(tk.engines.call(engine, method, params,
            on_progress=lambda d: job.progress(d.get("progress"), stage=d.get("stage") or method, message=d.get("message")), log=job.log))
        try:
            while not task.done():
                await asyncio.wait([task], timeout=0.2)
                if job.cancelled:
                    await tk.engines.abort(engine)
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    job.check_cancelled()
            job.check_cancelled()
            return await task
        except asyncio.CancelledError:
            await tk.engines.abort(engine)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    @app.get("/training/runs/{run_id}", tags=["training"], dependencies=[Depends(auth)])
    async def training_run(run_id: str, tk: Toolkit = Depends(get_tk)):
        return _run(tk, run_id).info

    @app.get("/training/runs/{run_id}/events", tags=["training"], dependencies=[Depends(auth)])
    async def training_events(run_id: str, since: int = 0, wait: float = 0, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Events number `since` onwards (0-based). wait: seconds to wait for a new one when there is none yet."""
        run = _run(tk, run_id)
        events = run.read_events_since(since)
        if not events and wait > 0 and run.info.status not in RUN_FINISHED:
            queue = run.subscribe()
            try:
                await asyncio.wait_for(queue.get(), timeout=min(wait, 60))
            except asyncio.TimeoutError:
                pass
            finally:
                run.unsubscribe(queue)
            events = run.read_events_since(since)
        return {"since": since, "next": since + len(events), "events": events, "status": run.info.status}

    @app.get("/training/runs/{run_id}/log", tags=["training"], dependencies=[Depends(auth)])
    async def training_log(run_id: str, lines: int = 200, name: str = "train.log", tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        run = _run(tk, run_id)
        if name not in ("train.log", "prepare.log", "export.log", "render.log", "tensorboard.log"):
            raise HTTPException(400, "name: train.log, prepare.log, export.log, render.log or tensorboard.log")
        return {"lines": run.log_tail(lines, name), "path": str(run.dir / name)}

    @app.post("/training/runs/{run_id}/stop", tags=["training"], dependencies=[Depends(auth)])
    async def training_stop(run_id: str, tk: Toolkit = Depends(get_tk)):
        """Asks the trainer to save a checkpoint and end (killed after a few minutes if it doesn't)."""
        _run(tk, run_id)
        return await tk.training.stop(run_id)

    @app.post("/training/runs/{run_id}/tensorboard", tags=["training"], dependencies=[Depends(auth)])
    async def training_tensorboard(run_id: str, tk: Toolkit = Depends(get_tk)):
        from ..engines.manager import process_environment
        run = _run(tk, run_id)
        if run.info.spec.engine_name() not in {"diffsinger", "diffsinger_benchmark", "singingvocoders", "rvc", "tifa_train", "sofa_train", "game_train", "label_train", "refiner_train", "qwen_tts", "voxcpm"}:
            raise HTTPException(400, "This trainer has no TensorBoard logs; use its built-in metrics")
        spec = tk.engines.spec(run.info.spec.engine_name())
        python, cwd, env = process_environment(spec, tk.engines.envs, tk.settings)
        try:
            return await tk.tensorboard.start(run_id, run.dir, python, cwd, env)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/training/runs/{run_id}/tensorboard", tags=["training"], dependencies=[Depends(auth)])
    async def training_tensorboard_stop(run_id: str, tk: Toolkit = Depends(get_tk)):
        _run(tk, run_id)
        if tk.tensorboard.run_id == run_id:
            await tk.tensorboard.stop()
        return {"ok": True}

    @app.post("/training/runs/{run_id}/resume", tags=["training"], dependencies=[Depends(auth)])
    async def training_resume(run_id: str, tk: Toolkit = Depends(get_tk)):
        """Queues a stopped or failed run again; it continues from its last checkpoint."""
        _run(tk, run_id)
        try:
            return await tk.training.resume(run_id)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    @app.delete("/training/runs/{run_id}", tags=["training"], dependencies=[Depends(auth)])
    async def training_delete(run_id: str, files: bool = False, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Forgets the run (stopping it first); files=true also removes its folder."""
        _run(tk, run_id)
        if tk.tensorboard.run_id == run_id:
            await tk.tensorboard.stop()
        await tk.training.delete(run_id, files=files)
        return {"ok": True}

    @app.get("/training/runs/{run_id}/checkpoints", tags=["training"], dependencies=[Depends(auth)])
    async def training_checkpoints(run_id: str, tk: Toolkit = Depends(get_tk)):
        _run(tk, run_id)
        return await tk.training.checkpoints(run_id)

    @app.get("/training/runs/{run_id}/samples", tags=["training"], dependencies=[Depends(auth)])
    async def training_samples(run_id: str, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """The validation audio of each checkpoint: step -> files (paths for GET /files/download)."""
        run = _run(tk, run_id)
        return {step: [str(run.dir / f) for f in files] for step, files in sorted(run.info.samples.items(), key=lambda kv: int(kv[0]))}

    @app.post("/training/runs/{run_id}/checkpoints/{step}/retain", tags=["training"], dependencies=[Depends(auth)])
    async def training_retain(run_id: str, step: int, req: CheckpointRetentionRequest, tk: Toolkit = Depends(get_tk)):
        _run(tk, run_id)
        try:
            return tk.training.retain_checkpoint(run_id, step, req.starred)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/training/runs/{run_id}/export", tags=["training"], dependencies=[Depends(auth)])
    async def training_export(run_id: str, req: TrainingExportRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """ONNX export of a checkpoint with the engine's exporter (a job)."""
        run = _run(tk, run_id)
        if req.freeze_spk and req.export_spk:
            raise HTTPException(400, "Choose exported speakers or a frozen speaker, not both")
        if tk.training._busy():
            raise HTTPException(409, "Wait for active training or rendering to finish before exporting")
        out_dir = req.out_dir or str(run.dir / "export" / (str(req.step) if req.step else "latest"))

        async def perform(job: Job):
            data_report = None
            report_value = run.info.spec.config.get("data_report")
            if isinstance(report_value, dict):
                snapshot = run.dir / "dataset-report.json"
                data_report = json.loads(snapshot.read_text(encoding="utf-8")) if snapshot.is_file() else report_value
            else:
                report_path = Path(str(report_value or ""))
                if report_path.is_file():
                    data_report = json.loads(report_path.read_text(encoding="utf-8"))
            export_speakers = req.export_spk
            if not export_speakers and not req.freeze_spk and data_report and run.info.spec.engine_name() not in {"singingvocoders", "g2p", "rvc"}:
                data_speakers = data_report.get("speakers") or []
                if any(s.get("role") == "donor" for s in data_speakers):
                    export_speakers = sorted({s["speaker"] for s in data_speakers if s.get("role") != "donor"})
                    if not export_speakers:
                        raise ValueError("The run contains only donors; select a main speaker before export")
            result = await training_call(tk, job, run.info.spec.engine_name(), "export", {
                "run_dir": str(run.dir), "kind": run.info.spec.kind, "out_dir": out_dir, "step": req.step,
                "export_spk": export_speakers, "freeze_spk": req.freeze_spk, "freeze_gender": req.freeze_gender,
                "freeze_velocity": req.freeze_velocity, "freeze_glide": req.freeze_glide, "freeze_expr": req.freeze_expr,
            })
            if run.info.spec.engine_name() == "singingvocoders":
                result = await training_call(tk, job, "diffsinger", "export_vocoder", {
                    "checkpoint": result["checkpoint"], "out_dir": out_dir,
                })
            if data_report:
                snapshot = Path(out_dir) / "dataset-report.json"
                snapshot.write_text(json.dumps(data_report, ensure_ascii=False, indent=1), encoding="utf-8")
                if isinstance(result, dict):
                    result.setdefault("files", []).append(str(snapshot))
            return result

        async def work(job: Job):
            async with tk.training.inference_slot():
                return await perform(job)

        return submit(tk, "training_export", work, {"run": run_id, **req.model_dump()})

    @app.post("/training/runs/{run_id}/checkpoint-copy", tags=["training"], dependencies=[Depends(auth)])
    async def training_checkpoint_copy(run_id: str, req: CheckpointCopyRequest, tk: Toolkit = Depends(get_tk)):
        run = _run(tk, run_id)
        if not run.info.spec.kind.startswith("diffsinger"):
            raise HTTPException(400, "Speaker filtering requires a DiffSinger checkpoint")
        async def work(job: Job):
            async with tk.training.inference_slot():
                return await training_call(tk, job, run.info.spec.engine_name(), "copy_checkpoint", {
                    "run_dir": str(run.dir), "step": req.step, "retain": req.retain, "fill": req.fill})
        return submit(tk, "checkpoint_copy", work, {"run": run_id, **req.model_dump()})

    @app.post("/training/runs/{run_id}/render", tags=["training"], dependencies=[Depends(auth)])
    async def training_render(run_id: str, req: TrainingRenderRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Sings a .ds file with a checkpoint (a job); the result lists the files made."""
        run = _run(tk, run_id)
        variance_run = _run(tk, req.variance_run) if req.variance_run else None
        if variance_run is not None and (run.info.spec.kind != "diffsinger_acoustic" or variance_run.info.spec.kind != "diffsinger_variance"):
            raise HTTPException(400, "A render chain requires a DiffSinger variance run and an acoustic run")
        if tk.training._busy():
            raise HTTPException(409, "Wait for active training or rendering to finish before rendering a test phrase")
        source = Path(req.ds).expanduser().resolve()
        try:
            if not source.is_file() or source.suffix.lower() != ".ds" or source.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Choose a .ds file smaller than 16 MB on the toolkit computer")
            contents = source.read_text(encoding="utf-8-sig")
            segments = json.loads(contents)
            if not isinstance(segments, list) or not segments or any(not isinstance(s, dict) for s in segments):
                raise ValueError("A .ds phrase must contain a non-empty list of segments")
            if req.title and (Path(req.title).name != req.title or any(c in req.title for c in '/\\:*?\"<>|') or req.title in {".", ".."}):
                raise ValueError("The render title must be a file name without path separators")
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        out_dir = req.out_dir or str(run.dir / "renders" / uuid.uuid4().hex[:12])

        async def work(job: Job):
            async with tk.training.inference_slot():
                folder = Path(out_dir)
                folder.mkdir(parents=True, exist_ok=True)
                snapshot = folder / ("input-" + uuid.uuid4().hex[:8] + ".ds")
                snapshot.write_text(contents, encoding="utf-8")
                render_input = snapshot
                if variance_run is not None:
                    predicted = await training_call(tk, job, variance_run.info.spec.engine_name(), "render", {
                        "run_dir": str(variance_run.dir), "kind": "diffsinger_variance", "ds": str(snapshot),
                        "out_dir": str(folder / "variance"), "step": req.variance_step,
                        "spk": req.spk, "lang": req.lang, "title": "predicted", "predict": req.predict,
                        "seed": req.seed, "device": req.device})
                    ds_files = [Path(p) for p in predicted.get("files", []) if Path(p).suffix.lower() == ".ds"]
                    if len(ds_files) != 1:
                        raise ValueError("Variance inference did not produce exactly one .ds file")
                    render_input = ds_files[0]
                result = await training_call(tk, job, run.info.spec.engine_name(), "render", {
                    "run_dir": str(run.dir), "kind": run.info.spec.kind, "ds": str(render_input), "out_dir": out_dir, "step": req.step,
                    "spk": req.spk, "lang": req.lang, "title": req.title or source.stem, "steps": req.steps, "depth": req.depth,
                    "predict": req.predict, "seed": req.seed, "device": req.device,
                })
                record = {"time": time.time(), "step": req.step, "source": str(source), "input": str(snapshot),
                          "phrase_sha256": hashlib.sha256(json.dumps(segments, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                          "spk": req.spk, "lang": req.lang, "seed": req.seed, "variance_run": req.variance_run,
                          "variance_step": req.variance_step, "predicted_input": str(render_input), **result}
                run.info.renders.append(record)
                run.save(force=True)
                return record

        return submit(tk, "training_render", work, {"run": run_id, **req.model_dump()})

    @app.post("/training/dataset/build", tags=["training"], dependencies=[Depends(auth)])
    async def training_dataset_build(req: DatasetBuildRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Builds a DiffSinger raw dataset (wavs/ + transcriptions.csv per speaker, a dictionary per language) from
        folders of labelled recordings (a job; the result is dataset.json's content plus a run spec to start with)."""
        from ..training.dataset import SourceSpec, build_dataset, spec_from_result  # noqa: PLC0415

        async def work(job: Job):
            def progress(value: float, message: str) -> None:
                job.progress(value, stage="dataset", message=message)

            result = await asyncio.to_thread(
                build_dataset, [SourceSpec(**s.model_dump()) for s in req.sources], Path(req.out_dir),
                req.dictionaries or None, req.validation_files, progress, extra_phonemes=req.extra_phonemes,
                phonemizer_profiles=req.phonemizer_profiles)
            return {"out_dir": result.out_dir, "speakers": result.speakers, "dictionaries": result.dictionaries,
                    "phonemes": result.phonemes, "problems": result.problems, "files": result.files,
                    "seconds": result.seconds, "coverage": result.coverage, "recordings": result.recordings,
                    "spec": spec_from_result(result)}

        return submit(tk, "dataset_build", work, req.model_dump())

    @app.post("/training/dataset/recording-plan", tags=["training"], dependencies=[Depends(auth)])
    async def dataset_recording_plan(req: dict, tk: Toolkit = Depends(get_tk)):
        from ..training.recording_plan import RecordingPlanRequest, recording_plan
        try:
            return await asyncio.to_thread(recording_plan, RecordingPlanRequest.model_validate(req))
        except (ValueError, OSError, TypeError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/training/workflows", tags=["training"], dependencies=[Depends(auth)])
    async def bank_workflows(tk: Toolkit = Depends(get_tk)):
        return sorted(tk.workflows.states.values(), key=lambda s: s["created"], reverse=True)

    @app.post("/training/workflows/check", tags=["training"], dependencies=[Depends(auth)])
    async def bank_workflow_check(req: dict, tk: Toolkit = Depends(get_tk)):
        from ..training.workflows import BankWorkflowRequest, check_workflow
        try:
            return check_workflow(BankWorkflowRequest.model_validate(req))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/training/workflows", tags=["training"], dependencies=[Depends(auth)])
    async def bank_workflow_start(req: dict, tk: Toolkit = Depends(get_tk)):
        from ..training.workflows import BankWorkflowRequest
        try:
            return tk.workflows.start(BankWorkflowRequest.model_validate(req), training_call)
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/training/workflows/{identifier}/{action}", tags=["training"], dependencies=[Depends(auth)])
    async def bank_workflow_action(identifier: str, action: str, tk: Toolkit = Depends(get_tk)):
        if identifier not in tk.workflows.states:
            raise HTTPException(404, "Workflow not found")
        try:
            if action == "stop":
                return await tk.workflows.stop(identifier)
            if action == "resume":
                return tk.workflows.resume(identifier, training_call)
            raise ValueError("Choose stop or resume")
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/training/preparation/cache", tags=["training"], dependencies=[Depends(auth)])
    async def preparation_cache(tk: Toolkit = Depends(get_tk)):
        from ..training.preparation_cache import inventory
        result = []
        for category in ("recording-preparation", "source-pipeline"):
            result.extend({**entry, "key": category + "/" + entry["key"], "category": category}
                          for entry in await asyncio.to_thread(inventory, tk.home.cache / category))
        return result

    @app.post("/training/preparation/cache/remove", tags=["training"], dependencies=[Depends(auth)])
    async def preparation_cache_remove(req: dict, tk: Toolkit = Depends(get_tk)):
        from ..training.preparation_cache import remove
        from ..jobs import FINISHED
        if any(w["status"] in {"running", "stopping"} for w in tk.workflows.states.values()) or tk.training._busy() or any(j.info.status not in FINISHED for j in tk.jobs.jobs.values()):
            raise HTTPException(409, "Wait for active workflows, jobs and training before cleaning preparation cache")
        try:
            grouped = {}
            for key in req.get("keys", []):
                category, separator, identifier = key.partition("/")
                if not separator:
                    category, identifier = "recording-preparation", key
                if category not in {"recording-preparation", "source-pipeline"}:
                    raise ValueError("Unknown preparation cache category")
                grouped.setdefault(category, []).append(identifier)
            removed = 0
            for category, keys in grouped.items():
                removed += await asyncio.to_thread(remove, tk.home.cache / category, keys)
            return {"removed": removed}
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/training/import/check", tags=["training"], dependencies=[Depends(auth)])
    async def project_import_check(req: dict):
        from ..training.migration import inspect_project
        try:
            return await asyncio.to_thread(inspect_project, req["folder"])
        except (ValueError, OSError, KeyError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/training/import", tags=["training"], dependencies=[Depends(auth)])
    async def project_import(req: dict, tk: Toolkit = Depends(get_tk)):
        from ..training.migration import import_project
        async def work(job):
            from ..training.runs import Run, RunInfo
            result = await asyncio.to_thread(import_project, tk, req["folder"], req["project_dir"], req.get("configs"), req.get("copy_data", True), False, req.get("source", "main"))
            for info in result.pop("pending"):
                run = Run(RunInfo.model_validate(info))
                tk.training.runs[run.id] = run
            tk.training._save_index()
            return result
        return submit(tk, "project_import", work, req)

    @app.post("/training/render/external", tags=["training"], dependencies=[Depends(auth)])
    async def external_render(req: dict, tk: Toolkit = Depends(get_tk)):
        if req.get("source", "main") not in {"main", "benchmark"}:
            raise HTTPException(400, "Source must be main or benchmark")
        async def work(job):
            from ..training.migration import stage_external
            folder = tk.home.outputs / job.id
            run_dir = await asyncio.to_thread(stage_external, req["checkpoint"], req["config"], folder / "checkpoint")
            async with tk.training.inference_slot():
                return await training_call(tk, job, "diffsinger_benchmark" if req.get("source") == "benchmark" else "diffsinger", "render",
                    dict(run_dir=str(run_dir), kind=req.get("kind", "diffsinger_acoustic"), ds=req["ds"],
                        out_dir=str(folder / "audio"), steps=req.get("steps", 20), spk=req.get("speaker"), lang=req.get("language"), device=req.get("device", "auto")))
        return submit(tk, "external_checkpoint_render", work, req)

    @app.post("/training/voicebank/render", tags=["training"], dependencies=[Depends(auth)])
    async def onnx_bank_render(req: dict, tk: Toolkit = Depends(get_tk)):
        async def work(job):
            params = {**req, "out_dir": str(tk.home.outputs / job.id)}
            async with tk.training.inference_slot():
                return await training_call(tk, job, "bank_render", "render", params)
        return submit(tk, "bank_onnx_render", work, req)

    @app.post("/training/test-phrase/{action}", tags=["training"], dependencies=[Depends(auth)])
    async def test_phrase(action: str, req: dict, tk: Toolkit = Depends(get_tk)):
        from ..training.test_phrases import PhraseRequest, inventory, build_phrase
        try:
            request = PhraseRequest.model_validate(req)
            if request.run_id:
                request.model_dir = str(_run(tk, request.run_id).dir)
            if not request.model_dir:
                raise ValueError("Select a prepared run or an exported bank")
            if action == "inventory":
                return await asyncio.to_thread(inventory, Path(request.model_dir), request.language, request.config)
            if action == "create":
                return await asyncio.to_thread(build_phrase, request, tk.home.outputs / ("phrase-" + uuid.uuid4().hex[:12]))
            raise ValueError("Choose inventory or create")
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/training/speech/render", tags=["training"], dependencies=[Depends(auth)])
    async def speech_render(req: dict, tk: Toolkit = Depends(get_tk)):
        from ..training.speech_preview import checked_speech_preview
        try:
            params = checked_speech_preview(req)
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error
        kind = params["kind"]
        async def work(job):
            async with tk.training.inference_slot():
                result = await training_call(tk, job, "qwen_tts" if kind == "qwen_tts_voice" else "voxcpm", "render_speech",
                    dict(**params, out_dir=str(tk.home.outputs / job.id / "speech")))
            return result
        return submit(tk, "speech_preview", work, params)

    @app.get("/training/phonemizers", tags=["training"], dependencies=[Depends(auth)])
    async def training_phonemizers() -> list[dict]:
        from ..training.phonemizers import PROFILES
        return list(PROFILES.values())

    @app.post("/training/phonemizers/preview", tags=["training"], dependencies=[Depends(auth)])
    async def phonemizer_preview(req: dict):
        from ..training.phoneme_preview import preview
        try:
            return await asyncio.to_thread(preview, req)
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/training/phonemizers/merge-suggestions", tags=["training"], dependencies=[Depends(auth)])
    async def phonemizer_merges(req: dict):
        from ..training.phoneme_preview import suggest_merges
        try:
            return suggest_merges(req.get("entries", []))
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/training/phonemizers/mapping-suggestions", tags=["training"], dependencies=[Depends(auth)])
    async def phoneme_mappings(req: dict):
        from ..training.phoneme_preview import suggest_mappings
        try:
            return suggest_mappings(req.get("entries", []), req.get("targets", []))
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/training/phonemizers/clip", tags=["training"], dependencies=[Depends(auth)])
    async def phoneme_clip(req: dict, tk: Toolkit = Depends(get_tk)):
        import math
        try:
            start, end = float(req["start"]), float(req["end"])
            if not math.isfinite(start + end) or start < 0 or not start < end or end - start > 30:
                raise ValueError("Choose a finite audio interval up to 30 seconds")
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        async def work(job):
            target = tk.home.outputs / job.id / "phoneme.wav"
            return await training_call(tk, job, "audio", "slice_audio", dict(audio=req["audio"], slices=[dict(start=start, end=end, path=str(target))]))
        return submit(tk, "phoneme_clip", work, req)

    @app.post("/training/dataset/prepare/check", tags=["training"], dependencies=[Depends(auth)])
    async def training_preparation_check(req: RecordingPreparationRequest, tk: Toolkit = Depends(get_tk)) -> dict:
        from ..training.preparation_check import preparation_check
        try:
            return await asyncio.to_thread(preparation_check, tk, req)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/training/dataset/prepare", tags=["training"], dependencies=[Depends(auth)])
    async def training_dataset_prepare(req: RecordingPreparationRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        from ..training.preparation import prepare_recordings

        async def work(job: Job):
            async with tk.training.inference_slot():
                return await prepare_recordings(tk, job, req, training_call)

        return submit(tk, "dataset_prepare", work, req.model_dump())

    @app.post("/training/voicebank", tags=["training"], dependencies=[Depends(auth)])
    async def training_voicebank(req: VoicebankRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Assembles an OpenUtau DiffSinger voicebank from export folders (a job): character files, dsconfig per
        model, dsmain/dsdur/dspitch/dsvariance, embeds, dsdict per language, readme; the result lists the files and
        what the linter found."""
        from ..training.openutau import VoicebankSpec, assemble  # noqa: PLC0415

        async def work(job: Job):
            job.progress(None, stage="voicebank", message="Assembling")
            result = await asyncio.to_thread(assemble, VoicebankSpec(**req.model_dump()))
            return {"dir": result.dir, "files": result.files, "problems": result.problems, "speakers": result.speakers,
                    "languages": result.languages, "models": result.models}

        return submit(tk, "voicebank", work, req.model_dump())

    @app.websocket("/training/runs/{run_id}/events")
    async def training_events_ws(websocket: WebSocket, run_id: str, since: int = 0) -> None:
        tk = tk_holder["tk"]
        token = websocket.query_params.get("token") or websocket.headers.get("x-mvt-token")
        if not _authorized(websocket.client.host if websocket.client else None, token, tk):
            await websocket.close(code=4401)
            return
        run = tk.training.get(run_id)
        await websocket.accept()
        if run is None:
            await websocket.send_json({"event": "error", "message": "run not found"})
            await websocket.close()
            return
        queue = run.subscribe()
        try:
            await websocket.send_json({"event": "state", "run": run.info.model_dump(mode="json")})
            for event in run.read_events_since(since):
                await websocket.send_json(event)
            if run.info.status in RUN_FINISHED:
                return
            while True:
                event = await queue.get()
                await websocket.send_json(event)
                if event.get("event") == "end":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            run.unsubscribe(queue)
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass

    # ---------------- files ----------------

    @app.post("/files", tags=["files"], dependencies=[Depends(auth)])
    async def upload(file: UploadFile = File(...), tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Uploads a file (for servers on another machine). Use the returned file_id in input items."""
        file_id = uuid.uuid4().hex[:16]
        folder = tk.home.uploads / file_id
        folder.mkdir(parents=True)
        name = Path(file.filename or "upload.wav").name
        target = folder / name
        with open(target, "wb") as out:
            shutil.copyfileobj(file.file, out)
        return {"file_id": file_id, "name": name, "size": target.stat().st_size}

    @app.get("/files/download", tags=["files"], dependencies=[Depends(auth)])
    async def download_result(path: str, tk: Toolkit = Depends(get_tk)) -> FileResponse:
        """Downloads a result file of a job (only files inside the toolkit home or listed in job results)."""
        target = Path(path).expanduser().resolve()
        if not target.is_file() or not _allowed_download(tk, target):
            raise HTTPException(404, "File not found")
        return FileResponse(target, filename=target.name)

    # ---------------- operations (jobs) ----------------

    @app.post("/transcribe", tags=["operations"], dependencies=[Depends(auth)])
    async def transcribe(req: TranscribeRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Speech/lyrics recognition (WhisperX: batched inference, VAD to skip silence and noise)."""
        return submit(tk, "transcribe", lambda job: run_transcribe(tk, job, req), req.model_dump())

    @app.post("/align", tags=["operations"], dependencies=[Depends(auth)])
    async def align(req: AlignRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Forced alignment (SOFA or HubertFA, chosen by the model). Items without text are transcribed first (unless transcribe is null).
        With review_transcription=true the job pauses after transcription; resume it with the corrected texts."""
        return submit(tk, "align", lambda job: run_align(tk, job, req), req.model_dump())

    @app.post("/pipelines/label", tags=["operations"], dependencies=[Depends(auth)])
    async def pipeline_label(req: AlignRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Full labelling workflow: transcribe (if needed) -> normalize -> G2P -> align -> rules -> export."""
        return submit(tk, "label", lambda job: run_align(tk, job, req), req.model_dump())

    @app.post("/segment", tags=["operations"], dependencies=[Depends(auth)])
    async def segment(req: SegmentRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Phoneme segmentation without text (WFL-ASR)."""
        return submit(tk, "segment", lambda job: run_segment(tk, job, req), req.model_dump())

    @app.post("/refine", tags=["operations"], dependencies=[Depends(auth)])
    async def refine(req: RefineRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Moves the phoneme boundaries of ready labels to where they are in the sound (a boundary refiner model)."""
        return submit(tk, "refine", lambda job: run_refine(tk, job, req), req.model_dump())

    @app.post("/midi/extract", tags=["operations"], dependencies=[Depends(auth)])
    async def midi_extract(req: MidiRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Notes (MIDI) from singing (GAME). tempo="auto" estimates the BPM first."""
        return submit(tk, "midi", lambda job: run_midi(tk, job, req), req.model_dump())

    @app.post("/separate", tags=["operations"], dependencies=[Depends(auth)])
    async def separate(req: SeparateRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Vocal separation (vocals / accompaniment, lead / backing, de-reverb). Explicit only: no other
        operation separates vocals by itself, because separation can degrade clean recordings."""
        return submit(tk, "separate", lambda job: run_separate(tk, job, req), req.model_dump())

    @app.post("/pitch", tags=["operations"], dependencies=[Depends(auth)])
    async def pitch(req: PitchRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """f0 curve (RMVPE, FCPE, Parselmouth) for piano rolls and pitch editing."""
        return submit(tk, "pitch", lambda job: run_pitch(tk, job, req), req.model_dump())

    @app.post("/resynth", tags=["operations"], dependencies=[Depends(auth)])
    async def resynth(req: ResynthRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """A recording sung again with another f0 (WORLD, or the NSF-HiFiGAN vocoder of DiffSinger)."""
        return submit(tk, "resynth", lambda job: run_resynth(tk, job, req), req.model_dump())

    @app.post("/oto/note", tags=["operations"], dependencies=[Depends(auth)])
    async def oto_note(req: OtoNoteRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """An oto.ini entry sung as a note of the given pitch and length (WORLD, NSF-HiFiGAN or an UTAU
        resampler), to hear how the voicebank will sound with it."""
        return submit(tk, "oto-note", lambda job: run_oto_note(tk, job, req), req.model_dump())

    @app.post("/tempo", tags=["operations"], dependencies=[Depends(auth)])
    async def tempo(req: TempoRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        return submit(tk, "tempo", lambda job: run_tempo(tk, job, req), req.model_dump())

    # ---------------- text (synchronous) ----------------

    @app.post("/text/normalize", tags=["text"], dependencies=[Depends(auth)])
    async def text_normalize(req: TextRequest, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        return await _guard(run_text(tk, None, TextRequest(texts=req.texts, language=req.language, model=None)))

    @app.post("/labels/fix", tags=["text"], dependencies=[Depends(auth)])
    async def labels_fix(req: FixLabelsRequest) -> dict[str, Any]:
        """Applies fixes (rule sets / rules) to existing label files in place, keeping a backup."""
        from ..pipelines.fix import fix_labels  # noqa: PLC0415

        try:
            return await asyncio.to_thread(fix_labels, req)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/text/g2p", tags=["text"], dependencies=[Depends(auth)])
    async def text_g2p(req: TextRequest, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Text -> tokens -> phonemes with the dictionary of an aligner model (and the G2P model of SOFA models)."""
        return await _guard(run_text(tk, None, req))

    @app.post("/text/validate", tags=["text"], dependencies=[Depends(auth)])
    async def text_validate(req: TextRequest, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        """Lists words missing in the model's dictionary before running a long alignment."""
        return await _guard(run_text(tk, None, req, validate_only=True))

    @app.post("/convert", tags=["text"], dependencies=[Depends(auth)])
    async def convert(req: ConvertRequest) -> dict[str, Any]:
        if req.content is None and req.path is None:
            raise HTTPException(400, "content or path is required")
        try:
            if req.path:
                label = formats.read(Path(req.path).expanduser(), req.from_format)
            else:
                label = formats.loads(req.content or "", req.from_format or "htk")
            return {"content": formats.dumps(label, req.to_format, tier=req.tier), "label": label}
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    # ---------------- jobs ----------------

    @app.get("/jobs", tags=["jobs"], dependencies=[Depends(auth)])
    async def jobs(active: bool = False, tk: Toolkit = Depends(get_tk)) -> list[JobInfo]:
        infos = [j.info for j in tk.jobs.jobs.values()]
        if active:
            infos = [i for i in infos if i.status not in FINISHED]
        return sorted(infos, key=lambda i: i.created_at, reverse=True)

    @app.get("/jobs/{job_id}", tags=["jobs"], dependencies=[Depends(auth)])
    async def job(job_id: str, wait: float = 0, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """wait: seconds to wait for the job to finish (long polling)."""
        found = _job(tk, job_id)
        if wait > 0 and found.task is not None and found.info.status not in FINISHED:
            try:
                await asyncio.wait_for(asyncio.shield(found.task), timeout=min(wait, 60))
            except asyncio.TimeoutError:
                pass
        return found.info

    @app.post("/jobs/{job_id}/cancel", tags=["jobs"], dependencies=[Depends(auth)])
    async def cancel(job_id: str, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        found = _job(tk, job_id)
        found.cancel()
        return found.info

    @app.post("/jobs/{job_id}/resume", tags=["jobs"], dependencies=[Depends(auth)])
    async def resume(job_id: str, data: dict[str, Any], tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Continues a paused job. For review_transcription: {"items": [{"name": ..., "text": ...}, ...]}"""
        found = _job(tk, job_id)
        try:
            found.resume(data)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        return found.info

    @app.websocket("/jobs/{job_id}/events")
    async def job_events(websocket: WebSocket, job_id: str) -> None:
        tk = tk_holder["tk"]
        token = websocket.query_params.get("token") or websocket.headers.get("x-mvt-token")
        if not _authorized(websocket.client.host if websocket.client else None, token, tk):
            await websocket.close(code=4401)
            return
        found = tk.jobs.get(job_id)
        await websocket.accept()
        if found is None:
            await websocket.send_json({"event": "error", "message": "job not found"})
            await websocket.close()
            return
        queue = found.subscribe()
        try:
            await websocket.send_json({"event": "state", "job": found.info.model_dump(mode="json")})
            if found.info.status in FINISHED:
                return
            while True:
                event = await queue.get()
                await websocket.send_json(event)
                if event.get("event") == "finished":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            found.unsubscribe(queue)
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass

    return app


def _engine(tk: Toolkit, name: str):
    try:
        return tk.engines.spec(name)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


def _job(tk: Toolkit, job_id: str) -> Job:
    found = tk.jobs.get(job_id)
    if found is None:
        raise HTTPException(404, "Job not found")
    return found


async def _guard(coro):
    try:
        return await coro
    except ModelNotFound as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except EngineError as e:
        raise HTTPException(502, str(e)) from e


def _allowed_download(tk: Toolkit, target: Path) -> bool:
    if target.is_relative_to(tk.home.root):
        return True
    # files of training runs (checkpoints, samples, renders) live in the run folders, often outside the home
    for run in tk.training.runs.values():
        if target.is_relative_to(run.dir):
            return True
    for job in tk.jobs.jobs.values():
        if job.info.result and str(target) in str(job.info.result):
            return True
    return False


_GPU_INFO: list = []


def _gpu_info() -> dict[str, Any] | None:
    """GPU info, asked once per run (GUIs poll /health; running nvidia-smi each time is slow and on Windows
    may flash a console window)."""
    if not _GPU_INFO:
        _GPU_INFO.append(_query_gpu())
    return _GPU_INFO[0]


def _query_gpu() -> dict[str, Any] | None:
    """GPU info without torch (the server itself doesn't depend on it): uses nvidia-smi if available."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, errors="replace", timeout=5,
            creationflags=0x08000000 if os.name == "nt" else 0,
        ).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        return None
    gpus = []
    for line in out:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({"name": parts[0], "memory_total_mb": int(parts[1]), "memory_used_mb": int(parts[2]),
                         "driver": parts[3]})
    return {"gpus": gpus} if gpus else None

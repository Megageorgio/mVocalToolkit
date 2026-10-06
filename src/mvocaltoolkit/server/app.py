"""HTTP API (FastAPI). Interactive documentation: http://<host>:<port>/docs"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import sys
import uuid
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
    FixLabelsRequest,
    MidiRequest,
    ModelImportRequest,
    PitchRequest,
    SegmentRequest,
    SeparateRequest,
    TempoRequest,
    TextRequest,
    TranscribeRequest,
)
from ..engines.manager import EngineError
from ..jobs import FINISHED, Job, JobInfo
from ..models.catalog import ENGINE_TASKS, TASKS
from ..models.store import ModelNotFound, model_listing
from ..pipelines.extra import run_midi, run_pitch, run_segment, run_separate, run_tempo, run_text
from ..pipelines.label import run_align, run_transcribe
from ..text.languages import language_info
from ..text.rules import RULE_SETS
from ..toolkit import Toolkit
from ..update import check_update

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def create_app(toolkit: Toolkit | None = None) -> FastAPI:
    tk_holder: dict[str, Toolkit] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        tk = toolkit or Toolkit()
        tk_holder["tk"] = tk
        await tk.start()
        yield
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
        }

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

    @app.patch("/settings", tags=["system"], dependencies=[Depends(auth)])
    async def patch_settings(values: dict[str, Any], tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        for key, value in values.items():
            if key in ("host", "port", "token"):
                continue  # need a restart / CLI
            if hasattr(tk.settings, key):
                setattr(tk.settings, key, value)
            else:
                tk.settings.extra[key] = value
        tk.home.save_settings(tk.settings)
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

    @app.get("/update/check", tags=["system"], dependencies=[Depends(auth)])
    async def update_check() -> dict[str, Any]:
        return await check_update()

    # ---------------- engines ----------------

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
            installed = await tk.models.download(entry, lambda v, m: job.progress(v, stage="download", message=m))
            return {"installed": [m.id for m in installed]}

        return submit(tk, "model_download", work, {"model": model_id})

    @app.delete("/models/{model_id}", tags=["models"], dependencies=[Depends(auth)])
    async def delete_model(model_id: str, tk: Toolkit = Depends(get_tk)) -> dict[str, Any]:
        if not tk.models.remove(model_id):
            raise HTTPException(404, "Not installed")
        return {"ok": True}

    @app.post("/models/import", tags=["models"], dependencies=[Depends(auth)])
    async def import_model(req: ModelImportRequest, tk: Toolkit = Depends(get_tk)):
        _engine(tk, req.engine)
        try:
            installed = await asyncio.to_thread(
                tk.models.import_local, req.engine, req.path, req.id, req.name, req.languages, req.text_frontend,
                req.copy_files,
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
        """Full LabelMakr-like workflow: transcribe (if needed) -> normalize -> G2P -> align -> rules -> export."""
        return submit(tk, "label", lambda job: run_align(tk, job, req), req.model_dump())

    @app.post("/segment", tags=["operations"], dependencies=[Depends(auth)])
    async def segment(req: SegmentRequest, tk: Toolkit = Depends(get_tk)) -> JobInfo:
        """Phoneme segmentation without text (WFL-ASR)."""
        return submit(tk, "segment", lambda job: run_segment(tk, job, req), req.model_dump())

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
            capture_output=True, text=True, timeout=5,
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

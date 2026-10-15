"""The training manager: a queue of runs, one training at a time, trainers as child processes that outlive the
server.

A run goes: queued -> preparing (the engine installed, assets downloaded, the data binarized) -> training (the
trainer process started; its events.jsonl followed) -> done | failed | stopped. At start the manager reads its
index, finds runs that were training when the server last stopped, and follows them again if their process is
still alive, else marks them stopped (resumable).

The engine takes part through three worker methods:
    prepare(run_dir, kind, spec, assets) -> {"launch": {"args": [...]}, "config": {...}}   (blocking, with progress)
    resume_command(run_dir, kind, spec) -> {"launch": {"args": [...]}}
    checkpoints(run_dir) -> [{"step", "path", "size", "time"}]
and the trainer process itself writes events.jsonl (see runs.py) and honours the STOP file.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from ..engines.manager import EngineError, process_environment
from .runs import ACTIVE, FINISHED, Checkpoint, Run, RunInfo, RunSpec, RunStatus

# no step event for this long while training: the trainer is probably stuck (a warning, not an end)
HANG_SECONDS = 30 * 60
# after a stop request the trainer gets this long to save a checkpoint and end by itself
STOP_GRACE_SECONDS = 180
STEP_WINDOW = 30  # step events kept for the speed estimate


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes  # noqa: PLC0415

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # handles are pointers: without these ctypes truncates them to 32 bits
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        synchronize = 0x00100000
        handle = kernel32.OpenProcess(synchronize, 0, pid)
        if not handle:
            return False
        try:
            # WAIT_TIMEOUT (258): still running
            return kernel32.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                       creationflags=0x08000000)
        return
    import signal  # noqa: PLC0415

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(50):
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class TrainingManager:
    def __init__(self, toolkit: Any):
        self.tk = toolkit
        self.home = toolkit.home
        self.runs: dict[str, Run] = {}
        self._queue_task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._loaded = False
        self._inference = False

    # ---------- index ----------

    @property
    def root(self) -> Path:
        return self.home.dir("training")

    @property
    def index_file(self) -> Path:
        return self.root / "runs.json"

    def _save_index(self) -> None:
        data = [{"id": r.id, "dir": r.info.dir} for r in self.runs.values()]
        temporary = self.index_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(temporary, self.index_file)

    def load(self) -> None:
        """Reads the index and re-attaches to runs that were active when the server last stopped."""
        if self._loaded:
            return
        self._loaded = True
        if self.index_file.exists():
            try:
                entries = json.loads(self.index_file.read_text(encoding="utf-8"))
            except ValueError:
                entries = []
            for entry in entries:
                run_dir = Path(entry.get("dir", ""))
                if not (run_dir / "run.json").exists():
                    continue
                try:
                    run = Run.load(run_dir)
                except Exception:  # noqa: BLE001
                    continue
                self.runs[run.id] = run
        for run in self.runs.values():
            info = run.info
            # run.json can lag behind the append-only event file. Consume only the events not included in
            # its saved counter, retaining an incomplete final line for the follower. This also positions
            # stopped runs correctly before a later resume (old end events must not end the new trainer).
            events = run.read_new_events()
            for event in events[info.events:]:
                self._apply(run, event)
                info.events += 1
            if info.status in FINISHED:
                info.pid = None
            if info.status in (RunStatus.training, RunStatus.stopping):
                if info.pid and _pid_alive(info.pid):
                    self._follow(run)
                else:
                    self._interrupted(run, "The trainer process is gone (the toolkit or the computer was restarted)")
            elif info.status in (RunStatus.preparing, RunStatus.queued):
                # preparation can't be resumed in the middle: it starts again
                info.status = RunStatus.queued
                info.stage, info.progress, info.message = "", None, ""
                run.save(force=True)
            run.save(force=True)
        self._ensure_queue()

    def _interrupted(self, run: Run, why: str) -> None:
        run.info.status = RunStatus.stopped
        run.info.finished_at = time.time()
        run.info.pid = None
        run.info.message = why
        run.stop_file.unlink(missing_ok=True)
        run.save(force=True)
        run.publish(run.emit("end", status="stopped", error=why))

    # ---------- public ----------

    def list(self) -> list[RunInfo]:
        self.load()
        return sorted((r.info for r in self.runs.values()), key=lambda i: i.created_at, reverse=True)

    def get(self, run_id: str) -> Run | None:
        self.load()
        return self.runs.get(run_id)

    def start(self, spec: RunSpec) -> Run:
        self.load()
        run_id = uuid.uuid4().hex[:10]
        run_dir = Path(spec.dir).expanduser().resolve() if spec.dir else self.root / run_id
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"The run folder is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = spec.model_copy(update={"dir": str(run_dir)})
        info = RunInfo(id=run_id, spec=spec, dir=str(run_dir),
                       queue_order=max((r.info.queue_order for r in self.runs.values()), default=0) + 1)
        run = Run(info)
        self.runs[run_id] = run
        run.save(force=True)
        self._save_index()
        run.publish(run.emit("queued", run=run_id))
        self._ensure_queue()
        return run

    async def stop(self, run_id: str) -> RunInfo:
        run = self._require(run_id)
        info = run.info
        if info.status == RunStatus.queued:
            info.status = RunStatus.stopped
            info.finished_at = time.time()
            run.save(force=True)
            run.publish(run.emit("end", status="stopped"))
            self._wake.set()
            return info
        if info.status == RunStatus.preparing:
            # binarizing runs inside the engine worker: the worker is stopped, which ends it
            info.status = RunStatus.stopping
            run.save(force=True)
            await self.tk.engines.abort(info.spec.engine_name())
            return info
        if info.status == RunStatus.training:
            info.status = RunStatus.stopping
            info.message = "Stopping: the trainer saves a checkpoint and ends"
            run.save(force=True)
            run.stop_file.write_text(str(time.time()), encoding="utf-8")
            run.publish(run.emit("stopping"))
            asyncio.create_task(self._enforce_stop(run))
        return info

    async def _enforce_stop(self, run: Run) -> None:
        deadline = time.time() + STOP_GRACE_SECONDS
        while time.time() < deadline:
            await asyncio.sleep(1)
            if run.info.status != RunStatus.stopping:
                return
        pid = run.info.pid
        if pid and _pid_alive(pid):
            await asyncio.to_thread(_kill_tree, pid)
            run.info.error = "The trainer didn't stop by itself; it was killed"

    async def resume(self, run_id: str) -> RunInfo:
        run = self._require(run_id)
        info = run.info
        if info.status in ACTIVE:
            raise ValueError("The run is active")
        if not run.dir.exists():
            raise ValueError("The run folder is gone")
        # Prepare-only and never-launched runs have no follower to consume their end event.
        # Drain that history before setting the new queued state.
        for event in run.read_new_events():
            self._apply(run, event)
            info.events += 1
        info.status = RunStatus.queued
        info.queue_order = max((r.info.queue_order for r in self.runs.values()), default=0) + 1
        info.error, info.error_kind = None, None
        info.finished_at = None
        info.message = ""
        info.speed, info.eta = None, None
        run.stop_file.unlink(missing_ok=True)
        run.save(force=True)
        run.publish(run.emit("queued", run=run_id, resume=True))
        self._ensure_queue()
        return info

    async def delete(self, run_id: str, files: bool = False) -> None:
        run = self._require(run_id)
        if run.info.status in (RunStatus.training, RunStatus.stopping, RunStatus.preparing):
            await self.stop(run_id)
            pid = run.info.pid
            if pid and _pid_alive(pid):
                await asyncio.to_thread(_kill_tree, pid)
        if run.task is not None:
            run.task.cancel()
        self.runs.pop(run_id, None)
        self._save_index()
        if files:
            shutil.rmtree(run.dir, ignore_errors=True)

    def reorder_queue(self, ids: list[str]) -> None:
        self.load()
        queued = {r.id for r in self.runs.values() if r.info.status == RunStatus.queued}
        if len(ids) != len(set(ids)) or set(ids) != queued:
            raise ValueError("Include every currently queued run exactly once; refresh the queue and try again")
        for order, run_id in enumerate(ids):
            run = self.runs[run_id]
            run.info.queue_order = order
            run.save(force=True)
        self._wake.set()

    def events_since(self, run_id: str, since: int) -> list[dict[str, Any]]:
        return self._require(run_id).read_events_since(since)

    async def checkpoints(self, run_id: str) -> list[Checkpoint]:
        run = self._require(run_id)
        if run.info.spec.engine_name() in {"diffsinger", "diffsinger_benchmark", "singingvocoders", "g2p", "rvc", "tifa_train", "sofa_train", "game_train", "label_train", "refiner_train", "qwen_tts", "voxcpm"}:
            from ..engines.diffsinger.checkpoint_store import listing

            divisor = 2 if run.info.spec.engine_name() == "singingvocoders" else 1
            run.info.checkpoints = [Checkpoint.model_validate({**c, "step": c["step"] // divisor}) for c in listing(run.dir)]
            run.save(force=True)
            return run.info.checkpoints
        try:
            found = await self.tk.engines.call(run.info.spec.engine_name(), "checkpoints", {"run_dir": str(run.dir)})
        except EngineError:
            found = []
        if found:
            run.info.checkpoints = [Checkpoint.model_validate(c) for c in found]
            run.save()
        return run.info.checkpoints

    def retain_checkpoint(self, run_id: str, step: int, keep: bool) -> RunInfo:
        run = self._require(run_id)
        if run.info.spec.engine_name() not in {"diffsinger", "diffsinger_benchmark", "singingvocoders", "g2p", "rvc", "tifa_train", "sofa_train", "game_train", "label_train", "refiner_train", "qwen_tts", "voxcpm"}:
            raise ValueError("Checkpoint retention is supported for DiffSinger and SingingVocoders runs")
        from ..engines.diffsinger.checkpoint_store import listing, retain

        divisor = 2 if run.info.spec.engine_name() == "singingvocoders" else 1
        retain(run.dir, step * divisor, keep)
        run.info.checkpoints = [Checkpoint.model_validate({**c, "step": c["step"] // divisor}) for c in listing(run.dir)]
        run.save(force=True)
        return run.info

    def _require(self, run_id: str) -> Run:
        run = self.get(run_id)
        if run is None:
            raise KeyError(f"Unknown run {run_id}")
        return run

    # ---------- the queue ----------

    def _ensure_queue(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (the CLI loads the index without serving): started later
        if self._queue_task is None or self._queue_task.done():
            self._queue_task = loop.create_task(self._queue_loop())
        self._wake.set()

    def _busy(self) -> bool:
        return self._inference or any(r.info.status in (RunStatus.preparing, RunStatus.training, RunStatus.stopping) or r.info.pid and self._alive(r) for r in self.runs.values())

    @asynccontextmanager
    async def inference_slot(self):
        if self._busy():
            raise ValueError("Wait for active training or rendering to finish before rendering a test phrase")
        self._inference = True
        try:
            yield
        finally:
            self._inference = False
            self._wake.set()

    async def _queue_loop(self) -> None:
        while True:
            self._wake.clear()
            if not self._busy():
                queued = [r for r in self.runs.values() if r.info.status == RunStatus.queued]
                if queued:
                    run = min(queued, key=lambda r: (r.info.queue_order, r.info.created_at))
                    await self._prepare_and_launch(run)
                    continue
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

    async def _prepare_and_launch(self, run: Run) -> None:
        info = run.info
        engine = info.spec.engine_name()
        info.status = RunStatus.preparing
        info.started_at = info.started_at or time.time()
        info.stage = "install"
        info.message = f"Getting the {engine} engine ready"
        run.save(force=True)
        run.publish(run.emit("prepare", stage="install", progress=None, message=info.message))
        try:
            spec = self.tk.engines.spec(engine)
            info.engine_version = spec.source.commit[:10] if spec.source else ""
            assets = await self._assets(run)
            if info.status != RunStatus.preparing:
                self._interrupted(run, "Stopped while preparing")
                return
            resumed = (bool(info.checkpoints) or (run.dir / "PREPARED").is_file()) and (run.dir / "config.yaml").exists()
            if resumed:
                # a resume: the config and the binary data are there
                result = await self.tk.engines.call(engine, "resume_command", {
                    "run_dir": str(run.dir), "kind": info.spec.kind, "spec": info.spec.config})
            else:
                if info.spec.kind.startswith("diffsinger_"):
                    from .resources import resource_options, measured_limits
                    resources = resource_options(info.spec.config)
                    if resources["auto_batch"]:
                        environment = await self.tk.engines.call(engine, "environment", {})
                        limits, measurement = measured_limits(info.spec.kind, info.spec.config, environment, info.spec.device)
                        info.spec.config["overlay"] = {**info.spec.config.get("overlay", {}), **limits}
                        (run.dir / "resource-report.json").write_text(json.dumps(measurement, indent=1), encoding="utf-8")
                        run.publish(run.emit("resource", **measurement))
                def on_progress(data: dict[str, Any]) -> None:
                    stage = data.get("stage") or "prepare"
                    info.stage = stage
                    info.progress = data.get("progress")
                    info.message = str(data.get("message") or "")[:200]
                    run.save()
                    run.publish(run.emit("prepare", stage=stage, progress=info.progress, message=info.message))

                result = await self.tk.engines.call(engine, "prepare", {
                    "run_dir": str(run.dir), "kind": info.spec.kind, "spec": info.spec.config, "assets": assets,
                    "device": info.spec.device,
                }, on_progress=on_progress, log=lambda line: None)
                info.config_written = result.get("config") or {}
                info.max_steps = result.get("max_steps") or info.config_written.get("max_updates") or info.max_steps
            if info.status != RunStatus.preparing:
                # stopped while preparing
                self._interrupted(run, "Stopped while preparing")
                return
            if not resumed:
                (run.dir / "PREPARED").write_text("ready", encoding="utf-8")
            if info.spec.config.get("prepare_only") and not resumed:
                info.status = RunStatus.done
                info.finished_at = time.time()
                info.message = "Data prepared; resume this run to start training"
                info.stage, info.progress = "prepared", 1.0
                run.save(force=True)
                run.publish(run.emit("end", status="done"))
                return
            launch = result.get("launch") or {}
            self._launch(run, spec, launch.get("args") or [], resumed)
        except Exception as e:  # noqa: BLE001
            if info.status == RunStatus.stopping:
                self._interrupted(run, "Stopped while preparing")
                return
            info.status = RunStatus.failed
            info.finished_at = time.time()
            info.error = str(e)
            info.error_kind = "oom" if isinstance(e, EngineError) and e.oom else "prepare"
            info.stage, info.progress = "", None
            run.save(force=True)
            run.publish(run.emit("end", status="failed", error=info.error, error_kind=info.error_kind))
        finally:
            self._wake.set()

    async def _assets(self, run: Run) -> dict[str, str]:
        """Models the engine needs (vocoder, pitch extractor...): downloaded when missing; id -> folder."""
        info = run.info
        wanted: dict[str, str] = dict(info.spec.config.get("assets") or {})
        engine = info.spec.engine_name()
        resolved: dict[str, str] = {}
        for key, model_id in wanted.items():
            if not model_id:
                continue
            path = Path(model_id)
            if path.exists():
                resolved[key] = str(path.resolve())
                continue

            def progress(value: float | None, message: str, _detail: Any = None, k=key) -> None:
                info.stage = "download"
                info.progress = value
                info.message = message[:200]
                run.save()
                run.publish(run.emit("prepare", stage="download", progress=value, message=info.message))

            model = await self.tk.models.require("diffsinger" if engine == "diffsinger_benchmark" else engine, model_id, progress)
            ckpt = model.file("checkpoint")
            resolved[key] = str(ckpt if ckpt is not None else model.path)
        return resolved

    def _launch(self, run: Run, spec: Any, args: list[str], resumed: bool) -> None:
        info = run.info
        python, cwd, env = process_environment(spec, self.tk.engines.envs, self.tk.settings, device=info.spec.device)
        env["MVT_RUN_DIR"] = str(run.dir)
        run.stop_file.unlink(missing_ok=True)
        log = (run.dir / "train.log").open("ab")
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} starting: {python} {' '.join(args)}\n".encode())
        log.flush()
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            # its own process group and no console: it keeps running when the server stops
            kwargs["creationflags"] = 0x00000200 | 0x08000000  # CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(
            [str(python), "-u", *args], cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, close_fds=True, **kwargs,
        )
        log.close()
        run.process = process
        info.pid = process.pid
        info.status = RunStatus.training
        info.stage, info.progress, info.message = "", None, ""
        info.error, info.error_kind = None, None
        run.step_times.clear()
        run.last_event_time = time.time()
        run.save(force=True)
        run.publish(run.emit("launched", pid=process.pid, resumed=resumed))
        self._follow(run)

    # ---------- following a trainer ----------

    def _follow(self, run: Run) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (the CLI reading the state): nothing to follow here
        if run.task is None or run.task.done():
            run.task = loop.create_task(self._follow_loop(run))

    def _alive(self, run: Run) -> bool:
        if run.process is not None:
            return run.process.poll() is None
        return bool(run.info.pid) and _pid_alive(run.info.pid or 0)

    async def _follow_loop(self, run: Run) -> None:
        info = run.info
        ended = False
        terminal_status = None
        hang_warned = False
        try:
            while True:
                events = run.read_new_events()
                for event in events:
                    info.events += 1
                    self._apply(run, event)
                    run.publish(event)
                    if event.get("event") == "end":
                        ended = True
                        terminal_status = info.status
                        info.status = RunStatus.stopping if terminal_status == RunStatus.stopped else RunStatus.training
                if events:
                    run.last_event_time = time.time()
                    run.save()
                if ended:
                    break
                if not self._alive(run):
                    # the process is gone without saying why: give the file a last moment, then report
                    await asyncio.sleep(1)
                    events = run.read_new_events()
                    for event in events:
                        info.events += 1
                        self._apply(run, event)
                        run.publish(event)
                        if event.get("event") == "end":
                            ended = True
                            terminal_status = info.status
                            info.status = RunStatus.stopping if terminal_status == RunStatus.stopped else RunStatus.training
                    if not ended:
                        self._died(run)
                    break
                if info.status == RunStatus.training and not hang_warned and time.time() - run.last_event_time > HANG_SECONDS:
                    hang_warned = True
                    warning = {"kind": "hang", "text": f"No news from the trainer for {HANG_SECONDS // 60} minutes"}
                    run.publish(run.emit("warning", **warning))
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            # the server stops while the trainer keeps running: the pid stays in run.json for the next start
            if terminal_status is not None: info.status = terminal_status
            run.save(force=True)
            raise
        # the trainer ended
        # A terminal event may precede CUDA teardown. Do not retry on top of the old process.
        for _ in range(100):
            if not self._alive(run): break
            await asyncio.sleep(.1)
        if self._alive(run):
            self._interrupted(run, "Trainer has not exited after its final event; resume after it exits")
            return
        for event in run.read_new_events():
            info.events += 1
            self._apply(run, event)
            run.publish(event)
            if event.get("event") == "end": terminal_status = info.status
        if terminal_status is not None: info.status = terminal_status
        run.stop_file.unlink(missing_ok=True)
        info.pid = None
        run.process = None
        from .resources import reduce_after_oom
        try:
            recovery = reduce_after_oom(run)
        except (ValueError, OSError, TypeError) as error:
            recovery = None
            run.publish(run.emit("warning", kind="oom_retry", text=f"Cannot change batch limits: {error}"))
        if recovery:
            info.warnings = (info.warnings + [recovery])[-50:]
            info.status, info.error, info.error_kind = RunStatus.queued, None, None
            info.finished_at = None
            info.message = recovery["text"]
            run.publish(run.emit("warning", **recovery))
            run.publish(run.emit("queued", run=run.id, resume=True))
        run.save(force=True)
        self._wake.set()

    def _died(self, run: Run) -> None:
        info = run.info
        code = run.process.returncode if run.process is not None else None
        tail = run.log_tail(40)
        text = "\n".join(tail)
        lowered = text.lower()
        info.status = RunStatus.failed if info.status != RunStatus.stopping else RunStatus.stopped
        info.finished_at = time.time()
        if "out of memory" in lowered or "outofmemory" in lowered:
            info.error_kind = "oom"
            info.error = "The GPU ran out of memory"
        else:
            info.error_kind = "crash"
            info.error = f"The trainer ended unexpectedly (exit code {code})" if code is not None else "The trainer process ended unexpectedly"
        if tail:
            info.error += ":\n" + "\n".join(tail[-12:])
        run.publish(run.emit("end", status=info.status.value, error=info.error, error_kind=info.error_kind, exit_code=code))

    def _apply(self, run: Run, event: dict[str, Any]) -> None:
        """Updates the run's numbers from one event."""
        info = run.info
        kind = event.get("event")
        now = float(event.get("time") or time.time())
        if kind == "start":
            info.max_steps = event.get("max_steps") or info.max_steps
            if event.get("step") is not None:
                info.step = int(event["step"])
        elif kind == "step":
            step = int(event.get("step") or 0)
            info.step = step
            if event.get("max_steps"):
                info.max_steps = int(event["max_steps"])
            if event.get("epoch") is not None:
                info.epoch = int(event["epoch"])
            losses = event.get("losses") or {}
            if isinstance(losses, dict):
                info.losses = {k: float(v) for k, v in losses.items() if isinstance(v, (int, float)) and math.isfinite(v)}
            if event.get("lr") is not None:
                info.lr = float(event["lr"])
            if event.get("gpu_mb") is not None:
                info.gpu_mb = float(event["gpu_mb"])
            run.step_times.append((now, step))
            del run.step_times[:-STEP_WINDOW]
            if len(run.step_times) >= 2:
                (t0, s0), (t1, s1) = run.step_times[0], run.step_times[-1]
                if t1 > t0 and s1 > s0:
                    info.speed = (s1 - s0) / (t1 - t0)
                    if info.max_steps:
                        info.eta = max(0.0, (info.max_steps - s1) / info.speed)
        elif kind == "validation":
            step = int(event.get("step") or info.step)
            loss = event.get("val_loss")
            if isinstance(loss, (int, float)) and math.isfinite(loss):
                info.val_loss = float(loss)
                info.val_step = step
                if info.best_val_loss is None or info.val_loss < info.best_val_loss:
                    info.best_val_loss, info.best_val_step = info.val_loss, step
                train = info.losses.get("total_loss")
                if train is None:
                    train = info.losses.get("loss")
                if train is not None:
                    info.validation_history = [h for h in info.validation_history if h["step"] < step]
                    info.validation_history.append({"step": float(step), "loss": float(loss), "train": train})
                    info.validation_history = info.validation_history[-12:]
                    recent = info.validation_history[-4:]
                    if (len(recent) == 4 and info.best_val_loss is not None
                            and all(h["loss"] > info.best_val_loss * 1.05 for h in recent)
                            and recent[-1]["train"] < recent[0]["train"] * 0.95
                            and all(recent[i]["loss"] <= recent[i + 1]["loss"] for i in range(3))
                            and not any(w.get("kind") == "overfit" for w in info.warnings)):
                        warning = {"kind": "overfit", "text": "Validation loss has risen across four checks while training loss fell. Compare samples with an earlier checkpoint; this is a hint, not an automatic stop."}
                        # Persist the marker immediately so multiple validation events in a batch do not repeat it.
                        info.warnings.append(warning)
                        run.emit("warning", **warning)
        elif kind == "checkpoint":
            ckpt = Checkpoint(step=int(event.get("step") or 0), path=str(event.get("path") or ""),
                              size=int(event.get("size") or 0), time=now)
            if info.spec.engine_name() in {"diffsinger", "diffsinger_benchmark", "singingvocoders", "g2p", "rvc", "tifa_train", "sofa_train", "game_train", "label_train", "refiner_train", "qwen_tts", "voxcpm"}:
                from ..engines.diffsinger.checkpoint_store import retained_path

                divisor = 2 if info.spec.engine_name() == "singingvocoders" else 1
                ckpt.starred = retained_path(run.dir, ckpt.step * divisor).is_file()
            info.checkpoints = [c for c in info.checkpoints if c.step != ckpt.step] + [ckpt]
            info.checkpoints.sort(key=lambda c: c.step)
        elif kind == "checkpoint_removed":
            step = int(event.get("step") or -1)
            if info.spec.engine_name() in {"diffsinger", "diffsinger_benchmark", "singingvocoders", "g2p", "rvc", "tifa_train", "sofa_train", "game_train", "label_train", "refiner_train", "qwen_tts", "voxcpm"}:
                from ..engines.diffsinger.checkpoint_store import retained_path

                divisor = 2 if info.spec.engine_name() == "singingvocoders" else 1
                backup = retained_path(run.dir, step * divisor) if step >= 0 else None
                if backup is not None and not backup.is_file():
                    backup = run.dir / "best" / backup.name
                if backup is not None and backup.is_file():
                    for ckpt in info.checkpoints:
                        if ckpt.step == step:
                            ckpt.path = str(backup)
                            ckpt.starred = retained_path(run.dir, step * divisor).is_file()
                    return
            info.checkpoints = [c for c in info.checkpoints if c.step != step]
        elif kind == "samples":
            step = str(event.get("step"))
            files = [str(f) for f in (event.get("files") or [])]
            info.samples[step] = sorted(set(info.samples.get(step, [])) | set(files))
        elif kind == "warning":
            warning = {k: v for k, v in event.items() if k not in ("event", "time")}
            if warning not in info.warnings:
                info.warnings.append(warning)
        elif kind == "end":
            status = str(event.get("status") or "done")
            try:
                info.status = RunStatus(status)
            except ValueError:
                info.status = RunStatus.failed
            if info.status == RunStatus.stopping:
                info.status = RunStatus.stopped
            info.finished_at = now
            info.error = event.get("error")
            info.error_kind = event.get("error_kind")
            info.speed, info.eta = None, None
            if info.status == RunStatus.done and info.stage != "prepared":
                info.message = "Finished"
            run.stop_file.unlink(missing_ok=True)

    async def stop_all(self) -> None:
        """The server stops: trainers keep running on their own; only the followers end."""
        for run in self.runs.values():
            if run.task is not None:
                run.task.cancel()
            run.save(force=True)
        if self._queue_task is not None:
            self._queue_task.cancel()

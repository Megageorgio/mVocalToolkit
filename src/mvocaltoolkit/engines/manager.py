"""Engine worker processes.

Each engine runs as a separate process with its own Python environment and talks JSON lines over stdin/stdout
(see _runtime/mvt_engine.py). Workers are started on first use, keep their models in memory between requests,
and are stopped after a period of inactivity to free (GPU) memory.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
from pathlib import Path
from typing import Any, Callable

from ..settings import Home, Settings
from .env import NO_WINDOW, EnvManager
from .spec import ENGINES_DIR, EngineSpec, load_engine_specs

RUNTIME_DIR = Path(__file__).parent / "_runtime"

ProgressFunc = Callable[[dict[str, Any]], None]


def _code_hint(code: int | None) -> str:
    """A few exit codes that say more than the number (Windows status codes and POSIX signals)."""
    if code is None:
        return ""
    known = {
        3221225477: "access violation: a crash inside native code (torch, CUDA or a DLL)",
        -1073741819: "access violation: a crash inside native code (torch, CUDA or a DLL)",
        3221225725: "stack overflow",
        -1073741571: "stack overflow",
        3221226505: "a native library aborted",
        -1073740791: "a native library aborted",
        3221225781: "a DLL was not found",
        -1073741515: "a DLL was not found",
        -9: "killed, often for running out of memory",
        -11: "segmentation fault: a crash inside native code",
        -6: "aborted by a native library",
        137: "killed, often for running out of memory",
    }
    hint = known.get(code)
    return f": {hint}" if hint else ""


class EngineError(RuntimeError):
    def __init__(self, engine: str, error: dict[str, Any]):
        self.engine = engine
        self.type = error.get("type", "Error")
        self.oom = bool(error.get("oom"))
        self.trace = error.get("trace", "")
        super().__init__(f"[{engine}] {self.type}: {error.get('message', '')}")


class EngineNotInstalled(RuntimeError):
    pass


class WorkerProcess:
    def __init__(self, spec: EngineSpec, envs: EnvManager, settings: Settings):
        self.spec = spec
        self.envs = envs
        self.settings = settings
        self.process: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending: dict[int, tuple[asyncio.Future[Any], ProgressFunc | None]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=200)
        self.last_used = 0.0
        # one request at a time per engine (models are on one GPU)
        self.lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(self) -> None:
        if self.running:
            return
        python = self.envs.python(self.spec)
        paths = [str(RUNTIME_DIR), str(self.spec.dir)]
        cwd = self.spec.dir
        source = self.envs.source_dir(self.spec)
        if source is not None and self.spec.source is not None:
            src = (source / self.spec.source.path).resolve()
            paths.insert(0, str(src))
            cwd = src
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # text files are read as UTF-8 even by libraries that do not say so (Windows would use its ANSI code page)
        env["PYTHONUTF8"] = "1"
        env["MVT_DEVICE"] = self.settings.device
        env["MVT_HOME"] = str(self.envs.home.root)
        env["MVT_CACHE"] = str(self.envs.home.cache)
        if self.settings.hf_token:
            env.setdefault("HF_TOKEN", self.settings.hf_token)
        # keep model caches (HF, torch hub) inside the toolkit home
        env.setdefault("HF_HOME", str(self.envs.home.cache / "huggingface"))
        env.setdefault("TORCH_HOME", str(self.envs.home.cache / "torch"))
        import datetime  # noqa: PLC0415

        self.stderr_tail.clear()
        self._log(f"--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S} starting {self.spec.name} ({python})")
        self.process = await asyncio.create_subprocess_exec(
            str(python),
            "-u",
            str(self.spec.dir / self.spec.worker),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
            limit=64 * 1024 * 1024,
            **NO_WINDOW,
        )
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._reader = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        try:
            await asyncio.wait_for(asyncio.shield(self._ready), timeout=300)
        except asyncio.TimeoutError:
            await self.stop()
            raise RuntimeError(f"Engine {self.spec.name} did not start:\n" + "\n".join(self.stderr_tail) + f"\nFull output: {self.log_path()}")

    async def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            async for raw in self.process.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self.stderr_tail.append(line)
                    self._log(line)
                    continue
                if message.get("event") == "ready":
                    if self._ready and not self._ready.done():
                        self._ready.set_result(None)
                    continue
                request_id = message.get("id")
                entry = self._pending.get(request_id)
                if entry is None:
                    continue
                future, on_progress = entry
                if "event" in message:
                    if on_progress is not None:
                        try:
                            on_progress(message.get("data") or {})
                        except Exception:  # noqa: BLE001
                            pass
                    continue
                self._pending.pop(request_id, None)
                if future.done():
                    continue
                if "error" in message:
                    future.set_exception(EngineError(self.spec.name, message["error"]))
                else:
                    future.set_result(message.get("result"))
        finally:
            waiting = any(not f.done() for f, _ in self._pending.values()) or (self._ready is not None and not self._ready.done())
            # a normal stop has nobody waiting: no report then
            error = RuntimeError(await self._death_report()) if waiting else RuntimeError(f"Engine {self.spec.name} stopped")
            for future, _ in self._pending.values():
                if not future.done():
                    future.set_exception(error)
            self._pending.clear()
            if self._ready and not self._ready.done():
                self._ready.set_exception(error)

    def log_path(self) -> Path:
        """Everything the engine printed (errors, warnings, downloads), kept between runs."""
        return self.envs.home.logs / "engines" / f"{self.spec.name}.log"

    def _log(self, line: str) -> None:
        try:
            path = self.log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    async def _death_report(self) -> str:
        """Why the worker is gone: its exit code and the end of its output (read to the end first)."""
        process = self.process
        if self._stderr_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._stderr_task), timeout=5)
            except Exception:  # noqa: BLE001
                pass
        code = None
        if process is not None:
            try:
                code = await asyncio.wait_for(process.wait(), timeout=5)
            except Exception:  # noqa: BLE001
                code = process.returncode
        tail = "\n".join(list(self.stderr_tail)[-60:]).strip()
        lines = [f"Engine {self.spec.name} stopped unexpectedly (exit code {code}{_code_hint(code)})."]
        lines.append(tail if tail else "It printed nothing before stopping.")
        lines.append(f"Full output: {self.log_path()}")
        report = "\n".join(lines)
        self._log(report)
        return report

    async def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        async for raw in self.process.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            self.stderr_tail.append(line)
            self._log(line)

    async def call(self, method: str, params: dict[str, Any] | None = None, on_progress: ProgressFunc | None = None):
        await self.start()
        assert self.process is not None and self.process.stdin is not None
        loop = asyncio.get_running_loop()
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = (future, on_progress)
        payload = json.dumps({"id": request_id, "method": method, "params": params or {}}, ensure_ascii=False)
        self.process.stdin.write((payload + "\n").encode("utf-8"))
        await self.process.stdin.drain()
        self.last_used = loop.time()
        try:
            return await future
        finally:
            self.last_used = loop.time()

    async def stop(self) -> None:
        if self.process is None:
            return
        process = self.process
        self.process = None
        if process.returncode is None:
            try:
                assert process.stdin is not None
                process.stdin.write(b'{"id": 0, "method": "shutdown"}\n')
                await process.stdin.drain()
                await asyncio.wait_for(process.wait(), timeout=10)
            except Exception:  # noqa: BLE001
                process.kill()
                await process.wait()
        for task in (self._reader, self._stderr_task):
            if task is not None:
                task.cancel()


class EngineManager:
    def __init__(self, home: Home, settings: Settings, extra_dirs: list[Path] | None = None):
        self.home = home
        self.settings = settings
        self.envs = EnvManager(home, settings)
        self.specs = load_engine_specs(ENGINES_DIR, *(extra_dirs or []))
        self.workers: dict[str, WorkerProcess] = {}
        self._idle_task: asyncio.Task[None] | None = None

    def spec(self, name: str) -> EngineSpec:
        if name not in self.specs:
            raise KeyError(f"Unknown engine: {name}")
        return self.specs[name]

    def list(self) -> list[dict[str, Any]]:
        result = []
        for spec in self.specs.values():
            worker = self.workers.get(spec.name)
            result.append(
                {
                    "name": spec.name,
                    "title": spec.title,
                    "description": spec.description,
                    "capabilities": spec.capabilities,
                    "size_hint": spec.size_hint,
                    "homepage": spec.homepage,
                    "license": spec.license,
                    **self.envs.status(spec),
                    "running": bool(worker and worker.running),
                }
            )
        return result

    async def ensure_installed(self, name: str, log: Callable[[str], None] | None = None) -> EngineSpec:
        spec = self.spec(name)
        if not self.envs.is_ready(spec):
            if not self.settings.auto_install_engines:
                raise EngineNotInstalled(f"Engine {name} is not installed")
            await self.envs.install(spec, log=log)
        return spec

    async def call(
        self,
        engine: str,
        method: str,
        params: dict[str, Any] | None = None,
        on_progress: ProgressFunc | None = None,
        log: Callable[[str], None] | None = None,
    ) -> Any:
        install_log = log
        if on_progress is not None and not self.envs.is_ready(self.spec(engine)):
            # the first use installs the engine (minutes, GBs for torch): show it as the job's progress
            def install_log(line: str) -> None:
                if log is not None:
                    log(line)
                on_progress({"stage": "install", "message": f"{engine}: {line.strip()[:120]}"})

        spec = await self.ensure_installed(engine, log=install_log)
        worker = self.workers.get(engine)
        if worker is None:
            worker = self.workers[engine] = WorkerProcess(spec, self.envs, self.settings)
        async with worker.lock:
            return await worker.call(method, params, on_progress)

    async def stop(self, engine: str) -> None:
        worker = self.workers.pop(engine, None)
        if worker is not None:
            await worker.stop()

    async def stop_all(self) -> None:
        for name in list(self.workers):
            await self.stop(name)
        if self._idle_task is not None:
            self._idle_task.cancel()

    def start_idle_watcher(self) -> None:
        if self._idle_task is None and self.settings.engine_idle_timeout > 0:
            self._idle_task = asyncio.create_task(self._idle_loop())

    async def _idle_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(30)
            for name, worker in list(self.workers.items()):
                idle = loop.time() - worker.last_used
                if worker.running and not worker.lock.locked() and idle > self.settings.engine_idle_timeout:
                    await self.stop(name)

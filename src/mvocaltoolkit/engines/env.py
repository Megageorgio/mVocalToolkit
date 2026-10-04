"""Isolated Python environments of the engines, managed with uv.

Each engine gets its own virtual environment under <home>/envs/<engine>, created on demand,
so engines with conflicting dependencies (different torch, numpy, lightning versions...) never interfere.
uv keeps a shared cache with hard links, so the same wheels (e.g. torch) are stored on disk once.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable

import httpx

from ..settings import Home, Settings
from .spec import EngineSpec

LogFunc = Callable[[str], None]


def find_uv() -> str:
    try:
        from uv import find_uv_bin  # type: ignore[import-not-found]  # noqa: PLC0415

        return find_uv_bin()
    except Exception:  # noqa: BLE001
        pass
    path = shutil.which("uv")
    if path:
        return path
    raise RuntimeError("uv is not installed. Install it with `pip install uv` or from https://docs.astral.sh/uv/")


def venv_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


class EnvManager:
    def __init__(self, home: Home, settings: Settings):
        self.home = home
        self.settings = settings
        self._locks: dict[str, asyncio.Lock] = {}

    def env_dir(self, spec: EngineSpec) -> Path:
        return self.home.envs / spec.name

    def source_dir(self, spec: EngineSpec) -> Path | None:
        if spec.source is None:
            return None
        return self.home.sources / spec.name / spec.source.commit

    def marker(self, spec: EngineSpec) -> Path:
        return self.env_dir(spec) / ".mvt-installed.json"

    def python(self, spec: EngineSpec) -> Path:
        if spec.is_system:
            return Path(sys.executable)
        return venv_python(self.env_dir(spec))

    def status(self, spec: EngineSpec) -> dict:
        if spec.is_system:
            return {"installed": True, "outdated": False}
        marker = self.marker(spec)
        if not marker.exists() or not self.python(spec).exists():
            return {"installed": False, "outdated": False}
        data = json.loads(marker.read_text(encoding="utf-8"))
        backend_changed = spec.torch and (data.get("torch_backend") or "auto") != (self.settings.torch_backend or "auto")
        return {
            "installed": True,
            # a changed torch backend (e.g. cu128 for RTX 50 GPUs, or cpu) needs a reinstall too
            "outdated": data.get("fingerprint") != spec.fingerprint() or bool(backend_changed),
            "installed_at": data.get("installed_at"),
            "torch_backend": data.get("torch_backend"),
            "size_mb": round(_dir_size(self.env_dir(spec)) / 2**20, 1),
        }

    def is_ready(self, spec: EngineSpec) -> bool:
        st = self.status(spec)
        return st["installed"] and not st["outdated"]

    async def install(self, spec: EngineSpec, log: LogFunc | None = None, force: bool = False) -> None:
        log = log or (lambda _msg: None)
        lock = self._locks.setdefault(spec.name, asyncio.Lock())
        async with lock:
            if spec.is_system:
                return
            if not force and self.is_ready(spec):
                return
            uv = find_uv()
            env_dir = self.env_dir(spec)
            if env_dir.exists():
                log(f"Removing old environment of {spec.name}")
                shutil.rmtree(env_dir, ignore_errors=True)
            if spec.source is not None:
                await self._download_source(spec, log)
            log(f"Creating Python {spec.python} environment for {spec.name}")
            await _run([uv, "venv", "--python", spec.python, str(env_dir)], log)
            cpu_only = spec.cpu_requirements is not None and _cpu_only(self.settings.torch_backend)
            requirements = list(spec.cpu_requirements if cpu_only else spec.requirements)
            if cpu_only:
                log(f"No NVIDIA GPU (or torch_backend=cpu): installing the CPU variant of {spec.name}")
            if requirements:
                cmd = [uv, "pip", "install", "--python", str(self.python(spec))]
                if spec.torch:
                    cmd += ["--torch-backend", self.settings.torch_backend or "auto"]
                req_file = env_dir / "mvt-requirements.txt"
                req_file.write_text("\n".join(requirements) + "\n", encoding="utf-8")
                cmd += ["-r", str(req_file)]
                log(f"Installing packages for {spec.name} (this can take a while the first time)")
                await _run(cmd, log)
            self.marker(spec).write_text(
                json.dumps(
                    {
                        "fingerprint": spec.fingerprint(),
                        "installed_at": time.time(),
                        "torch_backend": self.settings.torch_backend if spec.torch else None,
                    }
                ),
                encoding="utf-8",
            )
            log(f"{spec.name} is installed")

    def uninstall(self, spec: EngineSpec) -> None:
        if spec.is_system:
            return
        shutil.rmtree(self.env_dir(spec), ignore_errors=True)
        if spec.source is not None:
            shutil.rmtree(self.home.sources / spec.name, ignore_errors=True)

    async def _download_source(self, spec: EngineSpec, log: LogFunc) -> None:
        assert spec.source is not None
        target = self.source_dir(spec)
        assert target is not None
        if target.exists() and any(target.iterdir()):
            return
        url = f"https://codeload.github.com/{spec.source.repo}/zip/{spec.source.commit}"
        log(f"Downloading {spec.source.repo}@{spec.source.commit[:8]}")
        async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.content
        tmp = target.with_name(target.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            archive.extractall(tmp)
        # GitHub archives contain a single top-level folder
        children = [p for p in tmp.iterdir()]
        root = children[0] if len(children) == 1 and children[0].is_dir() else tmp
        shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(target))
        shutil.rmtree(tmp, ignore_errors=True)


def _cpu_only(torch_backend: str | None) -> bool:
    backend = (torch_backend or "auto").lower()
    if backend == "cpu":
        return True
    if backend != "auto":
        return False  # an explicit CUDA / ROCm backend
    return not has_nvidia_gpu()


def has_nvidia_gpu() -> bool:
    if os.environ.get("MVT_FORCE_GPU"):
        return True
    if sys.platform == "darwin":
        return False
    smi = shutil.which("nvidia-smi")
    if smi is None and os.name == "nt":
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(candidate) if candidate.exists() else None
    return smi is not None


async def _run(cmd: list[str], log: LogFunc) -> None:
    env = dict(os.environ)
    env.setdefault("UV_LINK_MODE", "hardlink" if os.name != "nt" else "copy")
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
    )
    assert process.stdout is not None
    tail: list[str] = []
    async for raw in process.stdout:
        line = raw.decode(errors="replace").rstrip()
        if line:
            tail = (tail + [line])[-30:]
            log(line)
    code = await process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}\n" + "\n".join(tail))


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total

"""A training run: what was asked, where it lives, how far it is.

A run is not an ordinary job: jobs are coroutines inside the server, a run is a child process (the trainer) that
must outlive the server and the window that started it. Everything about a run is in its folder:

    <run dir>/
        run.json        RunInfo: the spec, the state, the numbers (written by the manager)
        events.jsonl    one JSON object per line, from the manager (queued, prepare, launched, end) and from the
                        trainer process (start, step, validation, checkpoint, samples, warning, end)
        train.log       everything the trainer printed
        prepare.log     everything the preparation (binarizing) printed
        config.yaml     the trainer's config (written by the engine)
        STOP            asked to stop: the trainer saves a checkpoint and ends (removed when it ends)
        samples/<step>/ audio of the validation files at a checkpoint
        ...             checkpoints and whatever else the engine keeps

The manager keeps an index of the run folders it manages in <home>/training/runs.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    queued = "queued"
    preparing = "preparing"  # installing the engine, downloading assets, binarizing
    training = "training"
    stopping = "stopping"
    stopped = "stopped"  # by the user, or interrupted (the toolkit or the PC restarted); can be resumed
    done = "done"
    failed = "failed"  # can be resumed from the last checkpoint when there is one


ACTIVE = {RunStatus.queued, RunStatus.preparing, RunStatus.training, RunStatus.stopping}
FINISHED = {RunStatus.stopped, RunStatus.done, RunStatus.failed}


class RunSpec(BaseModel):
    """What to train. `config` is read by the engine (datasets, dictionaries, presets, overrides...)."""

    kind: str = Field(..., description="diffsinger_acoustic, diffsinger_variance, ...")
    engine: str = Field("", description="Engine that trains it (default: the part of kind before the underscore)")
    name: str = ""
    dir: str = Field("", description="Run folder (default: <home>/training/<id>)")
    device: str = Field("auto", pattern=r"^(auto|cpu|cuda(:[0-9]+)?|mps)$", description="auto, cuda, cuda:1, cpu")
    config: dict[str, Any] = Field(default_factory=dict)

    def engine_name(self) -> str:
        return self.engine or self.kind.split("_", 1)[0]


class Checkpoint(BaseModel):
    step: int
    path: str
    size: int = 0
    time: float = 0.0
    starred: bool = False


class RunInfo(BaseModel):
    id: str
    spec: RunSpec
    status: RunStatus = RunStatus.queued
    dir: str = ""
    created_at: float = Field(default_factory=time.time)
    queue_order: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    # the trainer process
    pid: int | None = None
    # preparation (engine install, downloads, binarizing)
    stage: str = ""
    progress: float | None = None
    message: str = ""
    # training numbers (from the trainer's events)
    step: int = 0
    max_steps: int | None = None
    epoch: int | None = None
    speed: float | None = None  # steps per second
    eta: float | None = None  # seconds left
    losses: dict[str, float] = Field(default_factory=dict)
    lr: float | None = None
    gpu_mb: float | None = None
    val_loss: float | None = None
    val_step: int | None = None
    best_val_loss: float | None = None
    best_val_step: int | None = None
    # A bounded history for conservative trend hints; checkpoint decisions still belong to the user.
    validation_history: list[dict[str, float]] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    # step -> audio files of that checkpoint's validation (relative to the run folder)
    samples: dict[str, list[str]] = Field(default_factory=dict)
    renders: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    error_kind: str | None = None  # oom, crash, hang...
    # how many lines events.jsonl has (clients ask for events since a line number)
    events: int = 0
    engine_version: str = ""
    config_written: dict[str, Any] = Field(default_factory=dict)

    @property
    def path(self) -> Path:
        return Path(self.dir)


class Run:
    """A run in memory: its info, its folder and whoever listens to its events."""

    def __init__(self, info: RunInfo):
        self.info = info
        self.dir = Path(info.dir)
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self.process: Any = None  # subprocess.Popen when launched by this server
        self.task: asyncio.Task[None] | None = None
        self._events_offset = 0  # bytes of events.jsonl already read
        self._pending_line = b""
        self._dirty = False
        self._saved_at = 0.0
        # (time, step) of recent step events, for the speed
        self.step_times: list[tuple[float, int]] = []
        self.last_event_time = time.time()

    @property
    def id(self) -> str:
        return self.info.id

    @property
    def events_file(self) -> Path:
        return self.dir / "events.jsonl"

    @property
    def stop_file(self) -> Path:
        return self.dir / "STOP"

    # ---------- persistence ----------

    def save(self, force: bool = False) -> None:
        """Writes run.json (at most once a second unless forced)."""
        now = time.time()
        if not force and now - self._saved_at < 1.0:
            self._dirty = True
            return
        self._dirty = False
        self._saved_at = now
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "run.json.tmp"
        tmp.write_text(self.info.model_dump_json(indent=1), encoding="utf-8")
        os.replace(tmp, self.dir / "run.json")

    @classmethod
    def load(cls, run_dir: Path) -> "Run":
        info = RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
        info.dir = str(run_dir)
        return cls(info)

    # ---------- events ----------

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def emit(self, event: str, **data: Any) -> dict[str, Any]:
        """An event from the manager: appended to events.jsonl and sent to the subscribers."""
        payload = {"event": event, "time": time.time(), **data}
        self.dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False)
        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        # the manager's own lines are read back by read_new_events() like the trainer's, so they are counted once
        return payload

    def read_new_events(self) -> list[dict[str, Any]]:
        """Lines of events.jsonl not read before (the trainer and the manager both append to it)."""
        path = self.events_file
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        with path.open("rb") as f:
            f.seek(self._events_offset)
            chunk = f.read()
        if not chunk:
            return []
        self._events_offset += len(chunk)
        data = self._pending_line + chunk
        lines = data.split(b"\n")
        self._pending_line = lines.pop()  # an incomplete last line waits for the rest
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(json.loads(raw.decode("utf-8", errors="replace")))
            except ValueError:
                continue
        return events

    def read_events_since(self, since: int, limit: int = 5000) -> list[dict[str, Any]]:
        """Events number `since` (0-based line index) onwards, from the file."""
        path = self.events_file
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for index, line in enumerate(f):
                if index < since:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
                if len(out) >= limit:
                    break
        return out

    def publish(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(event)

    def log_tail(self, lines: int = 100, name: str = "train.log") -> list[str]:
        path = self.dir / name
        if not path.exists():
            return []
        try:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 256 * 1024))
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        # progress bars rewrite their line with \r: keep the last state of each
        cleaned = [part.split("\r")[-1].rstrip() for part in text.splitlines()]
        return [line for line in cleaned if line][-lines:]

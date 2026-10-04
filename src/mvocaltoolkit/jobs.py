"""Asynchronous jobs with progress events, cancellation and pausing.

Every long operation (installing an engine, downloading a model, transcribing, aligning...) is a job.
Clients poll GET /jobs/{id} or subscribe to WS /jobs/{id}/events.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    paused = "paused"  # waiting for POST /jobs/{id}/resume
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


FINISHED = {JobStatus.done, JobStatus.failed, JobStatus.cancelled}


class JobCancelled(Exception):
    pass


class ItemError(BaseModel):
    item: str
    error: str


class JobInfo(BaseModel):
    id: str
    kind: str
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    # data shown while paused (e.g. transcriptions to review)
    pause_data: Any = None
    error: str | None = None
    item_errors: list[ItemError] = Field(default_factory=list)


class Job:
    def __init__(self, info: JobInfo, manager: "JobManager"):
        self.info = info
        self._manager = manager
        self._cancel = asyncio.Event()
        self._resume: asyncio.Future[Any] | None = None
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self.task: asyncio.Task[Any] | None = None

    @property
    def id(self) -> str:
        return self.info.id

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()

    def emit(self, event: str, **data: Any) -> None:
        payload = {"event": event, "job": self.info.id, "time": time.time(), **data}
        for queue in list(self._subscribers):
            queue.put_nowait(payload)

    def progress(self, value: float | None = None, stage: str | None = None, message: str | None = None, **extra: Any):
        if value is not None:
            self.info.progress = max(0.0, min(1.0, value))
        if stage is not None:
            self.info.stage = stage
        if message is not None:
            self.info.message = message
        self.emit("progress", progress=self.info.progress, stage=self.info.stage, message=self.info.message, **extra)

    def item_error(self, item: str, error: str) -> None:
        self.info.item_errors.append(ItemError(item=item, error=error))
        self.emit("item_error", item=item, error=error)

    def log(self, message: str) -> None:
        self.emit("log", message=message)

    async def pause(self, data: Any) -> Any:
        """Pauses the job until it's resumed (or cancelled). Returns the data passed to resume()."""
        loop = asyncio.get_running_loop()
        self._resume = loop.create_future()
        self.info.status = JobStatus.paused
        self.info.pause_data = data
        self.emit("paused", data=data)
        cancel_wait = asyncio.ensure_future(self._cancel.wait())
        try:
            done, _ = await asyncio.wait({self._resume, cancel_wait}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            cancel_wait.cancel()
        if self._cancel.is_set():
            raise JobCancelled()
        self.info.status = JobStatus.running
        self.info.pause_data = None
        self.emit("resumed")
        return self._resume.result()

    def resume(self, data: Any) -> None:
        if self.info.status != JobStatus.paused or self._resume is None or self._resume.done():
            raise ValueError("Job is not paused")
        self._resume.set_result(data)

    def cancel(self) -> None:
        self._cancel.set()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)


JobFunc = Callable[[Job], Awaitable[Any]]


class JobManager:
    def __init__(self, history_dir: Path | None = None, max_parallel: int = 2, keep: int = 200):
        self.jobs: dict[str, Job] = {}
        self.history_dir = history_dir
        self.keep = keep
        self._semaphore = asyncio.Semaphore(max_parallel)

    def submit(self, kind: str, func: JobFunc, params: dict[str, Any] | None = None) -> Job:
        info = JobInfo(id=uuid.uuid4().hex[:12], kind=kind, params=params or {})
        job = Job(info, self)
        self.jobs[info.id] = job
        job.task = asyncio.create_task(self._run(job, func))
        self._trim()
        return job

    async def _run(self, job: Job, func: JobFunc) -> None:
        async with self._semaphore:
            if job.cancelled:
                self._finish(job, JobStatus.cancelled)
                return
            job.info.status = JobStatus.running
            job.info.started_at = time.time()
            job.emit("started")
            try:
                job.info.result = await func(job)
                job.info.progress = 1.0
                self._finish(job, JobStatus.done)
            except (JobCancelled, asyncio.CancelledError):
                self._finish(job, JobStatus.cancelled)
            except Exception as e:  # noqa: BLE001
                job.info.error = f"{type(e).__name__}: {e}"
                job.info.message = traceback.format_exc(limit=8)
                self._finish(job, JobStatus.failed)

    def _finish(self, job: Job, status: JobStatus) -> None:
        job.info.status = status
        job.info.finished_at = time.time()
        job.emit("finished", status=status.value, error=job.info.error)
        if self.history_dir is not None:
            try:
                path = self.history_dir / f"{job.id}.json"
                path.write_text(job.info.model_dump_json(indent=1), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass

    def get(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job is None and self.history_dir is not None:
            path = self.history_dir / f"{job_id}.json"
            if path.exists():
                info = JobInfo.model_validate(json.loads(path.read_text(encoding="utf-8")))
                job = Job(info, self)
        return job

    def _trim(self) -> None:
        finished = [j for j in self.jobs.values() if j.info.status in FINISHED]
        excess = len(self.jobs) - self.keep
        for job in sorted(finished, key=lambda j: j.info.created_at)[: max(0, excess)]:
            self.jobs.pop(job.id, None)

    async def wait(self, job: Job) -> JobInfo:
        if job.task is not None:
            await asyncio.shield(job.task)
        return job.info

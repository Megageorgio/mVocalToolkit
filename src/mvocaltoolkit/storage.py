"""Disk space used by the toolkit home, and removal of files nobody needs any more.

Uploads, results written to <home>/outputs and the job history pile up with every job; they are removed after
`Settings.keep_files_days` (when the server starts, and on request). Models and engine environments stay until they
are removed by hand: they are listed here with their sizes so a program can offer that.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from .settings import Home

# parts of the home that hold only leftovers of jobs and downloads (the rest of cache/ holds models the engines
# downloaded themselves, Whisper's among them: removing it means downloading them again)
TEMPORARY = ("uploads", "outputs", "jobs", "cache/work")


def size_of(path: Path) -> int:
    """Bytes of a file or of everything in a folder (links are not followed)."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _children(folder: Path) -> list[Path]:
    return sorted(folder.iterdir()) if folder.is_dir() else []


def usage(home: Home) -> dict[str, Any]:
    """Sizes of the models, engine environments and temporary parts of the home, in bytes."""
    models = [{"id": p.name, "bytes": size_of(p)} for p in _children(home.models) if p.is_dir()]
    # an engine's environment and its upstream sources go away together
    names = sorted({p.name for p in _children(home.envs) if p.is_dir()} | {p.name for p in _children(home.sources) if p.is_dir()})
    envs = [{"id": n, "bytes": size_of(home.envs / n) + size_of(home.sources / n)} for n in names]
    temporary = {name: size_of(home.root / name) for name in TEMPORARY}
    engine_downloads = max(0, size_of(home.cache) - temporary["cache/work"])
    total = sum(m["bytes"] for m in models) + sum(e["bytes"] for e in envs) + sum(temporary.values()) + engine_downloads
    return {"home": str(home.root), "total": total, "models": models, "engines": envs, "temporary": temporary,
            "engine_downloads": engine_downloads}


def cleanup(home: Home, older_than_days: float | None, parts: tuple[str, ...] = TEMPORARY, busy: bool = False) -> dict[str, Any]:
    """Removes entries of [parts] last changed more than [older_than_days] ago (None: all of them).

    While a job runs ([busy]) only entries older than a day go, whatever was asked: a running job writes there."""
    if busy:
        older_than_days = max(1.0, older_than_days or 0.0)
    limit = time.time() - older_than_days * 86400 if older_than_days is not None else None
    freed, removed = 0, 0
    for part in parts:
        if part not in TEMPORARY:
            continue
        for entry in _children(home.root / part):
            try:
                changed = _newest_change(entry)
                if limit is not None and changed > limit:
                    continue
                size = size_of(entry)
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                freed += size
                removed += 1
            except OSError:
                pass  # in use (Windows) or gone already: next time
    return {"freed": freed, "removed": removed}


def _newest_change(path: Path) -> float:
    """Last change of a file, or of the newest file in a folder (a folder's own time misses changes deep inside)."""
    newest = path.stat().st_mtime
    if path.is_dir():
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    newest = max(newest, os.lstat(os.path.join(root, name)).st_mtime)
                except OSError:
                    pass
    return newest

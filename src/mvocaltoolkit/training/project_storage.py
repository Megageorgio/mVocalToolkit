"""Bounded, read-only project inventory. File contents are never loaded."""
from collections import defaultdict
import heapq
import os
from pathlib import Path
import shutil
import stat
from time import perf_counter

from pydantic import BaseModel, Field


class StorageRequest(BaseModel):
    model_config = {"extra": "forbid"}
    project_dir: str = Field(min_length=1)
    max_entries: int = Field(100_000, ge=1, le=250_000, strict=True)
    max_seconds: int = Field(10, ge=1, le=30, strict=True)
    top_limit: int = Field(20, ge=1, le=100, strict=True)


def _category(parts, bank):
    lower = tuple(p.casefold() for p in parts)
    if bank is not None: return "banks"
    if any(p.startswith("binary") or p == "binaries" for p in lower[:-1]): return "binaries"
    if lower[0] == "exports": return "exports"
    if any(p in {"cache", ".cache"} for p in lower[:-1]): return "cache"
    if lower[0] == "data": return "data"
    if Path(lower[-1]).suffix in {".ckpt", ".pth", ".pt", ".safetensors"} and ("runs" in lower or "checkpoints" in lower): return "checkpoints"
    if Path(lower[-1]).suffix == ".log" or lower[-1] == "events.jsonl" or "logs" in lower[:-1]: return "logs"
    return "models" if lower[0] == "models" else "other"


def inventory(req: StorageRequest) -> dict:
    if not req.project_dir.strip(): raise ValueError("Select a project folder")
    root = Path(req.project_dir).expanduser().resolve()
    if not root.is_dir(): raise ValueError("Project folder does not exist")
    started = perf_counter()
    def counter(): return defaultdict(lambda: {"bytes": 0, "files": 0})
    categories, directories, components = counter(), counter(), counter()
    largest, identities, examples = [], set(), []
    files = total = unique = entries = folders = links = inaccessible = identity_unavailable = 0
    partial_reason = ""
    pending = [(root, None)]
    while pending and not partial_reason:
        folder, bank = pending.pop()
        if perf_counter() - started >= req.max_seconds:
            partial_reason = "time_limit"; break
        try:
            if bank is None and (folder / "character.txt").is_file() and (folder / "dsconfig.yaml").is_file():
                bank = folder.relative_to(root).as_posix()
            with os.scandir(folder) as listing:
                folders += 1
                for entry in listing:
                    if entries >= req.max_entries or perf_counter() - started >= req.max_seconds:
                        partial_reason = "entry_limit" if entries >= req.max_entries else "time_limit"; break
                    entries += 1
                    try:
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                            links += 1; continue
                        path = Path(entry.path)
                        if stat.S_ISDIR(info.st_mode):
                            pending.append((path, bank)); continue
                        if not stat.S_ISREG(info.st_mode): continue
                        # Windows DirEntry.stat does not populate the file ID; a full stat does.
                        if not info.st_ino:
                            info = path.stat(follow_symlinks=False)
                            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                                links += 1; continue
                            if not stat.S_ISREG(info.st_mode): continue
                        relative, size = path.relative_to(root), info.st_size
                        files += 1; total += size
                        identity = (info.st_dev, info.st_ino)
                        if not info.st_ino:
                            identity_unavailable += 1; unique += size
                        elif identity not in identities:
                            identities.add(identity); unique += size
                        category = _category(relative.parts, bank)
                        group = relative.parts[0] if len(relative.parts) > 1 else "."
                        targets = [(categories, category), (directories, group)]
                        if bank is not None:
                            sub = path.relative_to(root / bank).parts
                            targets.append((components, (bank, sub[0] if len(sub) > 1 else "metadata")))
                        for counts, key in targets:
                            counts[key]["bytes"] += size; counts[key]["files"] += 1
                        candidate = (size, relative.as_posix(), category)
                        if len(largest) < req.top_limit: heapq.heappush(largest, candidate)
                        elif candidate > largest[0]: heapq.heapreplace(largest, candidate)
                    except OSError as exc:
                        inaccessible += 1
                        if len(examples) < 20: examples.append(dict(path=str(entry.path), error=str(exc)))
        except OSError as exc:
            inaccessible += 1
            if len(examples) < 20: examples.append(dict(path=str(folder), error=str(exc)))
    try:
        disk = shutil.disk_usage(root)
        volume = dict(total_bytes=disk.total, free_bytes=disk.free)
    except OSError: volume = None
    def ranked(values): return sorted(values.items(), key=lambda row: (-row[1]["bytes"], row[0]))
    return dict(project_dir=str(root), bytes=total, unique_file_bytes=unique, files=files,
        directories_scanned=folders, entries_scanned=entries, elapsed_seconds=perf_counter() - started,
        partial=bool(partial_reason or inaccessible), partial_reason=partial_reason or ("unreadable_entries" if inaccessible else ""), skipped_links=links, inaccessible=inaccessible,
        identity_unavailable=identity_unavailable, errors=examples, volume=volume,
        categories=[dict(category=key, **value) for key, value in ranked(categories)],
        directories=[dict(path=key, **value) for key, value in ranked(directories)[:req.top_limit]],
        bank_components=[dict(bank=key[0], component=key[1], **value) for key, value in ranked(components)[:req.top_limit]],
        largest_files=[dict(path=path, bytes=size, category=category) for size, path, category in sorted(largest, key=lambda row: (-row[0], row[1]))])

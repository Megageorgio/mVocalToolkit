"""Checkpoint retention shared by the server and detached trainer; standard library only."""
from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

PATTERN = re.compile(r"model_ckpt_steps_(\d+)\.ckpt$")


def retained_path(run_dir: Path, step: int) -> Path:
    if step < 0:
        raise ValueError("Checkpoint step must be non-negative")
    return run_dir / "retained" / f"model_ckpt_steps_{step}.ckpt"


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def retain(run_dir: Path, step: int, keep: bool) -> None:
    backup = retained_path(run_dir, step)
    if keep:
        if backup.is_file():
            return
        source = run_dir / backup.name
        if not source.is_file():
            source = run_dir / "best" / backup.name
        if not source.is_file():
            raise ValueError("This checkpoint is no longer available; refresh the run")
        _link_or_copy(source, backup)
    else:
        # Removing a star never removes the trainer's checkpoint.
        if backup.is_file() and not (run_dir / backup.name).exists():
            _link_or_copy(backup, run_dir / backup.name)
        backup.unlink(missing_ok=True)


def restore(run_dir: Path, step: int | None) -> None:
    """Upstream infer/export resolves checkpoints in the run root."""
    if step is None:
        return
    backup = retained_path(run_dir, step)
    if not backup.is_file():
        backup = run_dir / "best" / backup.name
    original = run_dir / backup.name
    if not original.is_file() and backup.is_file():
        _link_or_copy(backup, original)


def listing(run_dir: Path) -> list[dict]:
    found = {}
    for folder in (run_dir / "best", run_dir / "retained", run_dir):
        for path in folder.glob("model_ckpt_steps_*.ckpt"):
            match = PATTERN.fullmatch(path.name)
            if match is None:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue  # the trainer may just have rotated an unretained checkpoint
            step = int(match[1])
            found[step] = {"step": step, "path": str(path), "size": stat.st_size, "time": stat.st_mtime,
                           "starred": retained_path(run_dir, step).is_file()}
    return [found[step] for step in sorted(found)]


def retain_best(run_dir: Path, step: int) -> None:
    original = run_dir / retained_path(run_dir, step).name
    if not original.is_file():
        return
    target = run_dir / "best" / original.name
    if not target.is_file():
        _link_or_copy(original, target)
    for previous in target.parent.glob("model_ckpt_steps_*.ckpt"):
        if previous != target:
            previous.unlink(missing_ok=True)


def protect_removal(checkpoint_class, run_dir: Path) -> None:
    """Install only inside our detached trainer; upstream files are untouched."""
    original = checkpoint_class._remove_checkpoint

    def remove(self, trainer, filepath):
        match = PATTERN.fullmatch(Path(filepath).name)
        if match:
            saved = retained_path(run_dir, int(match[1]))
            if saved.is_file() or (run_dir / "best" / saved.name).is_file():
                return
        return original(self, trainer, filepath)

    checkpoint_class._remove_checkpoint = remove

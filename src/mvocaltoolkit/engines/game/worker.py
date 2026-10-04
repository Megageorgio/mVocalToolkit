"""GAME engine worker: runs GAME's own `infer.py extract` command in-process."""

from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mvt_engine as rt

OPTION_FLAGS = {
    "seg_threshold": "--seg-threshold",
    "seg_radius": "--seg-radius",
    "t0": "--t0",
    "nsteps": "--nsteps",
    "est_threshold": "--est-threshold",
    "batch_size": "--batch-size",
}


@rt.method()
def extract(
    model: dict[str, Any],
    audio: str,
    output_dir: str,
    language: str | None = None,
    tempo: float = 120.0,
    options: dict[str, Any] | None = None,
    output_formats: list[str] | None = None,
    pitch_format: str = "name",
    round_pitch: bool = False,
) -> dict[str, Any]:
    import infer  # noqa: PLC0415  (GAME)

    wanted = [f for f in (output_formats or ["mid", "csv"]) if f in ("mid", "txt", "csv")]
    root = Path(model["path"])
    checkpoint = root / model["layout"]["model"]
    source = Path(audio)
    try:
        with tempfile.TemporaryDirectory(prefix="mvt-game-") as tmp:
            work = Path(tmp)
            formats = sorted(set(wanted) | {"csv"})  # csv is always produced to read the notes back
            args = [
                "extract", str(source), "-m", str(checkpoint),
                "--output-formats", ",".join(formats),
                "--output-dir", str(work),
                "--tempo", str(tempo),
                "--pitch-format", pitch_format,
                "--num-workers", "0",
            ]
            if language:
                args += ["-l", language]
            if round_pitch:
                args.append("--round-pitch")
            for key, value in (options or {}).items():
                if key in OPTION_FLAGS and value is not None:
                    args += [OPTION_FLAGS[key], str(value)]
            rt.progress(0.1, f"Extracting notes from {source.name}")
            infer.main.main(args=args, standalone_mode=False)
            notes = _read_csv(work / f"{source.stem}.csv")
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            files = {}
            for fmt in wanted:
                produced = work / f"{source.stem}.{fmt}"
                if produced.exists():
                    target = out / produced.name
                    shutil.copyfile(produced, target)
                    files[fmt] = str(target)
        rt.progress(1.0, "Done")
        return {"ok": True, "notes": notes, "files": files}
    except Exception as e:  # noqa: BLE001
        if rt.is_oom(e):
            rt.free_memory()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    notes = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pitch = row.get("pitch", "")
            notes.append(
                {
                    "start": float(row["onset"]),
                    "end": float(row["offset"]),
                    "note": pitch,
                    "pitch": _to_float(pitch),
                }
            )
    return notes


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


if __name__ == "__main__":
    rt.run()

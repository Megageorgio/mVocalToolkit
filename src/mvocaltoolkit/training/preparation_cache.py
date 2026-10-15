"""Verified per-recording cache; copies never share writable inodes with sources."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_stamp(tk, identifiers, engines=()):
    stamp = {}
    for identifier in identifiers:
        if not identifier:
            continue
        installed = getattr(tk.models, "get_installed", lambda _id: None)(identifier)
        folder = Path(installed.path) if installed else Path(identifier.removeprefix("path:")).expanduser()
        if installed:
            stamp[identifier] = installed.model_dump()
        if folder.exists():
            paths = folder.rglob("*") if folder.is_dir() else [folder]
            stamp.setdefault(identifier, {})["files"] = [(str(p.relative_to(folder)) if folder.is_dir() else p.name, p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(paths) if p.is_file()]
    if hasattr(tk, "engines"):
        stamp["engines"] = {name: spec.fingerprint() for name, spec in tk.engines.specs.items() if name in engines}
    return stamp


def cache_key(audio, label, req, relative, model_stamp=None):
    settings = req.model_dump(exclude={"folder", "out_dir", "cache"})
    payload = dict(version=2, audio=digest(audio), name=str(relative), settings=settings,
                   label=label.model_dump() if label else None, model=model_stamp)
    payload["implementation"] = [digest(Path(__file__).with_name(name)) for name in
        ("preparation.py", "phonemizers.py", "phonemizer_profiles.json", "wfl_multi13_map.json")]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def restore(root: Path, key: str, out: Path, source_root: Path | None = None) -> dict | None:
    folder = root / key
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "mvt-preparation-cache-2":
            return None
        for relative, checksum in manifest["hashes"].items():
            source = (folder / relative).resolve()
            target = (out / relative).resolve()
            if not source.is_relative_to(folder.resolve()) or not target.is_relative_to(out.resolve()) or not source.is_file() or digest(source) != checksum:
                return None
        for relative in manifest["hashes"]:
            target = out / relative
            if target.exists():
                return None
        for relative in manifest["hashes"]:
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / relative, target)
        def relocate(value):
            old = manifest.get("out_dir", "")
            if isinstance(value, str) and old and value.startswith(old):
                return str(out) + value[len(old):]
            old_source = manifest.get("source_root", "")
            if isinstance(value, str) and source_root is not None and old_source and value.startswith(old_source):
                return str(source_root) + value[len(old_source):]
            if isinstance(value, dict):
                return {k: relocate(v) for k, v in value.items()}
            if isinstance(value, list):
                return [relocate(v) for v in value]
            return value
        return relocate(manifest["result"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def store(root: Path, key: str, out: Path, paths: list[Path], result: dict, source_root: Path | None = None):
    root.mkdir(parents=True, exist_ok=True)
    folder = root / key
    # Existing immutable entries are retained; corrupt ones get a fresh key only
    # after explicit cache cleanup. Refresh still recomputes the prepared output.
    if folder.exists():
        return
    staging = root / (".tmp-" + uuid.uuid4().hex)
    staging.mkdir()
    hashes = {}
    try:
        for path in paths:
            relative = path.relative_to(out).as_posix()
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[relative] = digest(target)
        (staging / "manifest.json").write_text(json.dumps(dict(format="mvt-preparation-cache-2", source_root=str(source_root) if source_root else "", out_dir=str(out), hashes=hashes, result=result), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(staging, folder)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def inventory(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    return [dict(key=p.name, bytes=sum(f.stat().st_size for f in p.rglob("*") if f.is_file()), modified=p.stat().st_mtime)
            for p in root.iterdir() if p.is_dir() and len(p.name) == 64 and all(c in "0123456789abcdef" for c in p.name) and (p / "manifest.json").is_file()]


def remove(root: Path, keys: list[str]) -> int:
    known = {p["key"] for p in inventory(root)}
    if set(keys) - known:
        raise ValueError("Choose existing preparation cache entries")
    for key in keys:
        target = (root / key).resolve()
        if target.parent != root.resolve():
            raise ValueError("Cache path escaped its root")
        shutil.rmtree(target)
    return len(keys)

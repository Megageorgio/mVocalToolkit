"""RVC input snapshots and detached trainer/export commands."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mvt_engine as rt

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("MVT_SOURCE_DIR") or os.getcwd()).resolve()
sys.path.insert(0, str(HERE.parent / "diffsinger"))
from checkpoint_store import listing


@rt.method()
def prepare(run_dir: str, kind: str, spec: dict, assets=None, device="auto"):
    path = Path(run_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    content = Path(spec["prepared_file"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != spec["prepared_sha256"]:
        raise ValueError("The recording manifest changed")
    data = json.loads(content)
    (path / "recordings.json").write_bytes(content)
    config = data["settings"]
    assets = assets or {}
    for key, asset in (("base_generator", "generator"), ("base_discriminator", "discriminator")):
        value = assets.get(asset)
        if value:
            source = Path(value)
            if not source.is_file():
                raise ValueError(f"Missing {asset} training weights")
            expected = data.get("base_hashes", {}).get(key)
            if expected:
                with source.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError("Base weights changed after the recording check")
            target = path / f"base-{asset}.pth"
            shutil.copy2(source, target)
            config[key] = str(target)
    embedder = Path(assets["embedder"])
    if embedder.is_file():
        embedder = embedder.parent
    if not all((embedder / name).is_file() for name in ("config.json", "pytorch_model.bin")):
        raise ValueError("ContentVec needs config.json and pytorch_model.bin")
    config["embedder"] = str(embedder)
    import yaml
    (path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    rt.progress(None, "Preparing RVC audio, Harvest pitch and ContentVec features", stage="preprocess")
    tail = []
    with (path / "prepare.log").open("ab") as log:
        process = subprocess.Popen([sys.executable, "-u", str(HERE / "prepare.py"), "--run-dir", str(path), "--device", device],
                                   cwd=SOURCE, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            log.write(line); log.flush()
            decoded = line.decode("utf-8", errors="replace").strip()
            tail = (tail + [decoded])[-15:]
            if decoded.startswith("MVT_PROGRESS "):
                progress = json.loads(decoded.removeprefix("MVT_PROGRESS "))
                rt.progress(progress["fraction"], progress["message"], stage="preprocess")
        if process.wait():
            raise RuntimeError("RVC preparation failed:\n" + "\n".join(tail))
    summary = json.loads((path / "prepared.json").read_text(encoding="utf-8"))
    return {"config": {**config, **summary}, "max_steps": config["max_steps"] or None,
            "launch": {"args": [str(HERE / "train.py"), "--run-dir", str(path), "--device", device]}}


@rt.method()
def resume_command(run_dir: str, kind: str, spec: dict):
    return {"launch": {"args": [str(HERE / "train.py"), "--run-dir", run_dir, "--resume"]}}


@rt.method()
def checkpoints(run_dir: str):
    return listing(Path(run_dir))


@rt.method()
def export(run_dir: str, kind: str, out_dir: str, step=None, **options):
    from exporter import export_model
    return export_model(Path(run_dir), Path(out_dir), step)


if __name__ == "__main__":
    rt.run()

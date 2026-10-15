"""Engine protocol for optional model trainers; expensive work runs in separate processes."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mvt_engine as rt
from mvt_train_common import listing

MAIN = Path(__file__).with_name("mvt_train_main.py")


def execute(run, args, log_name):
    tail = []
    with (Path(run) / log_name).open("ab") as log:
        process = subprocess.Popen([sys.executable, "-u", str(MAIN), *map(str, args)], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for raw in process.stdout:
            log.write(raw); log.flush()
            line = raw.decode("utf-8", errors="replace").strip()
            tail = (tail + [line])[-20:]
            if line.startswith("MVT_PROGRESS "):
                data = json.loads(line[13:]); rt.progress(data.get("progress"), data.get("message"), stage=data.get("stage", "prepare"))
        if process.wait(): raise RuntimeError("Trainer operation failed:\n" + "\n".join(tail))


@rt.method()
def prepare(run_dir, kind, spec, assets=None, device="auto"):
    run = Path(run_dir).resolve(); run.mkdir(parents=True, exist_ok=True)
    content = Path(spec["prepared_file"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != spec["prepared_sha256"]:
        raise ValueError("The training manifest changed")
    (run / "input-manifest.json").write_bytes(content)
    execute(run, ["prepare", "--run", run, "--kind", kind, "--device", device], "prepare.log")
    import yaml
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    return {"config": config, "max_steps": config["steps"],
            "launch": {"args": [str(MAIN), "train", "--run", str(run), "--kind", kind, "--device", device]}}


@rt.method()
def resume_command(run_dir, kind, spec):
    return {"launch": {"args": [str(MAIN), "train", "--run", run_dir, "--kind", kind, "--resume"]}}


@rt.method()
def checkpoints(run_dir):
    return listing(Path(run_dir))


@rt.method()
def export(run_dir, kind, out_dir, step=None, **options):
    args = ["export", "--run", run_dir, "--kind", kind, "--out", out_dir]
    if step is not None: args += ["--step", step]
    execute(run_dir, args, "export.log")
    return {"out_dir": str(Path(out_dir).resolve()), "files": [str(p) for p in Path(out_dir).rglob("*") if p.is_file()]}


@rt.method()
def environment():
    import torch
    return {"torch": torch.__version__, "cuda": torch.cuda.is_available()}


@rt.method()
def render_speech(**params):
    from mvt_speech_preview import render
    return render(**params)

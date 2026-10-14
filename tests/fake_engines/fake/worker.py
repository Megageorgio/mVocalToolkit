"""Fake training engine for tests: the trainer is fake_train.py next to this file (a process of its own)."""

import json
import os
from pathlib import Path

import mvt_engine as rt

ENGINE_DIR = Path(os.environ.get("MVT_ENGINE_DIR") or Path(__file__).parent)


@rt.method()
def environment():
    return {"torch": "0.0", "cuda": False, "gpus": []}


@rt.method()
def prepare(run_dir, kind, spec, assets=None, device="auto"):
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        rt.progress(i / 3, f"binarizing {i}", stage="binarize")
    config = {"max_updates": int(spec.get("max_steps", 30)), "val_check_interval": int(spec.get("val_every", 10)),
              "step_seconds": float(spec.get("step_seconds", 0.02)), "fail_at": spec.get("fail_at"), "assets": assets or {}}
    (run / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (run / "prepare.log").write_text("binarized\n", encoding="utf-8")
    return {"launch": {"args": [str(ENGINE_DIR / "fake_train.py"), "--run-dir", str(run), "--reset"]},
            "config": config, "max_steps": config["max_updates"]}


@rt.method()
def resume_command(run_dir, kind, spec=None):
    return {"launch": {"args": [str(ENGINE_DIR / "fake_train.py"), "--run-dir", str(run_dir)]}}


@rt.method()
def checkpoints(run_dir):
    out = []
    for p in sorted(Path(run_dir).glob("model_ckpt_steps_*.ckpt")):
        out.append({"step": int(p.stem.rsplit("_", 1)[1]), "path": str(p), "size": p.stat().st_size, "time": p.stat().st_mtime})
    return out


@rt.method()
def export(run_dir, kind, out_dir, step=None, **_kwargs):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.onnx").write_text(f"exported step {step}", encoding="utf-8")
    return {"out_dir": str(out), "files": [str(out / "model.onnx")]}


if __name__ == "__main__":
    rt.run()

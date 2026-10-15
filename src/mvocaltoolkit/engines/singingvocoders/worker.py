"""Pinned SingingVocoders worker; heavy libraries stay in subprocesses."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import mvt_engine as rt

SOURCE = Path(os.environ.get("MVT_SOURCE_DIR") or os.getcwd()).resolve()
ENGINE = Path(__file__).parent.resolve()
sys.path.insert(0, str(ENGINE.parent / "diffsinger"))


def run_script(args, log_path):
    tail = []
    with log_path.open("ab") as log:
        process = subprocess.Popen([sys.executable, "-u", *map(str, args)], cwd=SOURCE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        for raw in process.stdout:
            log.write(raw); log.flush()
            line = raw.decode("utf-8", errors="replace").strip()
            tail = (tail + [line])[-25:]
            if line.startswith("MVT_PROGRESS "):
                data = json.loads(line.removeprefix("MVT_PROGRESS "))
                rt.progress(data.get("fraction"), data["message"], stage="preprocess")
        if process.wait() != 0:
            raise RuntimeError("SingingVocoders failed:\n" + "\n".join(tail))


def load_config(path):
    import yaml
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    bases = config.pop("base_config", [])
    if isinstance(bases, str):
        bases = [bases]
    result = {}
    for base in bases:
        deep_update(result, load_config(SOURCE / base))
    deep_update(result, config)
    return result


def deep_update(base, values):
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


@rt.method()
def environment():
    import shutil
    import torch

    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "source": str(SOURCE), "gpus": [], "disk_free_mb": shutil.disk_usage(SOURCE).free // 2**20}
    if info["cuda"]:
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            info["gpus"].append({"index": index, "name": torch.cuda.get_device_name(index),
                                 "memory_total_mb": total // 2**20, "memory_free_mb": free // 2**20})
    return info


@rt.method()
def prepare(run_dir: str, kind: str, spec: dict, assets: dict | None = None, device: str = "auto"):
    import yaml
    folder = Path(run_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    report = spec["data_report"]
    config = load_config(SOURCE / "configs/ft_hifigan.yaml")
    deep_update(config, spec["overlay"])
    config.update({"finetune_ckpt_path": (assets or {}).get("base"), "finetune_enabled": True,
                   "DataIndexPath": str(folder / "binary"), "work_dir": str(folder),
                   "train_set_name": "train", "valid_set_name": "valid"})
    if not config["finetune_ckpt_path"]:
        raise ValueError("The full SingingVocoders base checkpoint is required")
    (folder / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (folder / "source-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    rt.progress(0, "Preparing clean audio, pitch and mel spectra", stage="preprocess")
    run_script([ENGINE / "mvt_sv_prepare.py", "--run-dir", folder], folder / "prepare.log")
    return {"config": config, "max_steps": config["max_updates"] // 2,
            "launch": {"args": [str(ENGINE / "mvt_sv_train.py"), "--run-dir", str(folder)]}}


@rt.method()
def resume_command(run_dir: str, kind: str, spec: dict | None = None):
    config = load_config(Path(run_dir) / "config.yaml")
    return {"max_steps": config["max_updates"] // 2,
            "launch": {"args": [str(ENGINE / "mvt_sv_train.py"), "--run-dir", run_dir]}}


@rt.method()
def checkpoints(run_dir: str):
    from checkpoint_store import listing

    return [{**c, "step": c["step"] // 2} for c in listing(Path(run_dir))]


@rt.method()
def export(run_dir: str, kind: str, out_dir: str, step: int | None = None, **kwargs):
    saved = checkpoints(run_dir)
    chosen = next((c for c in reversed(saved) if step is None or c["step"] == step), None)
    if chosen is None:
        raise ValueError("The selected vocoder checkpoint is no longer available")
    from checkpoint_store import restore

    restore(Path(run_dir), chosen["step"] * 2)
    checkpoint = Path(run_dir) / f"model_ckpt_steps_{chosen['step'] * 2}.ckpt"
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_script([SOURCE / "export_ckpt.py", "--ckpt_path", checkpoint, "--save_path", out / "model.ckpt"],
               Path(run_dir) / "export.log")
    (out / "LICENSE-NOTICE.txt").write_text(
        "Fine-tuned from OpenVPI PC-NSF-HiFiGAN 2025.02. CC BY-NC-SA 4.0.\n"
        "https://github.com/openvpi/vocoders/releases/tag/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02\n"
        "https://creativecommons.org/licenses/by-nc-sa/4.0/\n", encoding="utf-8")
    return {"out_dir": str(out), "files": [str(p) for p in out.iterdir() if p.is_file()], "checkpoint": str(out / "model.ckpt")}


if __name__ == "__main__":
    rt.run()

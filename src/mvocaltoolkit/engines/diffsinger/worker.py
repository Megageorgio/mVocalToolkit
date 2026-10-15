"""DiffSinger training engine worker: writes the config, binarizes, tells the manager how to start the trainer
(mvt_ds_train.py next to this file, as a process of its own), lists checkpoints, exports ONNX, renders .ds files.

Runs inside the engine's environment with the pinned openvpi/DiffSinger source as the working directory
(MVT_SOURCE_DIR). Nothing in the upstream source is changed."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import mvt_engine as rt

SOURCE = Path(os.environ.get("MVT_SOURCE_DIR") or os.getcwd()).resolve()
ENGINE_DIR = Path(os.environ.get("MVT_ENGINE_DIR") or Path(__file__).parent).resolve()
CKPT_RE = re.compile(r"model_ckpt_steps_(\d+)\.ckpt$")
TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[")


def _yaml():
    import yaml  # noqa: PLC0415

    return yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    return _yaml().safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_update(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _is_acoustic(kind: str) -> bool:
    return kind.endswith("acoustic")


def full_config(kind: str, overlay: dict[str, Any]) -> dict[str, Any]:
    """base.yaml <- acoustic.yaml | variance.yaml <- overlay, every key explicit, no base_config."""
    cfg = _load_yaml(SOURCE / "configs" / "base.yaml")
    specific = _load_yaml(SOURCE / "configs" / ("acoustic.yaml" if _is_acoustic(kind) else "variance.yaml"))
    specific.pop("base_config", None)
    _deep_update(cfg, specific)
    overlay = dict(overlay)
    overlay.pop("base_config", None)
    _deep_update(cfg, overlay)
    cfg.pop("base_config", None)
    return cfg


def _run(args: list[str], log_path: Path, on_line: Callable[[str], None] | None = None,
         env: dict[str, str] | None = None) -> None:
    """Runs a DiffSinger script in the engine's interpreter, its output into log_path (and to on_line)."""
    full_env = dict(os.environ)
    full_env.update(env or {})
    full_env["PYTHONPATH"] = str(SOURCE) + os.pathsep + full_env.get("PYTHONPATH", "")
    tail: list[str] = []
    with log_path.open("ab") as log:
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n".encode())
        process = subprocess.Popen(
            [sys.executable, "-u", *args], cwd=str(SOURCE), env=full_env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        buffer = b""
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            buffer += chunk
            # progress bars end their updates with \r, lines with \n
            while True:
                cut = -1
                for sep in (b"\n", b"\r"):
                    i = buffer.find(sep)
                    if i >= 0 and (cut < 0 or i < cut):
                        cut = i
                if cut < 0:
                    break
                line = buffer[:cut].decode("utf-8", errors="replace").strip()
                buffer = buffer[cut + 1:]
                if line:
                    tail = (tail + [line])[-40:]
                    if on_line is not None:
                        on_line(line)
        code = process.wait()
    if code != 0:
        text = "\n".join(tail)
        lowered = text.lower()
        error = RuntimeError(f"{Path(args[0]).name} failed (exit code {code}):\n{text}")
        if "out of memory" in lowered:
            raise MemoryError(f"out of memory:\n{text}") from error
        raise error


@rt.method()
def environment() -> dict[str, Any]:
    import shutil  # noqa: PLC0415

    import torch  # noqa: PLC0415

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpus": [],
        "source": str(SOURCE),
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            info["gpus"].append({"index": i, "name": torch.cuda.get_device_name(i),
                                 "memory_total_mb": round(total / 2**20), "memory_free_mb": round(free / 2**20)})
    mps = getattr(torch.backends, "mps", None)
    info["mps"] = bool(mps is not None and mps.is_available())
    home = os.environ.get("MVT_HOME") or str(Path.home())
    try:
        usage = shutil.disk_usage(home)
        info["disk_free_mb"] = round(usage.free / 2**20)
    except OSError:
        pass
    return info


@rt.method()
def write_config(run_dir: str, kind: str, overlay: dict[str, Any]) -> dict[str, Any]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    cfg = full_config(kind, overlay)
    cfg["binary_data_dir"] = cfg.get("binary_data_dir") or str(run_path / "binary")
    path = run_path / "config.yaml"
    path.write_text(_yaml().safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"path": str(path), "config": cfg}


def _assets_into(overlay: dict[str, Any], assets: dict[str, str]) -> None:
    if assets.get("vocoder"):
        overlay["vocoder_ckpt"] = assets["vocoder"]
    if assets.get("pe"):
        overlay["pe_ckpt"] = assets["pe"]
    if assets.get("hnsep"):
        overlay["hnsep_ckpt"] = assets["hnsep"]


@rt.method()
def prepare(run_dir: str, kind: str, spec: dict[str, Any], assets: dict[str, str] | None = None,
            device: str = "auto") -> dict[str, Any]:
    """Writes the config and binarizes the datasets; returns how to start the trainer."""
    run_path = Path(run_dir).resolve()
    overlay = dict(spec.get("overlay") or {})
    _assets_into(overlay, assets or {})
    written = write_config(str(run_path), kind, overlay)
    cfg = written["config"]
    rt.progress(0.0, "Binarizing the data", stage="binarize")
    state = {"done": 0, "total": 0}

    def on_line(line: str) -> None:
        m = TQDM_RE.search(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            if total >= state["total"] or done < state["done"]:
                state["done"], state["total"] = done, total
            rt.progress(done / total if total else None, f"Binarizing: {done} / {total}", stage="binarize")
        elif line.startswith("|") or "Error" in line or "error" in line:
            rt.progress(None, line[:160], stage="binarize")

    env = {"MVT_DEVICE": device}
    _run(["scripts/binarize.py", "--config", str(run_path / "config.yaml")], run_path / "prepare.log", on_line, env)
    rt.progress(1.0, "Binarized", stage="binarize")
    return {
        "launch": {"args": [str(ENGINE_DIR / "mvt_ds_train.py"), "--run-dir", str(run_path), "--reset"]},
        "config": cfg,
        "max_steps": cfg.get("max_updates"),
    }


@rt.method()
def resume_command(run_dir: str, kind: str, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    cfg = _load_yaml(run_path / "config.yaml") if (run_path / "config.yaml").exists() else {}
    return {
        "launch": {"args": [str(ENGINE_DIR / "mvt_ds_train.py"), "--run-dir", str(run_path)]},
        "max_steps": cfg.get("max_updates"),
    }


@rt.method()
def checkpoints(run_dir: str) -> list[dict[str, Any]]:
    from checkpoint_store import listing

    return listing(Path(run_dir))


@rt.method()
def export(run_dir: str, kind: str, out_dir: str, step: int | None = None, export_spk: list[str] | None = None,
           freeze_spk: str | None = None, freeze_gender: float | None = None, freeze_velocity: bool = False,
           freeze_glide: bool = False, freeze_expr: bool = False, name: str | None = None) -> dict[str, Any]:
    """ONNX export of a checkpoint (the latest by default) into out_dir with DiffSinger's own exporter; the files
    are named after `name` (default: acoustic or variance)."""
    run_path = Path(run_dir).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from checkpoint_store import restore

    restore(run_path, step)
    model_name = re.sub(r"[^0-9A-Za-z_-]+", "_", (name or ("acoustic" if _is_acoustic(kind) else "variance")).strip()) or "model"
    # export.py through a wrapper that keeps onnxsim's metadata from breaking the graph merge (see mvt_ds_export.py)
    args = [str(ENGINE_DIR / "mvt_ds_export.py"), "acoustic" if _is_acoustic(kind) else "variance", "--exp", str(run_path), "--out", str(out)]
    if step is not None:
        args += ["--ckpt", str(step)]
    for spk in export_spk or []:
        args += ["--export_spk", spk]
    if freeze_spk:
        args += ["--freeze_spk", freeze_spk]
    if _is_acoustic(kind):
        if freeze_gender is not None:
            args += ["--freeze_gender", str(freeze_gender)]
        if freeze_velocity:
            args.append("--freeze_velocity")
    else:
        if freeze_glide:
            args.append("--freeze_glide")
        if freeze_expr:
            args.append("--freeze_expr")
    rt.progress(None, "Exporting to ONNX", stage="export")
    _run(args, run_path / "export.log", lambda line: rt.progress(None, line[:160], stage="export") if line.startswith("|") else None,
         env={"MVT_EXPORT_NAME": model_name})
    files = sorted(str(p) for p in out.rglob("*") if p.is_file())
    return {"out_dir": str(out), "files": files, "name": model_name}


@rt.method()
def render(run_dir: str, kind: str, ds: str, out_dir: str, step: int | None = None, spk: str | None = None,
           lang: str | None = None, title: str | None = None, steps: int | None = None, depth: float | None = None,
           predict: list[str] | None = None, seed: int | None = None, device: str = "auto") -> dict[str, Any]:
    """Sings a .ds file with a checkpoint (acoustic: a WAV; variance: a .ds with the predicted parameters)."""
    run_path = Path(run_dir).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ds_path = Path(ds).resolve()
    from checkpoint_store import restore

    restore(run_path, step)
    name = title or ds_path.stem
    args = [str(ENGINE_DIR / "mvt_ds_infer.py"), "acoustic" if _is_acoustic(kind) else "variance", str(ds_path), "--exp", str(run_path),
            "--out", str(out), "--title", name]
    if step is not None:
        args += ["--ckpt", str(step)]
    if spk:
        args += ["--spk", spk]
    if lang:
        args += ["--lang", lang]
    if seed is not None:
        args += ["--seed", str(seed)]
    if _is_acoustic(kind):
        if steps:
            args += ["--steps", str(steps)]
        if depth is not None:
            args += ["--depth", str(depth)]
    else:
        from inference_options import prediction_tags
        config = _load_yaml(run_path / "config.yaml")
        segments = json.loads(ds_path.read_text(encoding="utf-8-sig"))
        for p in prediction_tags(config, segments if isinstance(segments, list) else [segments], predict or []):
            args += ["--predict", p]
    rt.progress(None, "Rendering", stage="render")
    _run(args, run_path / "render.log", env={"MVT_INFER_DEVICE": device})
    produced = sorted(str(p) for p in out.glob(f"{name}*") if p.is_file())
    return {"out_dir": str(out), "files": produced}


@rt.method()
def export_vocoder(checkpoint: str, out_dir: str) -> dict[str, Any]:
    """Convert a SingingVocoders generator export into OpenUtau's ONNX vocoder format."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = out / "export-config.yaml"
    config.write_text(_yaml().safe_dump({"mel_base": "e", "audio_sample_rate": 44100, "hop_size": 512,
        "win_size": 2048, "fft_size": 2048, "audio_num_mel_bins": 128, "fmin": 40, "fmax": 16000}), encoding="utf-8")
    _run([str(ENGINE_DIR / "mvt_ds_export.py"), "nsf-hifigan", "--config", str(config), "--ckpt", checkpoint,
          "--out", str(out), "--name", "vocoder"], out / "export.log")
    return {"out_dir": str(out), "files": sorted(str(p) for p in out.iterdir() if p.is_file())}


@rt.method()
def drop_speakers(checkpoint: str, out: str, drop: list[str] | None = None, retain: list[str] | None = None,
                  fill: str = "zeros") -> dict[str, Any]:
    args = ["scripts/drop_spk.py", str(Path(checkpoint).resolve()), str(Path(out).resolve()), "--fill", fill, "--overwrite"]
    if drop:
        args += ["--drop", ",".join(drop)]
    if retain:
        args += ["--retain", ",".join(retain)]
    _run(args, Path(out).resolve().parent / "export.log")
    return {"path": str(Path(out).resolve())}


@rt.method()
def copy_checkpoint(run_dir: str, retain: list[str], step: int | None = None, fill: str = "zeros") -> dict[str, Any]:
    """A shareable copy with other speaker rows replaced and optimizer state removed."""
    import shutil
    import uuid
    import torch
    from checkpoint_store import listing
    root = Path(run_dir).resolve()
    choices = [c for c in listing(root) if step is None or c["step"] == step]
    if not choices:
        raise ValueError("No matching checkpoint")
    selected = max(choices, key=lambda c: c["step"])
    mapping = json.loads((root / "spk_map.json").read_text(encoding="utf-8"))
    if not retain or set(retain) - set(mapping):
        raise ValueError("Select existing speaker names: " + ", ".join(mapping))
    if fill not in {"zeros", "random", "mean", "cyclic"}:
        raise ValueError("Unknown speaker fill method")
    kept = sorted({int(mapping[name]) for name in retain})
    loaded = torch.load(selected["path"], map_location="cpu", weights_only=False)
    weights = loaded["state_dict"]
    changed = 0
    for key in ("model.fs2.spk_embed.weight", "model.spk_embed.weight"):
        if key not in weights:
            continue
        tensor = weights[key].clone()
        dropped = sorted(set(range(tensor.shape[0])) - set(kept))
        mean = tensor[kept].mean(dim=0)
        for i, speaker_id in enumerate(dropped):
            if fill == "zeros": tensor[speaker_id].zero_()
            elif fill == "random": tensor[speaker_id].normal_()
            elif fill == "mean": tensor[speaker_id] = mean
            else: tensor[speaker_id] = tensor[kept[i % len(kept)]]
        weights[key] = tensor
        changed += len(dropped)
    out = root / "checkpoint-copies" / uuid.uuid4().hex[:12]
    out.mkdir(parents=True)
    target = out / f"model_ckpt_steps_{selected['step']}.ckpt"
    torch.save({"state_dict": weights, "global_step": selected["step"], "epoch": loaded.get("epoch", 0)}, target)
    config = _load_yaml(root / "config.yaml")
    config["datasets"] = [d for d in config.get("datasets", []) if d.get("speaker") in retain]
    (out / "config.yaml").write_text(_yaml().safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "spk_map.json").write_text(json.dumps({name: mapping[name] for name in retain}, ensure_ascii=False), encoding="utf-8")
    for pattern in ("*phonemes*.json", "*languages*.json", "lang_map.json", "dictionary-*.txt"):
        for path in root.glob(pattern):
            if path.is_file(): shutil.copy2(path, out / path.name)
    return {"out_dir": str(out), "path": str(target), "retained": retain, "replaced_rows": changed,
            "optimizer_removed": True, "files": sorted(str(p) for p in out.iterdir())}


@rt.method()
def read_config(run_dir: str) -> dict[str, Any]:
    path = Path(run_dir) / "config.yaml"
    return _load_yaml(path) if path.exists() else {}


@rt.method()
def speakers(run_dir: str) -> dict[str, Any]:
    """The speaker and language maps the trainer copied next to its checkpoints."""
    run_path = Path(run_dir)
    out: dict[str, Any] = {}
    for name in ("spk_map.json", "lang_map.json"):
        path = run_path / name
        if path.exists():
            out[name.split("_")[0]] = json.loads(path.read_text(encoding="utf-8"))
    return out


if __name__ == "__main__":
    rt.run()

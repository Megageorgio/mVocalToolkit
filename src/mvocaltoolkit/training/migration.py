"""Read-only discovery and non-destructive import of DiffTrainer/checkpoint folders."""
from __future__ import annotations
import json
import re
import shutil
import uuid
from pathlib import Path
import yaml

from .dataset import SourceSpec, build_dataset, spec_from_result
from .runs import RunSpec, RunStatus, Checkpoint


def read_config(path: Path, depth=0, seen=None) -> dict:
    if depth > 8 or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Config nesting/size limit exceeded")
    path = path.resolve()
    seen = set(seen or ())
    if path in seen:
        raise ValueError("Cyclic base_config")
    seen.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    merged = {}
    bases = data.get("base_config") or []
    for base in [bases] if isinstance(bases, str) else bases:
        candidate = Path(base)
        choices = [candidate] if candidate.is_absolute() else [path.parent / candidate, path.parent.parent / candidate]
        found = next((p for p in choices if p.is_file()), None)
        if found is None:
            raise ValueError(f"Missing base config: {base}")
        merge_config(merged, read_config(found, depth + 1, seen))
    merge_config(merged, {k: v for k, v in data.items() if k != "base_config"})
    return merged


def merge_config(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value


def inspect_project(folder: str) -> dict:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Choose a DiffTrainer project or checkpoint folder")
    configs = []
    # Avoid traversing environments, model caches or arbitrary directory trees.
    candidates = [*root.glob("*.yaml"), *root.glob("*.yml"), *root.glob("checkpoints/*/config.yaml"), *root.glob("*/config.yaml"), *root.glob("configs/*.yaml")]
    for path in sorted(set(candidates)):
        try:
            cfg = read_config(path)
            if not any(key in cfg for key in ("datasets", "raw_data_dir", "task_cls", "diffusion_type")):
                continue
            kind = "diffsinger_variance" if cfg.get("predict_dur") or cfg.get("predict_pitch") or "variance" in str(cfg.get("task_cls", "")).lower() else "diffsinger_acoustic"
            checkpoints = sorted(path.parent.glob("*.ckpt"))
            configs.append(dict(path=str(path), kind=kind, checkpoints=[str(p) for p in checkpoints], config=cfg))
        except (ValueError, OSError, yaml.YAMLError) as error:
            configs.append(dict(path=str(path), error=str(error)))
    return dict(root=str(root), configs=configs)


def import_project(tk, folder: str, project_dir: str, selected: list[str] | None = None, copy_data=True, register=True, source="main") -> dict:
    if source not in {"main", "benchmark"}:
        raise ValueError("Source must be main or benchmark")
    discovery = inspect_project(folder)
    root = Path(discovery["root"])
    out = Path(project_dir).expanduser().resolve()
    if out == root or root in out.parents or out in root.parents:
        raise ValueError("Import destination and source must not overlap")
    usable = [r for r in discovery["configs"] if not r.get("error") and (not selected or r["path"] in selected)]
    if not usable:
        raise ValueError("No usable selected DiffSinger configs")
    imports, sources, dictionaries, pending = [], [], {}, []
    for record in usable:
        cfg = record["config"]
        run_dir = out / "runs" / ("import-" + uuid.uuid4().hex[:10])
        run_dir.mkdir(parents=True, exist_ok=False)
        origin = Path(record["path"]).parent
        def resolve(value):
            p = Path(value).expanduser()
            if p.is_absolute():
                return p
            return next((candidate for candidate in [root / p, origin / p] if candidate.exists()), root / p)
        rows = cfg.get("datasets") or []
        if not rows and cfg.get("raw_data_dir"):
            rows = [dict(raw_data_dir=p) for p in ([cfg["raw_data_dir"]] if isinstance(cfg["raw_data_dir"], str) else cfg["raw_data_dir"])]
        for i, entry in enumerate(rows):
            if not isinstance(entry, dict) or not entry.get("raw_data_dir"):
                raise ValueError("Dataset entries need raw_data_dir")
            src = SourceSpec(folder=str(resolve(entry["raw_data_dir"])), speaker=str(entry.get("speaker") or f"singer{i}"),
                language=str(entry.get("language") or next(iter(cfg.get("dictionaries") or {}), "ru")))
            sources.append(src.__dict__)
        native_dicts = {lang: str(resolve(p)) for lang, p in (cfg.get("dictionaries") or {}).items()}
        if copy_data and rows:
            built = build_dataset([SourceSpec(**p) for p in sources[-len(rows):]], run_dir / "dataset", native_dicts or None)
            snapshot = spec_from_result(built)
            cfg["datasets"] = snapshot["datasets"]
            cfg["dictionaries"] = snapshot["dictionaries"]
            dictionaries.update(snapshot["dictionaries"])
        else:
            cfg["datasets"] = [{**entry, "raw_data_dir": str(resolve(entry["raw_data_dir"]))} for entry in rows]
            cfg["dictionaries"] = native_dicts
        for name in ("vocoder_ckpt", "pe_ckpt", "hnsep_ckpt"):
            if cfg.get(name):
                cfg[name] = str(resolve(cfg[name]))
        original_binary = resolve(cfg["binary_data_dir"]) if cfg.get("binary_data_dir") else None
        if original_binary and original_binary.is_dir():
            if copy_data:
                shutil.copytree(original_binary, run_dir / "binary")
                cfg["binary_data_dir"] = str(run_dir / "binary")
            else:
                cfg["binary_data_dir"] = str(original_binary)
        cfg["work_dir"] = str(run_dir)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        # Checkpoint bytes are copied, never unpickled during inspection/import.
        copied = []
        for number, checkpoint in enumerate(record["checkpoints"]):
            source = Path(checkpoint)
            match = re.search(r"steps_(\d+)", source.name)
            step = int(match.group(1)) if match else number
            target = run_dir / f"model_ckpt_steps_{step}.ckpt"
            if target.exists():
                raise ValueError("Checkpoint step collision; select a folder with unique steps")
            shutil.copy2(source, target)
            copied.append(Checkpoint(step=step, path=str(target), size=target.stat().st_size))
        for pattern in ("*phonemes*.json", "*languages*.json", "spk_map.json", "lang_map.json"):
            for path in origin.glob(pattern):
                shutil.copy2(path, run_dir / path.name)
        # Register stopped, without scheduling training or changing source files.
        from .runs import Run, RunInfo
        config = dict(overrides=cfg, datasets=cfg.get("datasets", []), dictionaries=cfg.get("dictionaries", {}), imported=True, source=source)
        spec = RunSpec(kind=record["kind"], name=origin.name, dir=str(run_dir), config=config)
        run = Run(RunInfo(id=uuid.uuid4().hex[:10], spec=spec, dir=str(run_dir), status=RunStatus.stopped,
            checkpoints=copied, step=max((c.step for c in copied), default=0), config_written=cfg, message="Imported; review paths and settings before resuming"))
        run.save(force=True)
        if register:
            tk.training.runs[run.id] = run
            tk.training._save_index()
        else:
            pending.append(run.info.model_dump(mode="json"))
        (run_dir / "import-report.json").write_text(json.dumps(dict(source=record["path"], checkpoints=record["checkpoints"], copied_data=copy_data), indent=1), encoding="utf-8")
        imports.append(dict(id=run.id, dir=str(run_dir), kind=spec.kind, checkpoints=len(copied)))
    return dict(runs=imports, sources=list({p["folder"]: p for p in sources}.values()), dictionaries=dictionaries, pending=pending)


def stage_external(checkpoint: str, config: str, out: Path) -> Path:
    source, cfg_path = Path(checkpoint).expanduser().resolve(), Path(config).expanduser().resolve()
    if source.suffix.lower() != ".ckpt" or not source.is_file():
        raise ValueError("Choose an existing DiffSinger .ckpt")
    if out.exists() and any(out.iterdir()):
        raise ValueError("External checkpoint staging must be empty")
    cfg = read_config(cfg_path)
    if not any(k in cfg for k in ("task_cls", "audio_sample_rate", "sample_rate")):
        raise ValueError("Config does not look like a resolved DiffSinger config")
    out.mkdir(parents=True, exist_ok=True)
    match = re.search(r"steps_(\d+)", source.name)
    shutil.copy2(source, out / f"model_ckpt_steps_{int(match.group(1)) if match else 0}.ckpt")
    for key in ("dictionaries",):
        cfg[key] = {lang: str((cfg_path.parent / p).resolve()) if not Path(p).is_absolute() else p for lang, p in (cfg.get(key) or {}).items()}
    for key in ("vocoder_ckpt", "pe_ckpt", "hnsep_ckpt"):
        if cfg.get(key) and not Path(cfg[key]).is_absolute():
            cfg[key] = str((cfg_path.parent / cfg[key]).resolve())
    cfg["work_dir"] = str(out)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    for pattern in ("*phonemes*.json", "*languages*.json", "spk_map.json", "lang_map.json"):
        for path in cfg_path.parent.glob(pattern):
            shutil.copy2(path, out / path.name)
    return out

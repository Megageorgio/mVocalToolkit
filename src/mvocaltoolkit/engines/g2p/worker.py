"""Dictionary snapshots and detached G2P trainer commands."""
import json
import sys
from pathlib import Path

import mvt_engine as rt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "diffsinger"))
from checkpoint_store import listing


@rt.method()
def environment():
    import torch
    return {"torch": torch.__version__, "cuda": torch.cuda.is_available()}


@rt.method()
def prepare(run_dir: str, kind: str, spec: dict, assets=None, device="auto"):
    import yaml
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    import hashlib
    content = Path(spec["prepared_file"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != spec["prepared_sha256"]:
        raise ValueError("The prepared dictionary snapshot changed")
    data = json.loads(content)
    if data.get("base_weights"):
        import hashlib
        import shutil
        source = Path(data["base_weights"]["path"])
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != data["base_weights"]["sha256"]:
                raise ValueError("The base model changed after preflight; check the dictionary again")
        shutil.copy2(source, path / "base-model.ptsd")
    (path / "dictionary-data.json").write_bytes(content)
    (path / "config.yaml").write_text(yaml.safe_dump(data["model"], allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"config": {k: v for k, v in data.items() if k not in {"train", "validation", "entries", "model"}},
            "max_steps": data["max_updates"], "launch": {"args": [str(HERE / "train.py"), "--run-dir", str(path), "--device", device]}}


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


@rt.method()
def predict(run_dir: str, words: list[str], step=None):
    from exporter import load_model
    model, data, _ = load_model(Path(run_dir), step)
    import unicodedata
    return {"items": [{"word": w, "phones": model.predict_str(unicodedata.normalize("NFC", w).lower() if data["lowercase"] else unicodedata.normalize("NFC", w))} for w in words]}


if __name__ == "__main__":
    rt.run()

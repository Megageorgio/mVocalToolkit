"""Native WFL/MCFL networks and boundary-refiner training adapters."""
import json
import os
from pathlib import Path

from mvt_train_common import device, merge, seed
from mvt_torch_loop import fit


def train(run, source, kind, requested, resume):
    import torch
    import yaml
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    cfg["resolved_device"] = device(requested); seed(cfg["seed"])
    os.chdir(run)
    if kind == "refiner_boundary":
        refiner(run, source, cfg, resume); return
    from mcfl import engine as E, project as P, wfl as W
    native = P.load_config()
    merge(native, cfg["overrides"])
    stage = "wfl" if kind == "wfl_asr_teacher" else "mcfl"
    native[stage]["train"].update(epochs=cfg["epochs"], batch=cfg["batch_size"], lr=cfg["learning_rate"], patience=cfg["patience"])
    native["sampling"]["epoch_size"] = max(cfg["batch_size"], min(6000, cfg["steps"] * cfg["batch_size"]))
    native["data"]["min_val_files"] = 1
    if stage == "mcfl":
        native["mcfl"]["size"] = {"small": "xxxs", "medium": "s", "large": "l"}.get(cfg["size"], cfg["size"])
    if cfg["encoder"]: native["wfl"]["whisper_model"] = cfg["encoder"]
    if not cfg["allow_download"]: os.environ["HF_HUB_OFFLINE"] = "1"
    original_trainer = E.Trainer
    class ManagedTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if stage == "mcfl" and cfg["base_model"] and not resume:
                base = Path(cfg["base_model"])
                if base.is_dir() and (base / "model.safetensors").is_file():
                    from safetensors.torch import load_file
                    weights = load_file(str(base / "model.safetensors"))
                else:
                    if base.is_dir(): base = base / "best.pt"
                    state = torch.load(base, map_location="cpu", weights_only=False)
                    weights = state.get("state_dict", state.get("model", state))
                self.model.load_state_dict(weights, strict=True)
        def resumed(self):
            return resume
        def fit(self, train_dl, sampler, step, validate, save_best, ref=None):
            def validation():
                results = validate()
                return E.score(results), {"languages": results, "teacher": ref, "metric": "PER and boundary error"}
            metadata = {"kind": kind, "stage": stage}
            if stage == "mcfl":
                metadata.update(yaml.safe_load((run / "mcfl/mcfl.yaml").read_text(encoding="utf-8")))
            fit(run, cfg, self.model, self.opt, self.sched, train_dl, step, validation,
                metadata=metadata, ema=self.ema, resume=resume)
    E.Trainer = ManagedTrainer
    if stage == "wfl":
        original_build, original_config = W.build, W.wfl_config
        if cfg["base_model"]:
            base = Path(cfg["base_model"])
            prepare_data = P.prepare
            P.prepare = lambda data_root, run_dir, config, teacher_dir=None: prepare_data(data_root, run_dir, config, str(base))
            def model_config(manifest, config, groups):
                labels, langs, _, _ = P.read_label_files(str(base))
                if labels != manifest["labels"] or [l for l,_ in sorted(langs.items(), key=lambda p:p[1])] != manifest["langs"]:
                    raise ValueError("WFL fine-tuning needs the same label IDs and language IDs as the base model")
                return yaml.safe_load((base / "config.yaml").read_text(encoding="utf-8"))
            W.wfl_config = model_config
        def build(config, labels, home):
            if cfg["base_model"] and (Path(cfg["base_model"]) / "encoder").is_dir():
                config["model"]["whisper_model"] = str(Path(cfg["base_model"]) / "encoder")
            model = original_build(config, labels, home)
            W.save_folder(model, config, str(run), str(run / "wfl"))
            if cfg["base_model"] and not resume:
                state = torch.load(W.checkpoint_of(cfg["base_model"]), map_location="cpu", weights_only=False)
                weights = state.get("state_dict", state)
                model.load_state_dict({k.removeprefix("model."):v for k,v in weights.items()}, strict=True)
            return model
        W.build = build
        from mcfl.train_wfl import train_wfl
        train_wfl(str(run), str(run / "raw"), native, cfg["resolved_device"], 0)
    else:
        from mcfl.train_mcfl import train_mcfl
        train_mcfl(str(run), str(run / "raw"), native, cfg["resolved_device"], 0, cfg["teacher"])


def refiner(run, source, cfg, resume):
    import importlib.util
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from mrefinermodel.data import BoundaryDataset, collate, read_lab
    from mrefinermodel.model import BoundaryRefiner
    native = yaml.safe_load((source / "config/default.yaml").read_text())
    if cfg["base_model"]:
        merge(native,yaml.safe_load((Path(cfg["base_model"]) / "config.yaml").read_text(encoding="utf-8")))
    merge(native, cfg["overrides"])
    native.setdefault("training", {}).update(lr=cfg["learning_rate"], real_frac=0, max_epochs=cfg["epochs"])
    train_files = sorted((run / "raw/train").glob("*.wav")); val_files = sorted((run / "raw/val").glob("*.wav"))
    phones = sorted({p for file in train_files for _,_,p in read_lab(str(file.with_suffix(".lab")))})
    if cfg["base_model"]:
        phones = (Path(cfg["base_model"]) / "phonemes.txt").read_text(encoding="utf-8").splitlines()
        missing = {p for file in [*train_files, *val_files] for _,_,p in read_lab(str(file.with_suffix(".lab")))} - set(phones)
        if missing: raise ValueError("Base refiner lacks phonemes: " + ", ".join(sorted(missing)))
    (run / "phonemes.txt").write_text("\n".join(phones) + "\n", encoding="utf-8")
    (run / "refiner.yaml").write_text(yaml.safe_dump(native), encoding="utf-8")
    p2i = {p:i for i,p in enumerate(phones)}
    train_ds = BoundaryDataset(list(map(str, train_files)), p2i, native, train=True, seed=cfg["seed"])
    val_ds = BoundaryDataset(list(map(str, val_files)), p2i, native, train=False, seed=cfg["seed"])
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], collate_fn=collate, num_workers=0)
    validation = DataLoader(val_ds, batch_size=1, collate_fn=collate, num_workers=0)
    model = BoundaryRefiner(len(phones), native).to(cfg["resolved_device"])
    if cfg["base_model"] and not resume:
        weights = torch.load(Path(cfg["base_model"]) / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(weights.get("state_dict", weights), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    spec = importlib.util.spec_from_file_location("mvt_refiner_native", source / "train.py")
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    def loss(item):
        if not (item["tgt"] >= 0).any(): raise ValueError("No touching phoneme boundaries in this batch")
        out = model(item["win"].to(cfg["resolved_device"]), item["ctx"].to(cfg["resolved_device"]), item["dur"].to(cfg["resolved_device"]))
        return torch.nn.functional.cross_entropy(out, item["tgt"].to(cfg["resolved_device"]), label_smoothing=.05, ignore_index=-100), {}
    def validate():
        result = evaluator.evaluate(model, validation, native["frontend"]["half_window_frames"], native["frontend"]["hop_ms"]/1000, cfg["resolved_device"])
        if result is None: raise ValueError("Validation has no usable boundaries")
        return result["after_mae"], {"boundary_ms": result, "metric": "Boundary MAE on synthetic shifts"}
    fit(run, cfg, model, optimizer, None, loader, loss, validate, metadata={"kind": "refiner_boundary"}, resume=resume)

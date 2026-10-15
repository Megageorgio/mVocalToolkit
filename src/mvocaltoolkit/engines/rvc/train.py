"""One-device RVC trainer with portable checkpoint state and cooperative batch-boundary stop."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diffsinger"))
from checkpoint_store import listing
from adapter import adapt


class Finished(Exception):
    pass


def atomic_save(value, path):
    import torch
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(run: Path, device: str, resume: bool):
    import numpy as np
    import soundfile as sf
    import torch
    import yaml
    from torch.utils.data import DataLoader
    source = Path(os.environ["MVT_SOURCE_DIR"]).resolve()
    os.chdir(source)
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps" or device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("The selected RVC training device is unavailable")
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    # The pinned trainer uses availability checks to move tensors; honor an explicit CPU selection.
    if device == "cpu":
        torch.cuda.is_available = lambda: False
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    saved = listing(run)
    state = torch.load(saved[-1]["path"], map_location="cpu", weights_only=True) if resume and saved else None
    resume_epoch = state["epoch"] if state else 1
    resume_batch = state["batch"] if state else 0
    initial_step = state["step"] if state else 0
    limit = config["max_steps"]
    if state and limit and initial_step >= limit:
        raise ValueError("The run already reached its step limit; start a longer run to continue training")

    def emit(event, **values):
        with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, "time": time.time(), **values}, allow_nan=False) + "\n")

    last = None
    validation_loader = None
    best_validation = float("inf")
    if (run / "validation.json").is_file():
        best_validation = json.loads((run / "validation.json").read_text(encoding="utf-8")).get("best", best_validation)
    def checkpoint(values, epoch_finished=False):
        step = namespace["global_step"]
        path = run / f"model_ckpt_steps_{step}.ckpt"
        epoch = values["epoch"]
        epoch_finished = epoch_finished or values["batch_idx"] + 1 >= len(values["train_loader"])
        optimizers = [values["optim_g"], values["optim_d"]]
        payload = {"format": "rvc-applio-v1", "step": step,
                   "epoch": epoch + 1 if epoch_finished else epoch,
                   "batch": 0 if epoch_finished else values["batch_idx"] + 1,
                   "g": values["net_g"].state_dict(), "d": values["net_d"].state_dict(),
                   "optim_g": optimizers[0].state_dict(), "optim_d": optimizers[1].state_dict(),
                   "scaler": values["scaler"].state_dict(), "python_rng": random.getstate(),
                   "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                   "numpy_rng": {"kind": np.random.get_state()[0], "keys": torch.from_numpy(np.random.get_state()[1].astype(np.int64)),
                                 "position": np.random.get_state()[2], "has_gauss": np.random.get_state()[3], "cached": np.random.get_state()[4]},
                   "native_config": json.loads((run / "config.json").read_text(encoding="utf-8")), "settings": config}
        if epoch_finished:
            # Upstream steps schedulers after returning from the epoch. Store the next epoch's rates.
            decay = namespace["config"].train.lr_decay
            for key in ("optim_g", "optim_d"):
                for group in payload[key]["param_groups"]:
                    group["lr"] *= decay
        atomic_save(payload, path)
        emit("checkpoint", step=step, path=str(path), size=path.stat().st_size)
        for old in listing(run)[:-3]:
            if not old["starred"] and Path(old["path"]).parent == run and not (run / "best" / Path(old["path"]).name).is_file():
                Path(old["path"]).unlink(missing_ok=True)
                emit("checkpoint_removed", step=old["step"])

    def sample(values):
        nonlocal validation_loader, best_validation
        folder = run / "samples" / str(namespace["global_step"])
        folder.mkdir(parents=True, exist_ok=True)
        model = values["net_g"]
        validation_file = run / "validation-filelist.txt"
        if validation_file.is_file() and validation_file.read_text(encoding="utf-8").strip():
            if validation_loader is None:
                import copy
                from data_utils import TextAudioLoaderMultiNSFsid, TextAudioCollateMultiNSFsid
                data = copy.deepcopy(namespace["config"].data)
                data.training_files = str(validation_file)
                python_rng = random.getstate()
                try:
                    dataset = TextAudioLoaderMultiNSFsid(data)
                finally: random.setstate(python_rng)
                validation_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=TextAudioCollateMultiNSFsid())
            from heldout import evaluate
            metrics, rendered = evaluate(model, validation_loader, folder, device, config["sample_rate"], config["seed"], lambda: (run / "STOP").exists())
            files = [str(Path(p).relative_to(run)) for p in rendered]
            score = metrics["heldout_log_spectral_l1"]
            better = score < best_validation
            best_validation = min(best_validation, score)
            (run / "validation.json").write_text(json.dumps(dict(step=namespace["global_step"], score=score, best=best_validation, metrics=metrics)), encoding="utf-8")
            emit("validation", step=namespace["global_step"], val_loss=score, metrics=metrics)
            emit("samples", step=namespace["global_step"], files=files)
            if better:
                from checkpoint_store import retain_best
                retain_best(run, namespace["global_step"])
            return
        model.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith("cuda") else []), torch.inference_mode():
            audio, *_ = model.infer(*values["reference"])
        model.train()
        sf.write(folder / "generated.wav", audio[0, 0].float().cpu().numpy(), config["sample_rate"])
        import shutil
        shutil.copy2(run / "reference.wav", folder / "original.wav")
        emit("samples", step=namespace["global_step"], files=[str((folder / n).relative_to(run)) for n in ("original.wav", "generated.wav")])

    restore_rng = state is not None
    def batch(values):
        nonlocal last
        last = values
        step = namespace["global_step"]
        losses = {"total_loss": float(values["loss_gen_all"].detach()), "discriminator": float(namespace["loss_disc"].detach()),
                  "mel": float(values["loss_mel"].detach()), "kl": float(values["loss_kl"].detach()),
                  "feature": float(values["loss_fm"].detach())}
        if not all(np.isfinite(v) for v in losses.values()):
            raise ValueError("Non-finite RVC loss; reduce learning rate and check recordings")
        max_steps = min(limit, len(values["train_loader"]) * config["epochs"]) if limit else len(values["train_loader"]) * config["epochs"]
        emit("step", step=step, max_steps=max_steps, epoch=values["epoch"], losses=losses,
             lr=values["optim_g"].param_groups[0]["lr"],
             gpu_mb=torch.cuda.memory_allocated() / 2**20 if device.startswith("cuda") else 0)
        stopped = (run / "STOP").exists()
        if stopped or limit and step >= limit:
            checkpoint(values)
            sample(values)
            emit("end", status="stopped" if stopped else "done")
            raise Finished()

    def reference(info):
        sf.write(run / "reference.wav", info[6][0, 0, :int(info[7][0])].float().cpu().numpy(), config["sample_rate"])

    def loader(*args, **kwargs):
        kwargs.update(num_workers=0, persistent_workers=False, prefetch_factor=None, pin_memory=device.startswith("cuda"))
        kwargs["generator"] = torch.Generator().manual_seed(config["seed"])
        return DataLoader(*args, **kwargs)

    def latest(directory, pattern):
        if state is None:
            return None
        return "G" if pattern.startswith("G") else "D"

    def load(name, model, optimizer=None, load_opt=1):
        if state is None:
            raise FileNotFoundError("No resume checkpoint")
        key = "g" if name == "G" else "d"
        model.load_state_dict(state[key], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(state["optim_" + key])
        return model, optimizer, config["learning_rate"], resume_epoch, state["scaler"]

    native_path = source / "rvc" / "train" / "train.py"
    sys.path.insert(0, str(native_path.parent))
    sys.argv = [str(native_path), "voice", str(config["save_every_epoch"]), str(config["epochs"]),
                config.get("base_generator") or "None", config.get("base_discriminator") or "None", "0",
                str(config["batch_size"]), str(config["sample_rate"]), "False", "False", "False", "False", "50", "False", "HiFi-GAN", "False"]
    namespace = {"__name__": "mvt_rvc_native", "__file__": str(native_path), "_mvt_batch": batch,
                 "_mvt_reference": reference, "_mvt_resume_epoch": resume_epoch,
                 "_mvt_resume_step": initial_step, "_mvt_resume_batch": resume_batch, "_mvt_has_resume": state is not None,
                 "_mvt_speaker_rows": int(state["g"]["emb_g.weight"].shape[0]) if state else None}
    code = compile(adapt(native_path.read_text(encoding="utf-8"), str(run)), str(native_path), "exec")
    exec(code, namespace)
    namespace.update(DataLoader=loader, latest_checkpoint_path=latest, load_checkpoint=load, train_dtype=torch.float32)
    original_epoch = namespace["train_and_evaluate"]
    def epoch(*args, **kwargs):
        nonlocal restore_rng
        if restore_rng:
            random.setstate(state["python_rng"]); torch.set_rng_state(state["torch_rng"])
            rng = state["numpy_rng"]
            np.random.set_state((rng["kind"], rng["keys"].numpy().astype(np.uint32), rng["position"], rng["has_gauss"], rng["cached"]))
            if device.startswith("cuda") and state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            restore_rng = False
        original_epoch(*args, **kwargs)
        number = args[1]
        if last and (number % config["save_every_epoch"] == 0 or number == config["epochs"]):
            checkpoint(last, epoch_finished=True)
            sample(last)
    namespace["train_and_evaluate"] = epoch
    # Native checkpoints duplicate hundreds of MB; our atomic bundle contains both models and optimizers.
    namespace["save_checkpoint"] = lambda *a, **k: None
    namespace["extract_model"] = lambda *a, **k: None
    emit("start", step=initial_step, max_steps=limit or None)
    try:
        namespace["run"](0, 1, str(run), sys.argv[4], sys.argv[5], config["epochs"], False,
                         namespace["config"], torch.device(device), torch.cuda.current_device() if device.startswith("cuda") else 0)
    except Finished:
        return
    emit("end", status="done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default=os.environ.get("MVT_DEVICE", "auto"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        train(args.run_dir.resolve(), args.device, args.resume)
    except Exception as exc:
        with (args.run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "end", "time": time.time(), "status": "failed", "error": str(exc),
                                     "error_kind": "oom" if "out of memory" in str(exc).lower() else "crash"}) + "\n")
        raise


if __name__ == "__main__":
    main()

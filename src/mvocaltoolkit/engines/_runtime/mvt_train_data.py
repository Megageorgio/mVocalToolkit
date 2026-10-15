"""Write native trainer inputs from a verified, recording-disjoint manifest."""
import csv
import hashlib
import json
import os
from pathlib import Path

from mvt_train_common import device, load_config, merge, resolve_model, seed


def progress(value, message):
    print("MVT_PROGRESS " + json.dumps({"progress": value, "message": message}), flush=True)


def prepare(run, source, kind, requested):
    import soundfile as sf
    import soxr
    import yaml
    data = json.loads((run / "input-manifest.json").read_text(encoding="utf-8"))
    cfg = data["settings"]
    cfg["device"] = requested
    seed(cfg["seed"])
    raw = (run / "raw").resolve()
    if raw.exists():
        if raw.parent != run.resolve() or not (raw / ".mvt-generated").is_file():
            raise ValueError("Existing raw data is not managed by this run; use a new run folder")
        import shutil
        shutil.rmtree(raw)
    raw.mkdir(); (raw / ".mvt-generated").write_text("training input snapshot", encoding="utf-8")
    rows = {}
    records = []
    for i, record in enumerate(data["records"]):
        original = Path(record["audio"])
        with original.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                raise ValueError(f"Recording changed after preflight: {original}")
            stream.seek(0); audio, rate = sf.read(stream, dtype="float32", always_2d=True)
        import numpy as np
        if not np.isfinite(audio).all(): raise ValueError(f"Non-finite audio: {original}")
        audio = audio.mean(axis=1)
        split = "val" if record["validation"] else "train"
        name = record["sha256"][:24]
        lang = record["language"]
        if kind in {"wfl_asr_teacher", "mcfl_student"}:
            folder = run / "raw"
            if record["validation"]: folder /= "val"
            folder = folder / lang / hashlib.sha256(record["speaker"].encode()).hexdigest()[:12]
            target_rate = 16000
        elif kind == "refiner_boundary":
            folder = run / "raw" / split
            name = hashlib.sha256(record["speaker"].encode()).hexdigest()[:12] + "__" + name
            target_rate = 16000
        elif kind in {"qwen_tts_voice", "voxcpm_voice"}:
            folder = run / "raw" / split; target_rate = 24000 if kind == "qwen_tts_voice" else 16000
        else:
            folder = run / "raw"
            if kind == "sofa_aligner": folder /= "full_label"
            folder = folder / split / lang / ("wavs" if kind == "sofa_aligner" else "waveforms")
            target_rate = 44100 if kind in {"game_notes", "sofa_aligner"} else 48000
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.wav"
        if rate != target_rate: audio = soxr.resample(audio, rate, target_rate)
        sf.write(path, audio, target_rate, subtype="PCM_16")
        record = {**record, "prepared_audio": str(path)}
        if "phones" in record:
            path.with_suffix(".lab").write_text("\n".join(f"{round(a*1e7)} {round(b*1e7)} {p}" for a,b,p in record["phones"]) + "\n", encoding="utf-8")
        if kind in {"tifa_aligner", "sofa_aligner", "game_notes"}:
            tier = record.get("notes") if kind == "game_notes" else record["phones"]
            row = {"name": name, "language": lang}
            if kind == "sofa_aligner":
                row.update(ph_seq=" ".join(p for _,_,p in tier), ph_dur=" ".join(str(b-a) for a,b,_ in tier))
                file = folder.parent / "transcriptions.csv"
            else:
                row.update({"notes" if kind == "game_notes" else "phones": " ".join(p for _,_,p in tier), "durations": " ".join(str(b-a) for a,b,_ in tier)})
                file = folder.parent / "index.csv"
            rows.setdefault(file, []).append(row)
        records.append(record)
        progress(.5 * (i + 1) / len(data["records"]), f"Snapshot {i+1}/{len(data['records'])}")
    for file, items in rows.items():
        with file.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(items[0])); writer.writeheader(); writer.writerows(items)
    (run / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.chdir(source)
    if kind in {"tifa_aligner", "game_notes"}:
        native = load_config(source, source / "configs" / ("supervised.yaml" if kind == "tifa_aligner" else "midi.yaml"))
        merge(native, cfg["overrides"])
        binary = native["binarizer"]
        binary["num_workers"] = 0
        if kind == "tifa_aligner":
            binary.update(phoneme_timing_data_dir=str(run / "raw"), text_only_data_dir=None, split={"subdir": "val", "count": -1, "seed": cfg["seed"]})
            dictionary_dir = run / "dictionaries"; dictionary_dir.mkdir(exist_ok=True)
            converters = []
            for language, content in data.get("dictionaries", {}).items():
                dictionary = dictionary_dir / f"{language}.txt"; dictionary.write_text(content, encoding="utf-8")
                converters.append({"id":"dictionary","language":language,"kwargs":{"dict_path":str(dictionary)}})
            binary["g2p"] = {"converters": [*converters, {"id": "passthrough"}]}
            present = {p for r in records for _,_,p in r["phones"]}
            vocabulary = binary.setdefault("vocabulary", {})
            vocabulary["merged_groups"] = [group for group in vocabulary.get("merged_groups", []) if any(p in present or p.split("/",1)[-1] in present for p in group)]
            if cfg["base_model"]:
                base_vocab = Path(cfg["base_model"])
                base_vocab = (base_vocab if base_vocab.is_dir() else base_vocab.parent) / "vocabulary.json"
                if base_vocab.is_file():
                    import shutil
                    target = run / "base-vocabulary.json"; shutil.copy2(base_vocab,target)
                    vocabulary["prebuilt_vocab_file"] = str(target)
            native.setdefault("inference", {})["g2p"] = binary["g2p"]
            native["training"]["semisupervised"]["enabled"] = False
        else:
            binary.update(data_dir=str(run / "raw"), validation_count=data["validation_files"])
        tr = native["training"]
        tr["dataloader"].update(num_workers=0, max_batch_size=cfg["batch_size"], max_val_batch_size=1)
        tr["trainer"].update(max_steps=cfg["steps"], max_epochs=cfg["epochs"], min_steps=0, min_epochs=0,
                            val_every_n_units=cfg["save_interval"], unit="step", devices=1, precision="32-true",
                            accelerator="cpu" if device(requested) == "cpu" else "gpu", num_sanity_val_steps=0,
                            accumulate_grad_batches=cfg["accumulate"], checkpoints=[{"tag":"mvt","type":"periodic","unit":"step",
                            "every_n_units":cfg["save_interval"],"since_m_units":0,"save_last_k":1,"weights_only":False}])
        tr["trainer"]["strategy"] = {"name": "auto", "kwargs": {}}
        tr["optimizer"]["kwargs"]["lr"] = cfg["learning_rate"]
        for key in ("natural_noise", "rir_reverb"):
            if key in tr.get("augmentation", {}): tr["augmentation"][key]["enabled"] = False
        if cfg["base_model"]:
            base = Path(cfg["base_model"])
            if base.is_dir():
                import yaml
                saved_config = yaml.safe_load((base / "config.yaml").read_text(encoding="utf-8"))
                native["model"] = saved_config["model"]
                candidates = [*base.glob("model.pt"), *base.glob("*.ckpt")]
                if not candidates: raise ValueError("The base folder has no model.pt or checkpoint")
                base = candidates[0]
            tr["finetuning"].update(pretraining_enabled=True, pretraining_from=str(base.resolve()))
        path = run / "native.yaml"; path.write_text(yaml.safe_dump(native, sort_keys=False), encoding="utf-8")
        if kind == "game_notes":
            from preprocessing.binarizer_base import BaseBinarizer
            def split(self, items):
                self.valid_items = [item for item in items if "val" in item.waveform_fn.relative_to(run / "raw").parts]
                self.train_items = [item for item in items if item not in self.valid_items]
                if not self.train_items or not self.valid_items: raise ValueError("Both splits need usable notes")
            BaseBinarizer.split_dataset = split
        import runpy, sys
        sys.argv = ["binarize.py", "--config", str(path)]
        try:
            runpy.run_path(str(source / "binarize.py"), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None,0): raise
        import numpy as np
        for split_name in ("train", "valid"):
            with np.load(run / "raw" / f"{split_name}.info.npz") as info:
                if not any(value.size for value in info.values()): raise ValueError(f"No usable {split_name} items after binarization")
    elif kind == "sofa_aligner":
        native = yaml.safe_load((source / "configs/binarize_config.yaml").read_text())
        if cfg["base_model"]:
            from mvt_sofa_base import prepare_base
            cfg["sofa_base_checkpoint"],params=prepare_base(run,cfg["base_model"])
            cfg["sofa_model_config"]=params["model_config"]
            native["melspec_config"]=params["melspec_config"]
        native.update(data_folder=str(run / "raw"), valid_set_size=data["validation_files"], valid_set_preferred_folders=["val"],
                      data_augmentation={"size": 0, "key_shift_choices": []})
        (run / "raw/binary").mkdir(exist_ok=True)
        (run / "raw/binary/global_config.yaml").write_text(yaml.safe_dump({"max_length":native["max_length"],
            "melspec_config":native["melspec_config"],"data_augmentation_size":0}),encoding="utf-8")
        import binarize
        import torch
        if device(requested) == "cpu": torch.cuda.is_available = lambda: False
        def load_wave(path, target_device, sample_rate=None):
            samples, original_rate = sf.read(path, dtype="float32", always_2d=True)
            samples = samples.mean(axis=1)
            if sample_rate and original_rate != sample_rate: samples = soxr.resample(samples, original_rate, sample_rate)
            return torch.from_numpy(samples).to(target_device)
        binarize.load_wav = load_wave
        binarize.ForcedAlignmentBinarizer(**native).process()
    elif kind in {"qwen_tts_voice", "voxcpm_voice"}:
        references = {}
        for record in records:
            if not record["validation"] and record["speaker"] not in references:
                references[record["speaker"]] = dict(audio=record["prepared_audio"], text=record["text"])
        cfg["speaker_references"] = references
        for record in records: record["ref_audio"] = references[record["speaker"]]["audio"]
        cfg["base_model_origin"] = cfg["base_model"]
        cfg["base_model"] = str(resolve_model(cfg["base_model"], cfg["allow_download"]))
        if kind == "qwen_tts_voice":
            from qwen_tts import Qwen3TTSTokenizer
            local = Path(cfg["base_model"]) / "speech_tokenizer"
            tokenizer_path = cfg["tokenizer"] or (str(local) if local.is_dir() else "Qwen/Qwen3-TTS-Tokenizer-12Hz")
            tokenizer_path = resolve_model(tokenizer_path, cfg["allow_download"])
            cfg["tokenizer"] = str(tokenizer_path)
            tokenizer = Qwen3TTSTokenizer.from_pretrained(str(tokenizer_path), device_map=device(requested))
            for i, r in enumerate(records):
                r["audio_codes"] = tokenizer.encode(r["prepared_audio"]).audio_codes[0].cpu().tolist()
                progress(.5 + .5 * (i+1)/len(records), f"Audio tokens {i+1}/{len(records)}")
        for validation, name in ((False, "train.jsonl"), (True, "validation.jsonl")):
            with (run / name).open("w", encoding="utf-8") as stream:
                for r in records:
                    if r["validation"] == validation:
                        stream.write(json.dumps({**r, "audio": r["prepared_audio"]}, ensure_ascii=False) + "\n")
        (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    progress(1, "Training data ready")

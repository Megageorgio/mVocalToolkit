"""DiffSinger run specs -> the config overlay the engine writes over openvpi's defaults.

The spec (RunSpec.config of a diffsinger_acoustic / diffsinger_variance run):

    {
      "datasets": [{"raw_data_dir": "...", "speaker": "masha", "spk_id": 0, "language": "ru",
                    "test_prefixes": ["a01", "b07"]}],
      "dictionaries": {"ru": "/path/dictionary-ru.txt"},
      "extra_phonemes": [], "merged_phoneme_groups": [],
      "assets": {"vocoder": "<model id or path>", "pe": "...", "hnsep": "..."},   # defaults: the catalog's
      "preset": {"goal": "full | check | preview | finetune", "gpu_gb": 16, "size": "small | medium | large"},
      "variance": {"dur": true, "pitch": true, "energy": false, "breathiness": true, "voicing": false,
                   "tension": false},                                               # variance runs
      "finetune": {"ckpt": "/path/model_ckpt_steps_N.ckpt", "ignored_params": [...], "strict_shapes": false},
      "overrides": {"max_updates": 50000}                                             # any config key, applied last
    }

The overlay holds what a user config (configs/templates/config_*.yaml) holds, with the community's tuned defaults
(RMVPE, f0_max 1600, L1 loss, pitch shift ±3 at scale 0.25, tension logits ±8) and the preset's numbers. The engine
merges it over configs/base.yaml and configs/acoustic.yaml | variance.yaml of the pinned source, so upstream's
defaults never have to be copied here.
"""

from __future__ import annotations

import copy
import os
from typing import Any

ASSET_DEFAULTS = {
    "vocoder": "diffsinger-pc-nsf-hifigan-2025.02",
    "pe": "diffsinger-rmvpe-230917",
    "hnsep": "diffsinger-hnsep-240512",
}

# GPU memory (GB) -> (max_batch_frames, max_batch_size); the trainer's probe epoch caps them further when needed
BATCH_ACOUSTIC = {6: (20000, 8), 8: (30000, 12), 12: (40000, 24), 16: (50000, 32), 24: (80000, 64)}
BATCH_VARIANCE = {6: (32000, 8), 8: (48000, 12), 12: (64000, 24), 16: (80000, 36), 24: (120000, 48)}

# training length by data size: (max_updates, val_check_interval, permanent_ckpt_start)
LENGTH_ACOUSTIC = {"small": (60000, 2000, 40000), "medium": (100000, 4000, 60000), "large": (160000, 4000, 100000)}
LENGTH_VARIANCE = {"small": (50000, 2000, 30000), "medium": (80000, 2000, 50000), "large": (120000, 4000, 80000)}

GOALS = ("full", "check", "preview", "finetune")
SIZES = ("small", "medium", "large")

VARIANCE_KEYS = {"dur": "predict_dur", "pitch": "predict_pitch", "energy": "predict_energy",
                 "breathiness": "predict_breathiness", "voicing": "predict_voicing", "tension": "predict_tension"}


def _template_acoustic() -> dict[str, Any]:
    """configs/templates/config_acoustic.yaml at the pinned commit, minus the dataset entries."""
    return {
        "use_mix_ln": False,
        "mix_ln_layer": [0, 2],
        "use_energy_embed": False,
        "use_breathiness_embed": False,
        "use_voicing_embed": False,
        "use_tension_embed": False,
        "use_key_shift_embed": True,
        "use_speed_embed": True,
        "augmentation_args": {
            "random_pitch_shifting": {"enabled": True, "range": [-3.0, 3.0], "scale": 0.25},
            "fixed_pitch_shifting": {"enabled": False, "targets": [-5.0, 5.0], "scale": 0.5},
            "random_time_stretching": {"enabled": True, "range": [0.5, 2.0], "scale": 0.75},
        },
        "diffusion_type": "reflow",
        "use_dual_timestep": True,
        "enc_ffn_kernel_size": 3,
        "use_rope": True,
        "rope_interleaved": False,
        "rope_theta": 10000,
        "use_stretch_embed": True,
        "use_variance_scaling": True,
        "use_shallow_diffusion": True,
        "T_start": 0.4,
        "T_start_infer": 0.4,
        "K_step": 300,
        "K_step_infer": 300,
        "hidden_size": 384,
        "backbone_type": "lynxnet2",
        "backbone_args": {"num_channels": 1024, "num_layers": 6, "kernel_size": 31, "dropout_rate": 0.0,
                          "use_conditioner_cache": True, "glu_type": "atanglu"},
        "shallow_diffusion_args": {
            "train_aux_decoder": True, "train_diffusion": True, "val_gt_start": False,
            "aux_decoder_arch": "convnext",
            "aux_decoder_args": {"num_channels": 512, "num_layers": 6, "kernel_size": 7, "dropout_rate": 0.1},
            "aux_decoder_grad": 0.1,
        },
        "lambda_aux_mel_loss": 0.2,
        "main_loss_type": "l1",
        "optimizer_args": {"optimizer_cls": "modules.optimizer.muon.Muon_AdamW", "lr": 0.0006,
                           "muon_args": {"weight_decay": 0.1}, "adamw_args": {"weight_decay": 0.0}},
        "lr_scheduler_args": {"step_size": 5000, "gamma": 0.8},
        "num_valid_plots": 10,
        "val_with_vocoder": True,
        "num_ckpt_keep": 8,
        "permanent_ckpt_interval": 10000,
        "pl_trainer_devices": "auto",
        "pl_trainer_precision": "16-mixed",
    }


def _template_variance() -> dict[str, Any]:
    return {
        "energy_db_min": -96.0, "energy_db_max": -12.0,
        "breathiness_db_min": -96.0, "breathiness_db_max": -20.0,
        "voicing_db_min": -96.0, "voicing_db_max": -12.0,
        "tension_logit_min": -8.0, "tension_logit_max": 8.0,
        "enc_ffn_kernel_size": 3,
        "use_rope": True,
        "rope_interleaved": False,
        "rope_theta": 10000,
        "use_stretch_embed": False,
        "use_variance_scaling": True,
        "hidden_size": 384,
        "dur_prediction_args": {"arch": "resnet", "hidden_size": 256, "dropout": 0.1, "num_layers": 5, "kernel_size": 3,
                                "log_offset": 1.0, "loss_type": "mse", "lambda_pdur_loss": 0.3, "lambda_wdur_loss": 1.0,
                                "lambda_sdur_loss": 3.0},
        "use_melody_encoder": True,
        "melody_encoder_args": {"hidden_size": 128, "enc_layers": 4},
        "use_glide_embed": False,
        "glide_types": ["up", "down"],
        "glide_embed_scale": 11.313708498984760,
        "diffusion_type": "reflow",
        "use_dual_timestep": True,
        "pitch_prediction_args": {"pitd_norm_min": -8.0, "pitd_norm_max": 8.0, "pitd_clip_min": -12.0, "pitd_clip_max": 12.0,
                                  "repeat_bins": 64, "backbone_type": "lynxnet2",
                                  "backbone_args": {"num_layers": 6, "num_channels": 512, "dropout_rate": 0.0,
                                                    "use_conditioner_cache": True, "glu_type": "atanglu"}},
        "variances_prediction_args": {"total_repeat_bins": 72, "backbone_type": "lynxnet2",
                                      "backbone_args": {"num_layers": 6, "num_channels": 384, "dropout_rate": 0.0,
                                                        "use_conditioner_cache": True, "glu_type": "atanglu"}},
        "lambda_dur_loss": 1.0, "lambda_pitch_loss": 1.0, "lambda_var_loss": 1.0,
        "main_loss_type": "l1",
        "optimizer_args": {"optimizer_cls": "modules.optimizer.muon.Muon_AdamW", "lr": 0.0006,
                           "muon_args": {"weight_decay": 0.1}, "adamw_args": {"weight_decay": 0.0}},
        "lr_scheduler_args": {"step_size": 5000, "gamma": 0.8},
        "num_valid_plots": 10,
        "num_ckpt_keep": 8,
        "permanent_ckpt_interval": 10000,
        "pl_trainer_devices": "auto",
        "pl_trainer_precision": "16-mixed",
    }


def _nearest(table: dict[int, tuple[int, int]], gpu_gb: float) -> tuple[int, int]:
    keys = sorted(table)
    chosen = keys[0]
    for key in keys:
        if gpu_gb >= key:
            chosen = key
    return table[chosen]


def is_acoustic(kind: str) -> bool:
    return kind.endswith("acoustic")


def build_overlay(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    """The config overlay for a run; raises ValueError for a spec that can't work."""
    acoustic = is_acoustic(kind)
    datasets = list(config.get("datasets") or [])
    if not datasets:
        raise ValueError("The spec has no datasets")
    dictionaries = dict(config.get("dictionaries") or {})
    if not dictionaries:
        raise ValueError("The spec has no dictionaries (one per language)")
    preset = dict(config.get("preset") or {})
    goal = str(preset.get("goal") or "full")
    if goal not in GOALS:
        raise ValueError(f"Unknown goal {goal}; one of {', '.join(GOALS)}")
    size = str(preset.get("size") or "medium")
    if size not in SIZES:
        raise ValueError(f"Unknown size {size}; one of {', '.join(SIZES)}")
    gpu_gb = float(preset.get("gpu_gb") or 16)

    overlay = _template_acoustic() if acoustic else _template_variance()
    overlay["pe"] = "rmvpe"
    overlay["hnsep"] = "vr"
    overlay["f0_max"] = 1600
    overlay["binarization_args"] = {"num_workers": int(config.get("binarize_workers") or 0)}
    # data loader workers: on Windows every worker is a process that loads torch and the CUDA libraries again, and
    # several of them exhaust the commit limit ("the paging file is too small"); the main process loads the data then
    workers = config.get("loader_workers")
    overlay["ds_workers"] = int(workers) if workers is not None else (0 if os.name == "nt" else 4)
    if overlay["ds_workers"] == 0:
        overlay["dataloader_prefetch_factor"] = None

    # datasets, speakers, languages
    spk_ids: dict[str, int] = {}
    ds_entries = []
    for ds in datasets:
        speaker = str(ds.get("speaker") or "speaker")
        if ds.get("spk_id") is not None:
            spk_id = int(ds["spk_id"])
            spk_ids.setdefault(speaker, spk_id)
        else:
            spk_id = spk_ids.setdefault(speaker, len(set(spk_ids.values())))
        language = str(ds.get("language") or next(iter(dictionaries)))
        if language not in dictionaries:
            raise ValueError(f"Dataset {ds.get('raw_data_dir')} uses language {language}, which has no dictionary")
        entry: dict[str, Any] = {"raw_data_dir": str(ds["raw_data_dir"]), "speaker": speaker, "spk_id": spk_id,
                                 "language": language}
        if ds.get("test_prefixes"):
            entry["test_prefixes"] = list(ds["test_prefixes"])
        ds_entries.append(entry)
    overlay["datasets"] = ds_entries
    overlay["dictionaries"] = {lang: str(path) for lang, path in dictionaries.items()}
    overlay["extra_phonemes"] = list(config.get("extra_phonemes") or [])
    overlay["merged_phoneme_groups"] = [list(g) for g in (config.get("merged_phoneme_groups") or [])]
    num_spk = max(e["spk_id"] for e in ds_entries) + 1
    overlay["use_spk_id"] = num_spk > 1 or bool(config.get("use_spk_id"))
    overlay["num_spk"] = num_spk
    overlay["num_lang"] = len(dictionaries)
    overlay["use_lang_id"] = len(dictionaries) > 1

    # the run's own folder keeps its binaries
    if config.get("binary_data_dir"):
        overlay["binary_data_dir"] = str(config["binary_data_dir"])

    # variance modules
    if not acoustic:
        variance = dict(config.get("variance") or {"dur": True, "pitch": True})
        for key, cfg_key in VARIANCE_KEYS.items():
            overlay[cfg_key] = bool(variance.get(key, False))
        if not any(overlay[k] for k in VARIANCE_KEYS.values()):
            raise ValueError("A variance run needs at least one of dur, pitch, energy, breathiness, voicing, tension")
        if overlay["predict_energy"] and overlay["predict_breathiness"] and overlay["predict_voicing"]:
            raise ValueError("energy, breathiness and voicing shouldn't all be on together (they are coupled)")
    else:
        embeds = dict(config.get("variance") or {})
        for key in ("energy", "breathiness", "voicing", "tension"):
            if key in embeds:
                overlay[f"use_{key}_embed"] = bool(embeds[key])

    # batch and length
    frames, batch = _nearest(BATCH_ACOUSTIC if acoustic else BATCH_VARIANCE, gpu_gb)
    overlay["max_batch_frames"] = frames
    overlay["max_batch_size"] = batch
    max_updates, val_interval, permanent_start = (LENGTH_ACOUSTIC if acoustic else LENGTH_VARIANCE)[size]
    overlay["max_updates"] = max_updates
    overlay["val_check_interval"] = val_interval
    overlay["permanent_ckpt_start"] = permanent_start
    if goal == "check":
        overlay.update({"max_updates": 200, "val_check_interval": 100, "num_valid_plots": 2, "num_ckpt_keep": 2,
                        "permanent_ckpt_start": 0, "permanent_ckpt_interval": 0, "log_interval": 10,
                        "num_sanity_val_steps": 1})
    elif goal == "preview":
        overlay.update({"max_updates": 3000, "val_check_interval": 500, "num_valid_plots": 4, "num_ckpt_keep": 3,
                        "permanent_ckpt_start": 0, "permanent_ckpt_interval": 0})
    if goal in ("finetune", "preview") or config.get("finetune"):
        ft = dict(config.get("finetune") or {})
        if not ft.get("ckpt"):
            raise ValueError("Fine-tuning needs finetune.ckpt (the checkpoint to start from)")
        overlay["finetune_enabled"] = True
        overlay["finetune_ckpt_path"] = str(ft["ckpt"])
        overlay["finetune_ignored_params"] = list(ft.get("ignored_params", [
            "model.fs2.encoder.embed_tokens", "model.fs2.txt_embed", "model.fs2.spk_embed"])
        )
        overlay["finetune_strict_shapes"] = bool(ft.get("strict_shapes", False))
        if goal == "finetune":
            overlay["max_updates"] = min(overlay["max_updates"], 40000 if acoustic else 30000)
        if ft.get("frozen_params"):
            overlay["freezing_enabled"] = True
            overlay["frozen_params"] = list(ft["frozen_params"])

    # the device the run was asked for
    device = str(config.get("device") or "auto")
    if device.startswith("cuda:"):
        overlay["pl_trainer_devices"] = [int(device.split(":", 1)[1])]
    elif device == "cpu":
        overlay["pl_trainer_accelerator"] = "cpu"
        overlay["pl_trainer_precision"] = "32"

    options = config.get("options") or {}
    backbone = options.get("backbone", "lynxnet2")
    if backbone not in ("lynxnet2", "lynxnet", "wavenet"):
        raise ValueError("backbone must be lynxnet2, lynxnet or wavenet")
    targets = [(overlay, 1024, 20, 4)] if acoustic else [
        (overlay["pitch_prediction_args"], 512, 20, 5),
        (overlay["variances_prediction_args"], 384, 10, 4)]
    for target, channels, layers, cycle in targets:
        args = {"num_channels": channels, "num_layers": 6, "dropout_rate": 0.0}
        if acoustic:
            args["kernel_size"] = 31
        if backbone == "wavenet":
            args = {"num_channels": channels // 2, "num_layers": layers, "dilation_cycle_length": cycle}
        elif backbone == "lynxnet":
            args["strong_cond"] = True
        else:
            args.update(use_conditioner_cache=True, glu_type="atanglu")
        target.update(backbone_type=backbone, backbone_args=args)
    for name, key in (("dual_timestep", "use_dual_timestep"), ("stretch", "use_stretch_embed"),
                      ("melody_encoder", "use_melody_encoder"), ("mix_layer_norm", "use_mix_ln")):
        if name in options:
            overlay[key] = bool(options[name])
    if acoustic and "mix_layer_norm" not in options:
        overlay["use_mix_ln"] = num_spk > 1
    if options.get("hnsep", "vr") not in ("vr", "world"):
        raise ValueError("hnsep must be vr or world")
    overlay["hnsep"] = options.get("hnsep", "vr")
    overlay["matmul_precision"] = "medium" if options.get("tensor_acceleration", True) else "highest"
    overlay["binarization_args"]["prefer_ds"] = bool(options.get("prefer_ds", False))
    if acoustic:
        for name, aug, embed in (("pitch_augmentation", "random_pitch_shifting", "use_key_shift_embed"),
                                 ("time_augmentation", "random_time_stretching", "use_speed_embed")):
            if name in options:
                enabled = bool(options[name])
                overlay[embed] = enabled
                overlay["augmentation_args"][aug]["enabled"] = enabled
    overrides = config.get("overrides") or {}
    if overrides:
        _deep_update(overlay, overrides)
    for key in ("max_updates", "val_check_interval", "max_batch_frames", "max_batch_size"):
        if isinstance(overlay.get(key), bool) or not isinstance(overlay.get(key), int) or overlay[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if overlay["val_check_interval"] > overlay["max_updates"]:
        raise ValueError("Validation interval must not exceed total training steps")
    if not acoustic:
        if not any(overlay.get(k) for k in VARIANCE_KEYS.values()):
            raise ValueError("Enable at least one variance prediction")
        if all(overlay.get("predict_" + k) for k in ("energy", "breathiness", "voicing")):
            raise ValueError("energy, breathiness and voicing cannot all be enabled")
    return overlay


def _deep_update(base: dict[str, Any], over: dict[str, Any]) -> None:
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def resolve_assets(config: dict[str, Any]) -> dict[str, str]:
    """Asset model ids (or paths) of a spec, with the catalog's defaults for the missing ones."""
    assets = dict(ASSET_DEFAULTS)
    for key, value in (config.get("assets") or {}).items():
        if value:
            assets[key] = str(value)
    options, overrides = config.get("options") or {}, config.get("overrides") or {}
    if overrides.get("hnsep", options.get("hnsep")) == "world":
        assets.pop("hnsep", None)
    if overrides.get("pe") in {"parselmouth", "harvest"}:
        assets.pop("pe", None)
    return assets

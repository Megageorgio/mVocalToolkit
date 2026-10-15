import pytest

from mvocaltoolkit.training.diffsinger import build_overlay, resolve_assets

BASE = {"datasets": [{"raw_data_dir": "raw", "speaker": "s", "language": "xx"}], "dictionaries": {"xx": "dict"}, "preset": {"goal": "check"}}


def test_effective_compute_controls_preserve_binarization_options_and_force_cpu_fp32():
    options = dict(ds_workers=3, accumulate_grad_batches=4, num_ckpt_keep=2, permanent_ckpt_interval=0,
                   pl_trainer_precision="bf16-mixed", binarization_args=dict(num_workers=2))
    gpu = build_overlay("diffsinger_acoustic", {**BASE, "overrides": options})
    assert gpu["ds_workers"] == 3 and gpu["dataloader_prefetch_factor"] == 2
    assert gpu["accumulate_grad_batches"] == 4 and gpu["pl_trainer_precision"] == "bf16-mixed"
    assert gpu["num_ckpt_keep"] == 2 and gpu["permanent_ckpt_interval"] == 0
    assert gpu["binarization_args"]["num_workers"] == 2 and not gpu["binarization_args"]["prefer_ds"]
    cpu = build_overlay("diffsinger_variance", {**BASE, "device": "cpu", "overrides": {**options, "ds_workers": 0}})
    assert cpu["pl_trainer_precision"] == "32" and cpu["dataloader_prefetch_factor"] is None
    assert options["pl_trainer_precision"] == "bf16-mixed"
    for count in (-1, 0):
        assert build_overlay("diffsinger_acoustic", {**BASE, "overrides":{"num_ckpt_keep":count}})["num_ckpt_keep"] == count


@pytest.mark.parametrize("overrides", [dict(ds_workers=-1), dict(ds_workers=True), dict(accumulate_grad_batches=0),
    dict(accumulate_grad_batches=1.5), dict(num_ckpt_keep=-2), dict(permanent_ckpt_interval=1),
    dict(binarization_args=dict(num_workers=65)), dict(binarization_args=[]), dict(pl_trainer_precision="int8")])
def test_invalid_compute_controls_fail_before_training(overrides):
    with pytest.raises(ValueError):
        build_overlay("diffsinger_acoustic", {**BASE, "overrides": overrides})


def test_backbone_and_augmentations_use_compatible_arguments():
    config = {**BASE, "options": {"backbone": "wavenet", "pitch_augmentation": False, "time_augmentation": False, "hnsep": "world", "prefer_ds": True, "mix_layer_norm": True}}
    result = build_overlay("diffsinger_acoustic", config)
    assert result["backbone_type"] == "wavenet"
    assert result["backbone_args"] == {"num_channels": 512, "num_layers": 20, "dilation_cycle_length": 4}
    assert not result["use_key_shift_embed"] and not result["augmentation_args"]["random_pitch_shifting"]["enabled"]
    assert not result["use_speed_embed"] and result["use_mix_ln"]
    assert result["binarization_args"]["prefer_ds"] and "hnsep" not in resolve_assets(config)
    result = build_overlay("diffsinger_variance", {**BASE, "options": {"backbone": "lynxnet"}})
    assert result["pitch_prediction_args"]["backbone_args"]["strong_cond"]
    assert "glu_type" not in result["pitch_prediction_args"]["backbone_args"]


def test_explicit_finetune_empty_ignore_list_and_final_override_validation():
    result = build_overlay("diffsinger_acoustic", {**BASE, "finetune": {"ckpt": "model.ckpt", "ignored_params": []}})
    assert result["finetune_ignored_params"] == []
    with pytest.raises(ValueError, match="Validation interval"):
        build_overlay("diffsinger_acoustic", {**BASE, "overrides": {"max_updates": 1}})
    with pytest.raises(ValueError, match="cannot all"):
        build_overlay("diffsinger_variance", {**BASE, "overrides": {"predict_energy": True, "predict_breathiness": True, "predict_voicing": True}})


def test_forced_expression_still_completes_missing_duration_and_pitch():
    from mvocaltoolkit.engines.diffsinger.inference_options import prediction_tags
    config = {"predict_dur": True, "predict_pitch": True, "predict_breathiness": True}
    assert prediction_tags(config, [{"ph_seq": "a"}], ["breathiness"]) == ["breathiness", "dur", "pitch"]
    assert prediction_tags(config, [{"ph_dur": "1", "f0_seq": "440", "breathiness": "-50"}], []) == []
    with pytest.raises(ValueError, match="cannot predict"):
        prediction_tags(config, [{}], ["tension"])

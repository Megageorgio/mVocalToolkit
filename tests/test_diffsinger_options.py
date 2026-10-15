import pytest

from mvocaltoolkit.training.diffsinger import build_overlay, resolve_assets

BASE = {"datasets": [{"raw_data_dir": "raw", "speaker": "s", "language": "xx"}], "dictionaries": {"xx": "dict"}, "preset": {"goal": "check"}}


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

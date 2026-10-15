import pytest

from mvocaltoolkit.engines._runtime.mvt_mel import compatibility_errors


@pytest.mark.parametrize("key,a,v", [("num_mel_bins",128,80), ("fft_size",2048,1024), ("win_size",2048,1024),
    ("mel_scale","slaney","htk"), ("mel_fmin",40,20), ("mel_fmax",12000,16000)])
def test_different_mel_contracts_cannot_be_silently_reinterpreted(key, a, v):
    assert any(key in error for error in compatibility_errors({key:a}, {key:v}))


def test_log_base_conversion_and_legacy_unspecified_metadata_are_preserved():
    assert compatibility_errors({"mel_base":"e"}, {"mel_base":10}) == []
    assert compatibility_errors({"sample_rate":44100}, {}) == []
    assert compatibility_errors({"sample_rate":16000,"mel_fmax":None}, {"sample_rate":16000,"mel_fmax":8000}) == []
    assert compatibility_errors({"mel_fmin":0}, {"mel_fmin":1e-7}) == []
    for metadata in ({"sample_rate":True}, {"num_mel_bins":0}, {"mel_fmin":float("nan")}, {"mel_scale":"custom"}, {"mel_base":"2"}):
        assert compatibility_errors(metadata, {})

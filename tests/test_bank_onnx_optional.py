"""Synthetic ONNX contracts only; no pretrained weights, GPU or listening claim."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
yaml = pytest.importorskip("yaml")
from onnx import helper, TensorProto, numpy_helper


def load_worker(monkeypatch):
    runtime = SimpleNamespace(method=lambda: lambda f: f, progress=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "mvt_engine", runtime)
    file = Path(__file__).parents[1] / "src/mvocaltoolkit/engines/bank_render/worker.py"
    spec = importlib.util.spec_from_file_location("bank_render_fixture", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_graph(path, nodes, inputs, outputs, constants=()):
    graph = helper.make_graph(nodes, "synthetic-contract", inputs, outputs, initializer=list(constants))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))


@pytest.mark.parametrize("heads", [False, True])
def test_actual_onnx_sessions_render_a_synthetic_bank_and_reject_unknown_phones(tmp_path, monkeypatch, heads):
    worker = load_worker(monkeypatch)
    bank = tmp_path / "bank"
    bank.mkdir()
    F, I = TensorProto.FLOAT, TensorProto.INT64
    def tensor(name, kind, dims): return helper.make_tensor_value_info(name, kind, dims)
    (bank / "phonemes.json").write_text(json.dumps(["SP", "AP", "a"]))
    (bank / "dsconfig.yaml").write_text(yaml.safe_dump(dict(acoustic="acoustic.onnx", phonemes="phonemes.json", sample_rate=16000, hop_size=160, vocoder="local", use_energy_embed=heads)))
    zero = numpy_helper.from_array(np.array(0, np.float32), "zero")
    save_graph(bank / "acoustic.onnx", [helper.make_node("Mul", ["f0", "zero"], ["mel"])],
        [tensor("tokens", I, [1, "phones"]), tensor("durations", I, [1, "phones"]), tensor("f0", F, [1, "frames"])],
        [tensor("mel", F, [1, "frames"])], [zero])
    if heads:
        for folder_name, component in [("dsdur", "dur"), ("dspitch", "pitch"), ("dsvariance", "variance")]:
            folder = bank / folder_name
            folder.mkdir()
            (folder / "phonemes.json").write_text(json.dumps(["SP", "AP", "a"]))
            (folder / "dsconfig.yaml").write_text(yaml.safe_dump(dict(linguistic="linguistic.onnx", phonemes="phonemes.json", **{component: component + ".onnx"}, predict_energy=component == "variance")))
            save_graph(folder / "linguistic.onnx", [helper.make_node("Cast", ["tokens"], ["encoder_out"], to=F)],
                [tensor("tokens", I, [1, "phones"])], [tensor("encoder_out", F, [1, "phones"])])
            if component == "dur":
                save_graph(folder / "dur.onnx", [helper.make_node("Identity", ["distribution"], ["dur_pred"])],
                    [tensor("encoder_out", F, [1, "phones"])], [tensor("dur_pred", F, [1, 2])],
                    [numpy_helper.from_array(np.array([[1., 3.]], np.float32), "distribution")])
            else:
                key = "pitch" if component == "pitch" else "energy"
                save_graph(folder / (component + ".onnx"), [helper.make_node("Identity", [key], [key + "_pred"])],
                    [tensor(key, F, [1, "frames"]), tensor("retake", TensorProto.BOOL, [1, "frames"] if component == "pitch" else [1, "frames", 1])],
                    [tensor(key + "_pred", F, [1, "frames"])])
    voc = bank / "dsvocoder"
    voc.mkdir()
    (voc / "vocoder.yaml").write_text(yaml.safe_dump(dict(model="vocoder.onnx", sample_rate=16000, hop_size=160)))
    save_graph(voc / "vocoder.onnx", [helper.make_node("Tile", ["mel", "repeat"], ["waveform"])],
        [tensor("mel", F, [1, "frames"]), tensor("f0", F, [1, "frames"])], [tensor("waveform", F, [1, "samples"])],
        [numpy_helper.from_array(np.array([1, 160], np.int64), "repeat")])
    ds = tmp_path / "phrase.ds"
    ds.write_text(json.dumps([dict(ph_seq="a a", ph_dur=".05 .05", ph_num="2", note_seq="C4", note_dur=".1")]))
    result = worker.render(str(bank), str(ds), str(tmp_path / "render"), predict=["dur", "pitch", "variance"] if heads else None)
    samples, rate = sf.read(result["files"][0])
    assert rate == 16000 and len(samples) == 1600 and np.isfinite(samples).all()
    assert (tmp_path / "render/render-report.json").is_file()
    from mvocaltoolkit.training.test_phrases import PhraseRequest, build_phrase
    phrase=build_phrase(PhraseRequest(model_dir=str(bank),mode="manual",phonemes="a | a",notes="C4 E4"),tmp_path/"generated-phrase")
    generated=worker.render(str(bank),phrase["path"],str(tmp_path/"generated-render"))
    samples,rate=sf.read(generated["files"][0])
    assert abs(len(samples)/rate-phrase["seconds"])<.011
    ds.write_text(json.dumps([dict(ph_seq="not-trained", ph_dur=".1", note_seq="C4", note_dur=".1")]))
    with pytest.raises(ValueError, match="Untrained phoneme"):
        worker.render(str(bank), str(ds), str(tmp_path / "invalid"))


def test_audio_padding_and_silence_fades_preserve_the_middle(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mvt_engine", SimpleNamespace(method=lambda: lambda f: f))
    file = Path(__file__).parents[1] / "src/mvocaltoolkit/engines/audio/worker.py"
    spec = importlib.util.spec_from_file_location("audio_fixture", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = tmp_path / "source.wav"
    sf.write(original, np.ones(1000) * .5, 1000, subtype="FLOAT")
    target = tmp_path / "part.wav"
    module.slice_audio(str(original), [dict(start=0, end=1, path=str(target), fade_in=.1, fade_out=.1, pad_before=.1, pad_after=.2)])
    samples, rate = sf.read(target)
    assert rate == 1000 and len(samples) == 1300
    assert np.all(samples[:100] == 0) and np.all(samples[-200:] == 0)
    assert samples[100] == 0 and samples[1099] == 0
    assert np.allclose(samples[250:950], .5)

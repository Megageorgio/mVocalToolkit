import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import make_wav
from mvocaltoolkit.engines.spec import load_engine_specs
from mvocaltoolkit.models.catalog import Catalog
from mvocaltoolkit.server.app import create_app
from mvocaltoolkit.settings import Home, Settings
from mvocaltoolkit.toolkit import Toolkit
from mvocaltoolkit.training.runs import Run, RunInfo, RunSpec, RunStatus
from mvocaltoolkit.training.vocoder import BASE_MODEL, build_overlay, scan_sources


def test_vocoder_presets_keep_mel_compatibility_and_count_real_steps():
    config = build_overlay({"accept_license": True, "preset": {"goal": "full", "gpu_gb": 8}})
    assert config["max_updates"] == 4000 and config["batch_size"] == 2
    assert config["audio_sample_rate"] == 44100 and config["audio_num_mel_bins"] == 128
    assert config["pl_trainer_precision"] == "32-true" and config["ds_workers"] == 0
    with pytest.raises(ValueError, match="accept"):
        build_overlay({})
    for overrides in ({"steps": -2}, {"steps": True}, {"steps": 10, "validation_interval": 11}, {"learning_rate": float("nan")}):
        with pytest.raises(ValueError):
            build_overlay({"accept_license": True, **overrides})


def test_vocoder_check_rejects_upsampled_source_rates_and_deduplicates(tmp_path):
    source = tmp_path / "source"
    first = make_wav(source / "first.wav", rate=44100)
    (source / "duplicate.wav").write_bytes(first.read_bytes())
    make_wav(source / "second.wav", rate=44100, seconds=1.1)
    result = scan_sources({"sources": [{"folder": str(source), "speaker": "singer", "license": "CC BY"}]})
    assert result["files"] == 2 and not any(p["severity"] == "error" for p in result["problems"])
    assert all(r["license"] == "CC BY" for r in result["recordings"])
    make_wav(source / "low.wav", rate=16000)
    result = scan_sources({"sources": [{"folder": str(source)}]})
    assert any("16000" in p["problem"] for p in result["problems"])


def test_full_training_base_is_bundled_and_engine_is_pinned(tmp_path):
    specs = load_engine_specs()
    assert len(specs["singingvocoders"].source.commit) == 40
    catalog = Catalog(Home(tmp_path), Settings(), engines=set(specs))
    catalog.load()
    entry = catalog.get(BASE_MODEL)
    assert entry and entry.engine == "singingvocoders" and entry.source.repo == "openvpi/SingingVocoders"


def test_vocoder_api_resolves_recipe_without_installing_or_training(tmp_path, monkeypatch):
    tk = Toolkit(Home(tmp_path / "home"))
    captured = []
    def start(spec):
        captured.append(spec)
        return Run(RunInfo(id="test", spec=spec, dir=spec.dir))
    monkeypatch.setattr(tk.training, "start", start)
    folder = tmp_path / "audio"
    make_wav(folder / "one.wav", rate=44100)
    make_wav(folder / "two.wav", rate=44100, seconds=1.1)
    config = {"sources": [{"folder": str(folder)}], "accept_license": True}
    with TestClient(create_app(tk)) as client:
        assert client.post("/training/vocoder/check", json=config).json()["files"] == 2
        spec = {"kind": "singingvocoders_vocoder", "config": config}
        review = client.post("/training/overlay", json=spec)
        assert review.status_code == 200 and review.json()["real_steps"] == 100
        result = client.post("/training/runs", json=spec)
        assert result.status_code == 200, result.text
        assert captured[-1].config["data_report"]["files"] == 2
        assert captured[-1].config["assets"]["base"] == BASE_MODEL
        config["accept_license"] = False
        assert client.post("/training/runs", json=spec).status_code == 400


def test_gan_events_use_real_steps_and_native_checkpoint_names(tmp_path, monkeypatch):
    # Exercise the real callback without importing torch or creating any training environment.
    lightning = types.ModuleType("lightning")
    pl = types.ModuleType("lightning.pytorch")
    pl.Callback = object
    lightning.pytorch = pl
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    for name, value in (("lightning", lightning), ("lightning.pytorch", pl), ("torch", torch)):
        monkeypatch.setitem(sys.modules, name, value)
    path = Path(__file__).resolve().parent.parent / "src/mvocaltoolkit/engines/diffsinger/mvt_ds_train.py"
    spec = importlib.util.spec_from_file_location("callback_under_test", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    callback = helper.make_callback(tmp_path, 10, step_divisor=2)
    trainer = types.SimpleNamespace(global_step=20, max_steps=200, current_epoch=1, progress_bar_metrics={"total_loss": 1.0},
                                    optimizers=[], should_stop=False)
    trainer.save_checkpoint = lambda path: Path(path).write_bytes(b"weights")
    callback.on_fit_start(trainer, None)
    (tmp_path / "STOP").touch()
    callback.on_train_batch_end(trainer, None, None, None, 0)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[0]["step"] == 10 and events[0]["max_steps"] == 100
    checkpoint = next(e for e in events if e["event"] == "checkpoint")
    assert checkpoint["step"] == 10 and Path(checkpoint["path"]).name == "model_ckpt_steps_20.ckpt"
    assert trainer.should_stop


def test_vocoder_export_chains_generator_and_onnx_with_provenance(tmp_path, monkeypatch):
    tk = Toolkit(Home(tmp_path / "home"))
    calls = []
    async def call(engine, method, params, **kwargs):
        calls.append((engine, method))
        folder = Path(params["out_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        if method == "export":
            checkpoint = folder / "model.ckpt"
            checkpoint.write_bytes(b"generator")
            return {"checkpoint": str(checkpoint), "files": [str(checkpoint)]}
        assert method == "export_vocoder" and Path(params["checkpoint"]).is_file()
        (folder / "vocoder.onnx").write_bytes(b"onnx")
        return {"files": [str(folder / "vocoder.onnx")]}
    monkeypatch.setattr(tk.engines, "call", call)
    with TestClient(create_app(tk)) as client:
        run = Run(RunInfo(id="voc", spec=RunSpec(kind="singingvocoders_vocoder", config={
            "data_report": {"license": "CC BY-NC-SA 4.0", "files": 3, "recordings": []}}),
            dir=str(tmp_path / "run"), status=RunStatus.done))
        run.save(force=True)
        tk.training.runs[run.id] = run
        response = client.post("/training/runs/voc/export", json={"step": 100})
        assert response.status_code == 200, response.text
        for _ in range(20):
            result = client.get(f"/jobs/{response.json()['id']}", params={"wait": 1}).json()
            if result["status"] in {"done", "failed"}:
                break
        assert result["status"] == "done", result
        assert calls == [("singingvocoders", "export"), ("diffsinger", "export_vocoder")]
        assert json.loads((tmp_path / "run/export/100/dataset-report.json").read_text())["license"] == "CC BY-NC-SA 4.0"

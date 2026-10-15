import importlib.util
import json
import wave
from pathlib import Path

import pytest

from mvocaltoolkit.training.rvc import check, resolve_assets, settings, snapshot


def audio(folder, name="a.wav", seconds=4, rate=44100):
    folder.mkdir(exist_ok=True)
    path = folder / name
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate)
        stream.writeframes(b"\x10\x00" * round(seconds * rate))
    return path


@pytest.mark.parametrize("change", [{"sample_rate": 44100}, {"batch_size": True}, {"epochs": 0},
    {"save_every_epoch": 101}, {"max_steps": -1}, {"chunk_seconds": float("nan")}, {"learning_rate": 0},
    {"overlap_seconds": 1, "chunk_seconds": 1}, {"index_vectors": 100001}, {"pitch_method": "unknown"}, {"prepare_only": "false"}])
def test_rvc_settings_fail_before_install(change):
    with pytest.raises(ValueError):
        settings(change)


def test_rvc_one_voice_hash_manifest_and_duplicate_exclusion(tmp_path):
    folder = tmp_path / "audio"
    source = audio(folder)
    (folder / "duplicate.wav").write_bytes(source.read_bytes())
    config = {"sources": [{"folder": str(folder), "speaker": "one", "license": "CC0"}], "pretrained": False}
    data = check(config)
    assert data["files"] == 1 and any(p["code"] == "duplicate" for p in data["problems"])
    original = source.read_bytes()
    first = snapshot(config, tmp_path / "snapshots")
    second = snapshot(config, tmp_path / "snapshots")
    assert first == second and source.read_bytes() == original
    assert "recordings" not in first["data_report"]
    manifest = json.loads(Path(first["prepared_file"]).read_text(encoding="utf-8"))
    assert manifest["recordings"][0]["sha256"] and manifest["recordings"][0]["license"] == "CC0"
    assert resolve_assets(settings(config)) == {"embedder": "rvc-contentvec"}
    with pytest.raises(ValueError, match="one speaker"):
        check({"sources": [*config["sources"], {"folder": str(folder), "speaker": "two"}]})


def test_rvc_bad_audio_and_missing_folder_block_snapshot(tmp_path):
    source = audio(tmp_path, seconds=.2, rate=8000)
    config = {"sources": [{"folder": str(tmp_path)}]}
    assert any(p["severity"] == "error" for p in check(config)["problems"])
    with pytest.raises(ValueError, match="recording errors"):
        snapshot(config, tmp_path / "snapshots")
    with pytest.raises(ValueError, match="both generator"):
        settings({"base_generator": str(source)})


def test_rvc_api_checks_without_engine_and_does_not_queue_invalid_data(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from mvocaltoolkit.server.app import create_app
    from mvocaltoolkit.settings import Home
    from mvocaltoolkit.toolkit import Toolkit
    toolkit = Toolkit(Home(tmp_path / "home"))
    async def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not install an engine")
    monkeypatch.setattr(toolkit.engines, "call", forbidden)
    source = audio(tmp_path / "audio")
    config = {"sources": [{"folder": str(source.parent), "speaker": "s"}]}
    with TestClient(create_app(toolkit)) as client:
        response = client.post("/training/rvc/check", json=config)
        assert response.status_code == 200 and response.json()["files"] == 1
        assert "recordings" not in response.json()
        response = client.post("/training/runs", json={"kind": "rvc_voice", "device": "mps", "config": config})
        assert response.status_code == 400
        response = client.post("/training/runs", json={"kind": "rvc_voice", "config": {**config, "sample_rate": 44100}})
        assert response.status_code == 400 and not toolkit.training.runs


def test_rvc_adapter_rejects_a_changed_upstream_contract():
    path = Path(__file__).parents[1] / "src/mvocaltoolkit/engines/rvc/adapter.py"
    spec = importlib.util.spec_from_file_location("rvc_adapter", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="Unsupported Applio"):
        module.adapt("print('unexpected trainer')", "C:/run")


def test_rvc_holds_out_whole_distinct_recordings_and_can_disable_it(tmp_path):
    folder=tmp_path/"audio"
    for i in range(5):audio(folder,f"take{i}.wav",seconds=4+i*.01)
    config=dict(sources=[dict(folder=str(folder),speaker="one")],validation_fraction=.2)
    first=check(config);second=check(config)
    assert [r["validation"] for r in first["recordings"]]==[r["validation"] for r in second["recordings"]]
    assert sum(r["validation"] for r in first["recordings"])==1
    assert not any(r["validation"] for r in check({**config,"validation_fraction":0})["recordings"])

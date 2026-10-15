import pytest

from mvocaltoolkit.training.g2p import prepare_dictionary, report


def test_word_disjoint_split_keeps_alternate_pronunciations_and_normalization(tmp_path):
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("Cat\tk a t\ncat\tk ae t\ncar\tk a r\nbar\tb a r\nbat\tb a t\n", encoding="utf-8")
    config = {"dictionary": str(dictionary), "epochs": 2, "batch_size": 2}
    first, second = prepare_dictionary(config), prepare_dictionary(config)
    assert first == second
    train = {w for w, _ in first["train"]}
    validation = {w for w, _ in first["validation"]}
    assert not train & validation and train | validation == {"cat", "car", "bar", "bat"}
    assert len([v for w, v in first["entries"] if w == "cat"]) == 2
    assert first["max_updates"] > 0 and first["warnings"]
    assert "train" not in report(config)
    dictionary.write_text("cat\ta\nCAT\tb\n")
    with pytest.raises(ValueError, match="two distinct"):
        prepare_dictionary(config)


@pytest.mark.parametrize("change", [{"learning_rate": float("nan")}, {"hidden_size": 3}, {"epochs": 0}, {"validation_fraction": .99}, {"batch_size": 0}])
def test_invalid_training_knobs_fail_before_engine_install(tmp_path, change):
    path = tmp_path / "dict.txt"
    path.write_text("a\ta\nb\tb\n")
    with pytest.raises(ValueError):
        prepare_dictionary({"dictionary": str(path), **change})


def test_finetuning_preserves_token_ids_and_expands_only_when_requested(tmp_path):
    import yaml
    from mvocaltoolkit.training.g2p import snapshot
    path = tmp_path / "dict.txt"
    path.write_text("cat\tk a t\ncar\tk a r\nbar\tb a r\n")
    original = prepare_dictionary({"dictionary": str(path)})
    base = tmp_path / "base"
    base.mkdir()
    (base / "cfg.yaml").write_text(yaml.safe_dump(original["model"]))
    (base / "model.ptsd").write_bytes(b"weights")
    path.write_text(path.read_text() + "qaq\tq a q\n")
    config = {"dictionary": str(path), "base_model": str(base)}
    with pytest.raises(ValueError, match="lacks"):
        prepare_dictionary(config)
    data = prepare_dictionary({**config, "expand_vocabulary": True})
    assert data["model"]["encoder"]["graphemes"][:-1] == original["model"]["encoder"]["graphemes"]
    assert data["added_graphemes"] == data["added_phonemes"] == ["q"]
    first = snapshot({**config, "expand_vocabulary": True}, tmp_path / "snapshots")
    second = snapshot({**config, "expand_vocabulary": True}, tmp_path / "snapshots")
    assert first == second and "train" not in first["data_report"] and "entries" not in first["data_report"]


def test_prediction_api_rejects_busy_device_and_bad_checkpoint_before_loading(tmp_path, monkeypatch):
    from conftest import FAKE_ENGINES
    from fastapi.testclient import TestClient
    from mvocaltoolkit.server.app import create_app
    from mvocaltoolkit.settings import Home
    from mvocaltoolkit.toolkit import Toolkit
    from mvocaltoolkit.training.runs import Run, RunInfo, RunSpec, RunStatus
    tk = Toolkit(Home(tmp_path / "home"), engine_dirs=[FAKE_ENGINES])
    calls = []
    async def call(engine, method, params, **kwargs):
        calls.append(params)
        return {"items": [{"word": params["words"][0], "phones": ["a"]}]}
    monkeypatch.setattr(tk.engines, "call", call)
    with TestClient(create_app(tk)) as client:
        run = Run(RunInfo(id="g2p", spec=RunSpec(kind="g2p_rnnt"), dir=str(tmp_path / "run"), status=RunStatus.done))
        tk.training.runs[run.id] = run
        run.save(force=True)
        url = "/training/runs/g2p/predict"
        assert client.post(url, json={"words": ["a"], "step": -1}).status_code == 400
        tk.training._inference = True
        assert client.post(url, json={"words": ["a"]}).status_code == 409
        tk.training._inference = False
        assert not calls
        response = client.post(url, json={"words": ["a"], "step": 10})
        assert response.status_code == 200 and response.json()["items"][0]["phones"] == ["a"]
        assert calls[0]["step"] == 10 and not tk.training._inference

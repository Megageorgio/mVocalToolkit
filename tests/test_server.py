"""API tests with fake engines (separate processes speaking the real worker protocol)."""

import time

from conftest import FAKE_ENGINES, make_sofa_model, make_wav
from fastapi.testclient import TestClient

from mvocaltoolkit.server.app import create_app
from mvocaltoolkit.settings import Home
from mvocaltoolkit.toolkit import Toolkit


def _client(tmp_path):
    home = Home(tmp_path / "home")
    tk = Toolkit(home, engine_dirs=[FAKE_ENGINES])
    return TestClient(create_app(tk))


def _wait(client, job_id, status=("done", "failed", "cancelled"), timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = client.get(f"/jobs/{job_id}", params={"wait": 2}).json()
        if info["status"] in status:
            return info
    raise AssertionError(f"job {job_id} timed out: {info}")


def _import_model(client, tmp_path):
    folder = make_sofa_model(tmp_path / "models_src" / "my_en_model")
    response = client.post("/models/import", json={"engine": "sofa", "path": str(folder), "id": "test-en",
                                                   "languages": ["en"], "text_frontend": "en"})
    assert response.status_code == 200, response.text
    return "test-en"


def test_health_and_engines(tmp_path):
    with _client(tmp_path) as client:
        health = client.get("/health").json()
        assert health["ok"] is True
        engines = {e["name"]: e for e in client.get("/engines").json()}
        assert engines["sofa"]["installed"] is True  # fake engine runs with the system interpreter
        assert "game" in engines and engines["game"]["installed"] is False
        info = client.get("/engines/sofa/info").json()
        assert "align" in info["methods"]


def test_align_folder_with_transcription(tmp_path):
    corpus = tmp_path / "corpus"
    make_wav(corpus / "a.wav")
    (corpus / "a.txt").write_text("Hello world!", encoding="utf-8")
    make_wav(corpus / "sub" / "b.wav")  # no text: transcribed by the fake recognizer ("Hello, World!")
    make_wav(corpus / "c.wav")
    (corpus / "c.txt").write_text("hello unknownword", encoding="utf-8")
    with _client(tmp_path) as client:
        model = _import_model(client, tmp_path)
        job = client.post("/align", json={
            "input": {"folder": str(corpus)},
            "model": model,
            "output": {"formats": ["htk", "textgrid", "ds_csv"], "dir": str(tmp_path / "out")},
        }).json()
        info = _wait(client, job["id"])
        assert info["status"] == "done", info
        items = {i["name"]: i for i in info["result"]["items"]}
        assert items["a"]["ok"] and items["a"]["tokens"] == ["hello", "world"]
        assert items["b"]["ok"] and items["b"]["text"] == "Hello, World!"
        assert not items["c"]["ok"] and items["c"]["unknown_words"] == ["unknownword"]
        assert (tmp_path / "out" / "htk" / "a.lab").exists()
        assert (tmp_path / "out" / "sub" / "textgrid" / "b.TextGrid").exists()
        csv_text = (tmp_path / "out" / "transcriptions.csv").read_text(encoding="utf-8")
        assert "hh ah l ow" in csv_text
        assert [e["item"] for e in info["item_errors"]] == ["c"]


def test_review_transcription_pause_resume(tmp_path):
    make_wav(tmp_path / "x.wav")
    with _client(tmp_path) as client:
        model = _import_model(client, tmp_path)
        job = client.post("/pipelines/label", json={
            "input": {"items": [{"path": str(tmp_path / "x.wav")}]},
            "model": model,
            "review_transcription": True,
            "output": {"formats": ["json"], "dir": str(tmp_path / "o"), "layout": "beside"},
        }).json()
        info = _wait(client, job["id"], status=("paused", "failed"))
        assert info["status"] == "paused", info
        assert info["pause_data"]["items"][0]["text"] == "Hello, World!"
        client.post(f"/jobs/{job['id']}/resume", json={"items": [{"name": "x", "text": "world"}]})
        info = _wait(client, job["id"])
        assert info["status"] == "done", info
        item = info["result"]["items"][0]
        assert item["tokens"] == ["world"]
        assert [p["text"] for p in item["label"]["tiers"]["phones"]] == ["SP", "w", "er", "l", "d", "SP"]
        assert (tmp_path / "o" / "x.json").exists()


def test_phonemes_input_and_engine_frontend(tmp_path):
    make_wav(tmp_path / "p.wav")
    with _client(tmp_path) as client:
        model = _import_model(client, tmp_path)
        job = client.post("/align", json={
            "input": {"items": [{"path": str(tmp_path / "p.wav"), "phonemes": ["a", "b"]},
                                {"path": str(tmp_path / "p.wav"), "name": "ja", "text": "ab", "language": "ja"}]},
            "model": model, "skip_unknown_words": True,
            "output": {"formats": ["htk"], "dir": str(tmp_path / "o")},
        }).json()
        info = _wait(client, job["id"])
        items = info["result"]["items"]
        assert [p["text"] for p in items[0]["label"]["tiers"]["phones"]][1:3] == ["a", "b"]
        # the fake textfront engine splits into characters
        assert items[1]["tokens"] == ["a", "b"]


def test_text_and_convert(tmp_path):
    with _client(tmp_path) as client:
        model = _import_model(client, tmp_path)
        result = client.post("/text/g2p", json={"texts": ["Hello, nope!"], "model": model}).json()
        item = result["items"][0]
        assert item["tokens"] == ["hello", "nope"]
        assert item["unknown_words"] == ["nope"]
        assert item["phonemes"][0] == ["hh", "ah", "l", "ow"] and item["phonemes"][1] == ["x"]
        converted = client.post("/convert", json={"content": "0 1000000 a\n", "from_format": "htk",
                                                  "to_format": "audacity"}).json()
        assert converted["content"].startswith("0.000000\t0.100000\ta")


def test_upload_and_websocket(tmp_path):
    wav = make_wav(tmp_path / "up.wav")
    with _client(tmp_path) as client:
        model = _import_model(client, tmp_path)
        with open(wav, "rb") as f:
            uploaded = client.post("/files", files={"file": ("up.wav", f, "audio/wav")}).json()
        job = client.post("/align", json={
            "input": {"items": [{"file_id": uploaded["file_id"], "text": "hello"}]},
            "model": model, "output": {"formats": ["htk"]},
        }).json()
        events = []
        with client.websocket_connect(f"/jobs/{job['id']}/events") as ws:
            while True:
                event = ws.receive_json()
                events.append(event["event"])
                if event["event"] == "finished" or (event["event"] == "state" and event["job"]["status"] == "done"):
                    break
        info = _wait(client, job["id"])
        path = info["result"]["items"][0]["files"]["htk"]
        assert path.startswith(str(tmp_path / "home" / "outputs"))
        assert client.get("/files/download", params={"path": path}).text.endswith("SP\n")


def test_token_required_when_enabled(tmp_path):
    home = Home(tmp_path / "home")
    settings = home.load_settings()
    settings.require_token_local = True
    settings.token = "secret"
    with TestClient(create_app(Toolkit(home, settings, engine_dirs=[FAKE_ENGINES]))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/engines").status_code == 401
        assert client.get("/engines", headers={"Authorization": "Bearer secret"}).status_code == 200

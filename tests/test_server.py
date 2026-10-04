"""API tests with fake engines (separate processes speaking the real worker protocol)."""

import time

from conftest import FAKE_ENGINES, make_hubertfa_model, make_sofa_model, make_wav
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


def test_languages_and_models_by_task(tmp_path):
    with _client(tmp_path) as client:
        _import_model(client, tmp_path)
        folder = make_hubertfa_model(tmp_path / "hfa_src" / "hfa")
        assert client.post("/models/import", json={"engine": "hubertfa", "path": str(folder), "id": "hfa"}).status_code == 200
        data = client.get("/languages", params={"task": "align"}).json()
        by_code = {lang["code"]: lang for lang in data["languages"]}
        assert by_code["ru"]["native_name"] == "Русский"
        assert "sofa-ru-hhskt-v0.0.1" in [m["id"] for m in by_code["ru"]["models"]]
        assert {"test-en", "hfa"} <= {m["id"] for m in by_code["en"]["models"]}
        assert "hfa" in [m["id"] for m in by_code["ja"]["models"]]
        assert "whisper-large-v3-turbo" not in str(data)  # transcription models are another task
        pitch = {m["id"] for m in client.get("/models", params={"task": "pitch"}).json()}
        assert pitch == {"rmvpe", "fcpe", "parselmouth"}
        separate = client.get("/languages", params={"task": "separate"}).json()["languages"]
        assert [lang["code"] for lang in separate] == ["*"]
        tasks = {t["task"]: t for t in client.get("/tasks").json()}
        assert tasks["align"]["language_specific"] and not tasks["pitch"]["language_specific"]
        assert tasks["align"]["engines"] == ["hubertfa", "sofa"]


def test_align_with_hubertfa_model(tmp_path):
    make_wav(tmp_path / "h.wav")
    with _client(tmp_path) as client:
        folder = make_hubertfa_model(tmp_path / "hfa_src" / "hfa")
        client.post("/models/import", json={"engine": "hubertfa", "path": str(folder), "id": "hfa"})
        job = client.post("/align", json={
            "input": {"items": [{"path": str(tmp_path / "h.wav"), "text": "Hello world"}]},
            "model": "hfa", "language": "en",
            "output": {"formats": ["htk"], "dir": str(tmp_path / "o")},
        }).json()
        info = _wait(client, job["id"])
        assert info["status"] == "done", info
        assert info["result"]["engine"] == "hubertfa"
        phones = [p["text"] for p in info["result"]["items"][0]["label"]["tiers"]["phones"]]
        assert phones == ["AP", "hh", "ah", "l", "ow", "w", "er", "l", "d", "SP"]
        g2p = client.post("/text/g2p", json={"texts": ["world, nope"], "model": "hfa", "language": "en"}).json()
        item = g2p["items"][0]
        assert item["phonemes"][0] == ["w", "er", "l", "d"] and item["unknown_words"] == ["nope"]
        # the multilingual model picks the dictionary of the requested language
        job = client.post("/align", json={
            "input": {"items": [{"path": str(tmp_path / "h.wav"), "words": ["ka"]}]}, "model": "hfa",
            "language": "ja", "non_lexical_phonemes": [], "output": {"formats": ["htk"], "dir": str(tmp_path / "o")},
        }).json()
        info = _wait(client, job["id"])
        phones = [p["text"] for p in info["result"]["items"][0]["label"]["tiers"]["phones"]]
        assert phones == ["SP", "k", "a", "SP"]


def test_separate_and_pitch(tmp_path):
    make_wav(tmp_path / "song" / "s.wav")
    with _client(tmp_path) as client:
        job = client.post("/separate", json={
            "input": {"folder": str(tmp_path / "song")}, "model": "separation-vocals-bs-roformer",
            "stems": ["vocals"], "output_dir": str(tmp_path / "stems"),
        }).json()
        info = _wait(client, job["id"])
        assert info["status"] == "done", info
        files = info["result"]["items"][0]["files"]
        assert list(files) == ["vocals"] and (tmp_path / "stems" / "s_vocals.wav").exists()
        assert client.get("/models/installed").json()[0]["id"] == "separation-vocals-bs-roformer"

        job = client.post("/pitch", json={
            "input": {"items": [{"path": str(tmp_path / "song" / "s.wav")}]}, "model": "fcpe",
            "output_formats": ["csv", "json"],
        }).json()
        info = _wait(client, job["id"])
        assert info["status"] == "done", info
        item = info["result"]["items"][0]
        assert info["result"]["method"] == "fcpe"
        assert item["data"]["f0"][:2] == [0.0, 220.0] and item["data"]["hop"] == 0.01
        csv_lines = (tmp_path / "song" / "s.f0.csv").read_text().splitlines()
        assert csv_lines[0] == "time,f0" and csv_lines[2] == "0.0100,220.000"


def test_extra_words_and_fix_labels(tmp_path):
    make_wav(tmp_path / "e.wav")
    (tmp_path / "e.txt").write_text("hello bakery", encoding="utf-8")
    with _client(tmp_path) as client:
        model = _import_model(client, tmp_path)
        check = client.post("/text/validate", json={"texts": ["hello bakery"], "model": model,
                                                     "extra_words": {"bakery": ["b", "ey", "k", "er", "iy"]}}).json()
        assert check["items"][0]["unknown_words"] == []
        job = client.post("/align", json={
            "input": {"items": [{"path": str(tmp_path / "e.wav")}]}, "model": model,
            "extra_words": {"bakery": ["b", "ey", "k", "er", "iy"]},
            "output": {"formats": ["htk"], "dir": str(tmp_path / "labels"), "layout": "beside"},
        }).json()
        info = _wait(client, job["id"])
        assert info["status"] == "done", info
        lab = tmp_path / "labels" / "e.lab"
        assert "ey" in lab.read_text()
        # duplicate phonemes are merged by the fix, the original goes to _backup
        lab.write_text("0 1000000 a\n1000000 2000000 a\n2000000 3000000 b\n", encoding="utf-8")
        result = client.post("/labels/fix", json={"folder": str(tmp_path / "labels"),
                                                  "rule_sets": ["merge_duplicates"]}).json()
        assert result["changed"] == 1 and result["backup"]
        assert [l.split()[2] for l in lab.read_text().splitlines()] == ["a", "b"]
        backups = list((tmp_path / "labels" / "_backup").rglob("e.lab"))
        assert len(backups) == 1 and backups[0].read_text().count("a") == 2
        # the backup folder is not processed again
        again = client.post("/labels/fix", json={"folder": str(tmp_path / "labels"),
                                                 "rule_sets": ["merge_duplicates"]}).json()
        assert again["changed"] == 0 and len(again["files"]) == 1

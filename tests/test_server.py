"""API tests with fake engines (separate processes speaking the real worker protocol)."""

import time

from conftest import FAKE_ENGINES, make_hubertfa_model, make_sofa_model, make_tifa_model, make_wav
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
        assert tasks["align"]["engines"] == ["hubertfa", "sofa", "tifa"]


def test_align_with_tifa_model(tmp_path):
    make_wav(tmp_path / "t.wav")
    with _client(tmp_path) as client:
        folder = make_tifa_model(tmp_path / "tifa_src" / "tifa")
        assert client.post("/models/import", json={"engine": "tifa", "path": str(folder), "id": "tf"}).status_code == 200
        job = client.post("/align", json={
            "input": {"items": [{"path": str(tmp_path / "t.wav"), "text": "Ab, cd!"}]},
            "model": "tf", "language": "zh", "extra_languages": ["en"],
            "output": {"formats": ["textgrid"], "dir": str(tmp_path / "o")},
        }).json()
        info = _wait(client, job["id"])
        assert info["status"] == "done", info
        item = info["result"]["items"][0]
        # the text reaches TIFA as it is (it does its own G2P), not through our text frontend
        phones = [p["text"] for p in item["label"]["tiers"]["phones"]]
        assert phones == ["SP", "A", "b", ",", "c", "d", "!", "SP"]
        assert item["label"]["tiers"]["texts"][0]["text"] == "T"
        assert item["data"]["diagnosis"]["agreement"] == 0.9
        assert item["data"]["diagnosis"]["extra_languages"] == ["en"]
        assert (tmp_path / "o" / "textgrid" / "t.TextGrid").exists()
        g2p = client.post("/text/g2p", json={"texts": ["to read"], "model": "tf", "language": "en"}).json()
        entry = g2p["items"][0]
        assert entry["tokens"] == ["to", "read"] and entry["phonemes"][0] == ["t", "o"]
        assert entry["candidates"] == {"read": [["r", "e", "a", "d"], ["x"]]}


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


def test_engine_crash_report_has_its_output(tmp_path):
    """A worker that dies mid-call: the error has its exit code and its last output, which also goes to a log."""
    import asyncio  # noqa: PLC0415

    from mvocaltoolkit.engines.manager import EngineManager  # noqa: PLC0415
    from mvocaltoolkit.settings import Home, Settings  # noqa: PLC0415

    home = Home(tmp_path / "home")
    manager = EngineManager(home, Settings(), extra_dirs=[FAKE_ENGINES])

    async def run():
        try:
            await manager.call("crasher", "boom", {})
        except RuntimeError as e:
            return str(e)
        finally:
            await manager.stop_all()
        return ""

    message = asyncio.run(run())
    assert "exit code 3" in message
    assert "no encoder weights" in message
    log = home.logs / "engines" / "crasher.log"
    assert "no encoder weights" in log.read_text(encoding="utf-8")


def _refiner_model(client, tmp_path):
    folder = tmp_path / "refiner_src" / "ref"
    folder.mkdir(parents=True)
    (folder / "config.yaml").write_text("frontend:\n  sample_rate: 16000\n", encoding="utf-8")
    (folder / "phonemes.txt").write_text("a\nSP\n", encoding="utf-8")
    (folder / "model.pt").write_bytes(b"\0" * 16)
    response = client.post("/models/import", json={"engine": "refiner", "path": str(folder), "id": "ref"})
    assert response.status_code == 200, response.text
    return "ref"


def test_align_with_refinement(tmp_path):
    make_wav(tmp_path / "r.wav")
    with _client(tmp_path) as client:
        model = _import_model(client, tmp_path)
        refiner = _refiner_model(client, tmp_path)
        body = {"input": {"items": [{"path": str(tmp_path / "r.wav"), "text": "hello world"}]}, "model": model,
                "output": {"formats": [], "return_labels": True}}
        plain = _wait(client, client.post("/align", json=body).json()["id"])["result"]["items"][0]
        info = _wait(client, client.post("/align", json={**body, "refine": {"model": refiner, "mode": "safe"}}).json()["id"])
        assert info["status"] == "done", info
        item = info["result"]["items"][0]
        before = plain["label"]["tiers"]["phones"]
        after = item["label"]["tiers"]["phones"]
        assert [p["text"] for p in after] == [p["text"] for p in before]
        assert abs(after[1]["start"] - (before[1]["start"] + 0.01)) < 1e-6
        # a word that started on a moved phone boundary moves with it
        assert abs(item["label"]["tiers"]["words"][0]["start"] - after[1]["start"]) < 1e-6
        assert item["data"]["refine"]["mode"] == "safe"


def test_refine_ready_labels(tmp_path):
    make_wav(tmp_path / "s.wav")
    (tmp_path / "s.lab").write_text("0 2000000 SP\n2000000 5000000 a\n5000000 10000000 SP\n", encoding="utf-8")
    with _client(tmp_path) as client:
        refiner = _refiner_model(client, tmp_path)
        job = client.post("/refine", json={"input": {"items": [{"path": str(tmp_path / "s.wav")}]}, "model": refiner,
                                           "output": {"formats": ["htk"], "dir": str(tmp_path / "o"), "layout": "beside"}}).json()
        info = _wait(client, job["id"])
        assert info["status"] == "done", info
        phones = info["result"]["items"][0]["label"]["tiers"]["phones"]
        assert abs(phones[1]["start"] - 0.21) < 1e-6 and abs(phones[2]["start"] - 0.51) < 1e-6
        assert (tmp_path / "o" / "s.lab").read_text(encoding="utf-8").splitlines()[1].startswith("2100000 ")
        # given segments instead of a file
        job = client.post("/refine", json={"input": {"items": [{"path": str(tmp_path / "s.wav"),
                                                                "segments": [[0, 0.3, "SP"], [0.3, 1.0, "a"]]}]},
                                           "model": refiner, "output": {"formats": []}}).json()
        phones = _wait(client, job["id"])["result"]["items"][0]["label"]["tiers"]["phones"]
        assert abs(phones[1]["start"] - 0.31) < 1e-6


def test_storage_usage_and_cleanup(tmp_path):
    import os  # noqa: PLC0415

    wav = make_wav(tmp_path / "up.wav")
    with _client(tmp_path) as client:
        model = _import_model(client, tmp_path)
        with open(wav, "rb") as f:
            uploaded = client.post("/files", files={"file": ("up.wav", f, "audio/wav")}).json()
        job = client.post("/align", json={
            "input": {"items": [{"file_id": uploaded["file_id"], "text": "hello"}]},
            "model": model, "output": {"formats": ["htk"]},
        }).json()
        info = _wait(client, job["id"])
        assert info["status"] == "done" and "detail" in info
        usage = client.get("/storage").json()
        assert [m["id"] for m in usage["models"]] == [model] and usage["models"][0]["bytes"] > 0
        assert usage["temporary"]["uploads"] > 0 and usage["temporary"]["outputs"] > 0
        assert usage["keep_files_days"] == 14
        # recent leftovers stay when only old ones are asked for
        assert client.post("/storage/cleanup", json={"older_than_days": 7}).json()["removed"] == 0
        home = tmp_path / "home"
        old = time.time() - 30 * 86400
        for entry in (home / "uploads").iterdir():
            for root, _dirs, files in os.walk(entry):
                for name in files:
                    os.utime(os.path.join(root, name), (old, old))
            os.utime(entry, (old, old))
        result = client.post("/storage/cleanup", json={"older_than_days": 7, "parts": ["uploads"]}).json()
        assert result["removed"] == 1 and result["freed"] > 0
        assert not any((home / "uploads").iterdir()) and any((home / "outputs").iterdir())
        client.post("/storage/cleanup", json={})
        assert not any((home / "outputs").iterdir()) and not any((home / "jobs").iterdir())
        assert (home / "models" / model).is_dir()


def test_own_words_and_borrowed_g2p(tmp_path):
    make_wav(tmp_path / "w.wav")
    with _client(tmp_path) as client:
        folder = make_hubertfa_model(tmp_path / "hfa_src" / "hfa")
        (folder / "dictionaries" / "en.txt").write_text("hello\thh ah l ow\nex\tx\n", encoding="utf-8")
        client.post("/models/import", json={"engine": "hubertfa", "path": str(folder), "id": "hfa"})

        def align(text):
            job = client.post("/align", json={
                "input": {"items": [{"path": str(tmp_path / "w.wav"), "text": text}]}, "model": "hfa",
                "language": "en", "non_lexical_phonemes": [], "output": {"formats": [], "return_labels": True},
            }).json()
            return _wait(client, job["id"])

        # HubertFA has no G2P: a word missing in the dictionary fails the file
        check = client.post("/text/validate", json={"texts": ["hello nope"], "model": "hfa", "language": "en"}).json()
        assert check["items"][0]["unknown_words"] == ["nope"]
        assert align("hello nope")["result"]["items"][0]["ok"] is False
        # an installed English SOFA model lends its G2P (the fake one spells every word "x", a phoneme hfa knows)
        _import_model(client, tmp_path)
        g2p = client.post("/text/g2p", json={"texts": ["hello nope"], "model": "hfa", "language": "en"}).json()
        assert g2p["items"][0]["phonemes"][1] == ["x"] and g2p["items"][0]["guessed"] == {"nope": ["x"]}
        info = align("hello nope")
        assert info["status"] == "done", info
        assert [p["text"] for p in info["result"]["items"][0]["label"]["tiers"]["phones"]] == ["SP", "hh", "ah", "l", "ow", "x", "SP"]
        # own words win over guesses and stay with the model
        assert client.put("/models/hfa/words", json={"nope": ["hh", "ow"], "": ["x"]}).json() == {"nope": ["hh", "ow"]}
        assert client.get("/models/hfa/words").json() == {"nope": ["hh", "ow"]}
        check = client.post("/text/validate", json={"texts": ["hello nope"], "model": "hfa", "language": "en"}).json()
        assert check["items"][0]["unknown_words"] == []
        phones = [p["text"] for p in align("hello nope")["result"]["items"][0]["label"]["tiers"]["phones"]]
        assert phones == ["SP", "hh", "ah", "l", "ow", "hh", "ow", "SP"]
        assert client.put("/models/hfa/words", json={}).json() == {}
        assert client.get("/models/hfa/words").json() == {}


def test_device_change_restarts_idle_engines(tmp_path):
    with _client(tmp_path) as client:
        model = _import_model(client, tmp_path)
        client.post("/text/g2p", json={"texts": ["hello nope"], "model": model})  # starts the sofa engine
        assert {e["name"]: e for e in client.get("/engines").json()}["sofa"]["running"] is True
        assert client.post("/settings", json={"device": "cpu"}).json()["device"] == "cpu"
        assert {e["name"]: e for e in client.get("/engines").json()}["sofa"]["running"] is False

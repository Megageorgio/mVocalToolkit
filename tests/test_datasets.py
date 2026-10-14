import asyncio
import http.server
import json
import threading
import zipfile
from functools import partial

from conftest import FAKE_ENGINES
from fastapi.testclient import TestClient

from mvocaltoolkit.datasets import DatasetStore, LicenseNotAccepted, ManualDownload
from mvocaltoolkit.models.catalog import Catalog
from mvocaltoolkit.models.store import ModelStore
from mvocaltoolkit.server.app import create_app
from mvocaltoolkit.settings import Home
from mvocaltoolkit.toolkit import Toolkit


def _raises(error, func):
    try:
        func()
    except error:
        return
    raise AssertionError(f"{error.__name__} not raised")


def _license(**allowed):
    return {"name": "CC BY-NC-SA 4.0", "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "summary": "Non-commercial, share alike.", "attribution": "Test dataset by someone", **allowed}


def _serve(tmp_path):
    www = tmp_path / "www"
    www.mkdir()
    with zipfile.ZipFile(www / "data.zip", "w") as z:
        z.writestr("data/wavs/a.wav", b"RIFF")
        z.writestr("data/wavs/a.lab", "0 100 a\n")
    (www / "extra.lab").write_text("0 100 b\n", encoding="utf-8")
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(http.server.SimpleHTTPRequestHandler, directory=str(www)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _catalog(home, base):
    (home.catalogs / "datasets.json").write_text(json.dumps({"name": "test", "models": [], "datasets": [
        {"id": "ok-set", "name": "OK set", "languages": ["ja"], "hours": 1.5,
         "license": _license(training=True, publish_models=True),
         "sources": [{"type": "url", "url": f"{base}/data.zip"},
                     {"type": "url", "url": f"{base}/extra.lab", "into": "labels", "part": "labels"}]},
        {"id": "no-training", "name": "Listening only", "languages": ["ja"],
         "license": _license(training=False)},
        {"id": "no-license", "name": "No license", "languages": ["ja"]},
        {"id": "by-hand", "name": "Form first", "languages": ["ko"], "license": _license(training=True),
         "manual_download": "https://example.org/form"},
    ]}), encoding="utf-8")


def _store(tmp_path, base="http://127.0.0.1:9"):
    home = Home(tmp_path / "home")
    settings = home.load_settings()
    _catalog(home, base)
    catalog = Catalog(home, settings)
    catalog.load()
    return home, DatasetStore(ModelStore(home, settings, catalog))


def test_only_datasets_that_allow_training_are_offered(tmp_path):
    _home, store = _store(tmp_path)
    ids = [d["id"] for d in store.listing()]
    assert sorted(ids) == ["by-hand", "ok-set"]
    assert [d["id"] for d in store.listing("ko")] == ["by-hand"]
    assert store.listing("ja")[0]["license"]["name"] == "CC BY-NC-SA 4.0"


def test_download_needs_the_license_accepted_and_keeps_it(tmp_path):
    server, base = _serve(tmp_path)
    try:
        home, store = _store(tmp_path, base)
        entry = store.catalog.datasets["ok-set"]
        _raises(LicenseNotAccepted, lambda: asyncio.run(store.download(entry, accept_license=False)))
        assert not store.is_installed("ok-set")
        path = asyncio.run(store.download(entry, accept_license=True))
        assert (path / "wavs" / "a.wav").is_file() and (path / "labels" / "extra.lab").is_file()
        text = (path / "LICENSE.txt").read_text(encoding="utf-8")
        assert "CC BY-NC-SA 4.0" in text and "Commercial use: no" in text and "Test dataset by someone" in text
        assert {d["id"]: d["installed"] for d in store.listing()} == {"ok-set": True, "by-hand": False}
        _raises(ManualDownload, lambda: asyncio.run(store.download(store.catalog.datasets["by-hand"], accept_license=True)))
        assert store.remove("ok-set") and not store.is_installed("ok-set")
    finally:
        server.shutdown()


def test_dataset_api(tmp_path):
    server, base = _serve(tmp_path)
    try:
        home = Home(tmp_path / "home")
        _catalog(home, base)
        with TestClient(create_app(Toolkit(home, engine_dirs=[FAKE_ENGINES]))) as client:
            listed = {d["id"]: d for d in client.get("/datasets").json()}
            assert set(listed) == {"ok-set", "by-hand"}
            assert "Training models: yes" in client.get("/datasets/ok-set/license").json()["text"]
            refused = client.post("/datasets/ok-set/download", json={})
            assert refused.status_code == 403 and refused.json()["detail"]["error"] == "license_not_accepted"
            assert client.post("/datasets/by-hand/download", json={"accept_license": True}).status_code == 409
            job = client.post("/datasets/ok-set/download", json={"accept_license": True}).json()
            deadline = 0
            while True:
                info = client.get(f"/jobs/{job['id']}", params={"wait": 2}).json()
                deadline += 1
                if info["status"] in ("done", "failed") or deadline > 15:
                    break
            assert info["status"] == "done", info
            assert {d["id"]: d["installed"] for d in client.get("/datasets").json()}["ok-set"]
            assert client.delete("/datasets/ok-set").json() == {"ok": True}
    finally:
        server.shutdown()

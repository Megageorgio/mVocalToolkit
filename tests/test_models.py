import asyncio
import http.server
import threading
import zipfile
from functools import partial

from conftest import make_sofa_model

from mvocaltoolkit.models.catalog import Catalog, CatalogEntry, ModelSource
from mvocaltoolkit.models.layout import detect_sofa, find_model_dirs
from mvocaltoolkit.models.store import ModelStore
from mvocaltoolkit.settings import Home


def _store(tmp_path):
    home = Home(tmp_path / "home")
    settings = home.load_settings()
    catalog = Catalog(home, settings)
    catalog.load()
    return home, ModelStore(home, settings, catalog)


def test_builtin_catalog_loads(tmp_path):
    _home, store = _store(tmp_path)
    ids = set(store.catalog.entries)
    assert "sofa-ru-hhskt-v0.0.1" in ids and "labelmakr-pack-v030" in ids and "whisper-large-v3-turbo" in ids
    assert all(e.engine == "sofa" for e in store.catalog.filter("sofa", "ru"))


def test_detect_safetensors_layout(tmp_path):
    folder = make_sofa_model(tmp_path / "ru_model")
    layout = detect_sofa(folder)
    assert layout["format"] == "safetensors"
    assert layout["dictionary"] == "dict.txt"
    assert layout["vocab"] == "vocab.yaml" and layout["train_config"] == "train_config.yaml"
    assert layout["g2p"] == {"config": "g2p/cfg.yaml", "weights": "g2p/model.ptsd"}


def test_import_zip_with_nested_folder_and_pack(tmp_path):
    home, store = _store(tmp_path)
    src = tmp_path / "src"
    make_sofa_model(src / "pack" / "tgm_sofa_en", safetensors=False)
    make_sofa_model(src / "pack" / "colstone_jp")
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for path in (src / "pack").rglob("*"):
            z.write(path, path.relative_to(src))
    found = find_model_dirs("sofa", src / "pack")
    assert len(found) == 2
    installed = store.import_local("sofa", str(archive))
    ids = sorted(m.id for m in installed)
    assert ids == ["colstone_jp", "tgm_sofa_en"]
    assert store.get_installed("tgm_sofa_en").languages == ["en"]
    assert store.get_installed("colstone_jp").text_frontend == "ja"
    assert (home.models / "tgm_sofa_en" / "model.ckpt").exists()


def test_download_url_source_and_pack(tmp_path):
    home, store = _store(tmp_path)
    serve_dir = tmp_path / "www"
    serve_dir.mkdir()
    make_sofa_model(tmp_path / "build" / "hhskt_ru_v0.0.1")
    with zipfile.ZipFile(serve_dir / "hhskt_ru_v0.0.1_safetensors.zip", "w") as z:
        for path in (tmp_path / "build").rglob("*"):
            z.write(path, path.relative_to(tmp_path / "build"))
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(serve_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/hhskt_ru_v0.0.1_safetensors.zip"
        entry = CatalogEntry(id="ru-test", engine="sofa", languages=["ru"], text_frontend="ru",
                             source=ModelSource(type="url", url=url))
        progress = []
        installed = asyncio.run(store.download(entry, lambda v, m: progress.append(m)))
        assert installed[0].id == "ru-test"
        assert installed[0].layout["format"] == "safetensors"
        assert (home.models / "ru-test" / "model.json").exists()
        assert any("Downloading" in m for m in progress)

        pack = CatalogEntry(id="pack", type="pack", engine="sofa", prefix="lm-", source=ModelSource(type="url", url=url))
        installed = asyncio.run(store.download(pack))
        assert [m.id for m in installed] == ["lm-hhskt_ru_v0.0.1"]
        assert installed[0].languages == ["ru"]
    finally:
        server.shutdown()

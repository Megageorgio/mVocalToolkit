import asyncio
import json
import http.server
import threading
import zipfile
from functools import partial

from conftest import make_hubertfa_model, make_sofa_model

from mvocaltoolkit.models.catalog import Catalog, CatalogEntry, ModelSource
from mvocaltoolkit.models.layout import detect_hubertfa, detect_sofa, find_model_dirs
from mvocaltoolkit.models.store import ModelStore, model_listing
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


def test_hubertfa_layout_and_import(tmp_path):
    _home, store = _store(tmp_path)
    folder = make_hubertfa_model(tmp_path / "hfa" / "my_hfa")
    layout = detect_hubertfa(folder)
    assert layout["model"] == "model.onnx" and layout["languages"] == ["en", "ja"]
    assert layout["dictionaries"]["ja"] == "dictionaries/ja.txt"
    installed = store.import_local("hubertfa", str(tmp_path / "hfa"))
    assert installed[0].languages == ["en", "ja"] and installed[0].tasks == ["align"]
    # an aligner model is found by path for any of the aligner engines
    resolved = store.resolve_local(["sofa", "hubertfa"], str(folder))
    assert resolved.engine == "hubertfa"


def test_pack_members_are_listed_and_downloaded_on_use(tmp_path):
    home, store = _store(tmp_path)
    serve_dir = tmp_path / "www"
    serve_dir.mkdir()
    make_sofa_model(tmp_path / "build" / "models" / "person1_ru")
    make_sofa_model(tmp_path / "build" / "models" / "other_model")
    with zipfile.ZipFile(serve_dir / "pack.zip", "w") as z:
        for path in (tmp_path / "build").rglob("*"):
            z.write(path, path.relative_to(tmp_path / "build"))
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(serve_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/pack.zip"
        (home.catalogs / "my.json").write_text(json.dumps({"name": "my", "models": [{
            "id": "my-pack", "type": "pack", "engine": "sofa", "prefix": "my-",
            "source": {"type": "url", "url": url},
            "models": [{"id": "person1-ru", "folder": "person1_ru", "name": "Person 1", "languages": ["ru"],
                        "text_frontend": "ru"}],
        }]}), encoding="utf-8")
        store.catalog.load()
        ru = [m["id"] for m in model_listing(store, task="align", language="ru")]
        assert "person1-ru" in ru and "sofa-ru-hhskt-v0.0.1" in ru and "my-pack" not in ru
        assert "person1-ru" not in [m["id"] for m in model_listing(store, task="align", language="ja")]
        model = asyncio.run(store.require(["sofa", "hubertfa"], "person1-ru"))
        assert model.id == "person1-ru" and model.languages == ["ru"] and model.source == "my-pack"
        # the other model of the pack is registered too, with the prefix
        assert store.get_installed("my-other_model") is not None
        listed = {m["id"]: m for m in model_listing(store, task="align")}
        assert listed["person1-ru"]["installed"] and listed["my-other_model"]["installed"]
    finally:
        server.shutdown()


def test_game_onnx_layout(tmp_path):
    from mvocaltoolkit.models.layout import detect_game, find_model_dirs  # noqa: PLC0415

    folder = tmp_path / "GAME-1.0.3-small-onnx"
    folder.mkdir()
    for name in ("encoder", "segmenter", "estimator", "dur2bd", "bd2dur"):
        (folder / f"{name}.onnx").write_bytes(b"\0")
    (folder / "config.json").write_text(json.dumps({"samplerate": 44100, "timestep": 0.01,
                                                    "languages": {"en": 1, "ja": 2, "zh": 3}}), encoding="utf-8")
    layout = detect_game(folder)
    assert layout["format"] == "onnx" and layout["languages"] == ["en", "ja", "zh"]
    assert find_model_dirs("game", tmp_path) == [folder]


def _fake_torch_file(path, keys, lightning=False):
    """A zip shaped like torch.save output: <stem>/data.pkl naming the keys."""
    body = b"".join(k.encode() for k in keys)
    if lightning:
        body = b"state_dict" + body + b"pytorch-lightning_version"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{path.stem}/data.pkl", b"\x80\x02" + body)


def test_import_one_wfl_checkpoint_of_a_training_run(tmp_path):
    from mvocaltoolkit.models.layout import wfl_generation

    home, store = _store(tmp_path)
    run = tmp_path / "runs" / "ru"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("finetuning:\n  enable: true\nmodel:\n  encoder_type: whisper\n", encoding="utf-8")
    (run / "phonemes.txt").write_text("B-a\nI-a\n", encoding="utf-8")
    (run / "langs.txt").write_text("ru,0\n", encoding="utf-8")
    (run / "train.json").write_bytes(b"x" * 3_000_000)  # big training data is left out
    _fake_torch_file(run / "model_step5000.pt", ["encoder.conv1.weight"])
    _fake_torch_file(run / "model_step6000.pt", ["encoder.conv1.weight"])
    installed = store.import_local("wfl_asr", str(run / "model_step5000.pt"), "my-ru", "My RU", ["ru"])
    assert [m.id for m in installed] == ["my-ru"]
    m = store.get_installed("my-ru")
    folder = home.models / "my-ru"
    assert sorted(p.name for p in folder.iterdir() if p.name != "model.json") == ["config.yaml", "langs.txt", "model_step5000.pt", "phonemes.txt"]
    assert m.layout["checkpoint"] == "model_step5000.pt" and m.layout["generation"] == 1 and m.languages == ["ru"]

    # refactor-branch models: Lightning checkpoint or the new config keys
    new = tmp_path / "new"
    new.mkdir()
    (new / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _fake_torch_file(new / "model.ckpt", ["model.encoder.x"], lightning=True)
    assert wfl_generation(new / "model.ckpt", new / "config.yaml") == 2
    (new / "config2.yaml").write_text("postprocess:\n  forced_alignment_args: {}\n", encoding="utf-8")
    _fake_torch_file(new / "plain.pt", ["encoder.x"])
    assert wfl_generation(new / "plain.pt", new / "config2.yaml") == 2


def test_catalog_has_the_refactor_wfl_models(tmp_path):
    _home, store = _store(tmp_path)
    e = store.catalog.get("wfl-archivoice-ja-2026-09")
    assert e is not None and e.source.asset == "ja.rar" and e.engine == "wfl_asr"


def test_catalog_has_the_7z_wfl_models(tmp_path):
    _home, store = _store(tmp_path)
    for mid, asset in (("wfl-generic-en-mega5", "mega-5-small.7z"), ("wfl-italian-small-1.1", "ita_small_b.7z")):
        e = store.catalog.get(mid)
        assert e is not None and e.source.asset == asset and e.engine == "wfl_asr"

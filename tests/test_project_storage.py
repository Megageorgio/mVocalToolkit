import os
from pathlib import Path

import pytest

from mvocaltoolkit.training.project_storage import StorageRequest, inventory


def write(root, name, size):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_storage_accounts_project_categories_and_bank_components_without_reading_contents(tmp_path, monkeypatch):
    root = tmp_path / "project"
    for name, size in [("data/prepared/song.wav",100), ("runs/a/binary/data.bin",70), ("runs/a/model_ckpt_steps_1.ckpt",30),
        ("exports/model/acoustic.onnx",90), ("exports/Bank/dsmain/acoustic.onnx",110), ("exports/Bank/dsvocoder/vocoder.onnx",60),
        ("exports/Bank/character.txt",0), ("exports/Bank/dsconfig.yaml",0), ("singforge.json",2)]:
        write(root, name, size)
    before = {str(p.relative_to(root)):p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    def forbidden(*a, **k): raise AssertionError("Storage inventory read file contents")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    result = inventory(StorageRequest(project_dir=str(root)))
    assert result["bytes"] == 462 and result["files"] == 9 and not result["partial"]
    assert {r["category"]:r["bytes"] for r in result["categories"]} == dict(data=100,binaries=70,checkpoints=30,exports=90,banks=170,other=2)
    assert sum(row["bytes"] for row in result["directories"]) == 462
    assert result["largest_files"][0]["path"] == "exports/Bank/dsmain/acoustic.onnx"
    assert {(r["bank"],r["component"]):r["bytes"] for r in result["bank_components"]} == {
        ("exports/Bank","dsmain"):110, ("exports/Bank","dsvocoder"):60, ("exports/Bank","metadata"):0}
    assert before == {str(p.relative_to(root)):p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_hard_links_are_counted_once_for_unique_file_size(tmp_path):
    root = tmp_path / "project"
    original = write(root, "a.bin", 31)
    try: os.link(original, root / "b.bin")
    except OSError: pytest.skip("Hard links unavailable on this filesystem")
    result = inventory(StorageRequest(project_dir=str(root)))
    assert result["bytes"] == 62 and result["unique_file_bytes"] == 31 and result["files"] == 2


def test_symbolic_link_does_not_expand_inventory_into_external_recordings(tmp_path):
    root = tmp_path / "project"
    write(root, "local.bin", 5)
    write(tmp_path, "external/large.wav", 1000)
    try: os.symlink(tmp_path / "external", root / "recordings", target_is_directory=True)
    except OSError: pytest.skip("Symbolic links unavailable on this filesystem")
    result = inventory(StorageRequest(project_dir=str(root)))
    assert result["bytes"] == 5 and result["skipped_links"] == 1 and result["files"] == 1


def test_storage_limits_and_unreadable_folders_are_reported_as_partial(tmp_path, monkeypatch):
    root = tmp_path / "project"
    write(root, "a.bin", 3); write(root, "b.bin", 4); write(root, "denied/c.bin", 5)
    limited = inventory(StorageRequest(project_dir=str(root), max_entries=1))
    assert limited["partial"] and limited["partial_reason"] == "entry_limit" and limited["entries_scanned"] == 1
    original = os.scandir
    def scandir(path):
        if Path(path).name == "denied": raise PermissionError("fixture denied")
        return original(path)
    monkeypatch.setattr(os, "scandir", scandir)
    result = inventory(StorageRequest(project_dir=str(root)))
    assert result["partial"] and result["partial_reason"] == "unreadable_entries"
    assert result["bytes"] == 7 and result["inaccessible"] == 1
    import mvocaltoolkit.training.project_storage as module
    ticks = iter(range(0,100,10))
    monkeypatch.setattr(module, "perf_counter", lambda:next(ticks))
    result = inventory(StorageRequest(project_dir=str(root), max_seconds=1))
    assert result["partial"] and result["partial_reason"] == "time_limit"


def test_storage_api_is_read_only_and_rejects_invalid_limits_without_jobs(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from mvocaltoolkit.server.app import create_app
    from mvocaltoolkit.settings import Home, Settings
    from mvocaltoolkit.toolkit import Toolkit
    root = tmp_path / "project"
    write(root, "singforge.json", 2)
    tk = Toolkit(Home(tmp_path / "home"), Settings(catalog_url="", auto_install_engines=False, auto_download_models=False))
    async def forbidden(*a, **k): raise AssertionError("Storage inventory invoked an engine")
    monkeypatch.setattr(tk.engines, "call", forbidden)
    with TestClient(create_app(tk)) as client:
        response = client.post("/training/project/storage", json={"project_dir":str(root)})
        assert response.status_code == 200 and response.json()["bytes"] == 2
        for changes in ({"max_entries":0}, {"max_seconds":True}, {"top_limit":101}, {"delete":True}, {"project_dir":" "}):
            assert client.post("/training/project/storage", json={"project_dir":str(root), **changes}).status_code == 400
        assert client.get("/jobs").json() == []

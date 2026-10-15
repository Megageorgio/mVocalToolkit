import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mvocaltoolkit.jobs import JobCancelled
from mvocaltoolkit.training.bank_audition import AuditionRequest, bank_inventory, plan_audition, run_audition


def bank_fixture(root):
    root.mkdir()
    (root / "dsconfig.yaml").write_text(yaml.safe_dump(dict(acoustic="acoustic.onnx", phonemes="phonemes.json", speakers=["one", "two"], use_lang_id=True)), encoding="utf-8")
    (root / "phonemes.json").write_text(json.dumps({"SP":0, "ru/a":1, "ru/p":2, "en/aa":3, "en/p":4}), encoding="utf-8")
    (root / "languages.json").write_text(json.dumps({"ru":1, "en":2}), encoding="utf-8")
    for speaker in ("one", "two"): (root / (speaker + ".emb")).write_bytes(b"embedding metadata fixture")
    (root / "dsvocoder").mkdir()
    (root / "dsvocoder/vocoder.yaml").write_text("model: model.onnx", encoding="utf-8")
    (root / "acoustic.onnx").write_bytes(b"graph metadata fixture")
    (root / "dsvocoder/model.onnx").write_bytes(b"graph metadata fixture")
    return root


def request(root, **changes):
    return AuditionRequest.model_validate(dict(bank=str(root), selections=[dict(speaker="one", language="ru"), dict(speaker="two", language="en")], phrase=dict(mode="cv", limit=2), **changes))


def test_inventory_and_planning_read_metadata_only_and_use_language_vocabularies(tmp_path):
    root = bank_fixture(tmp_path / "bank")
    before = {p:p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    assert bank_inventory(str(root)) == dict(bank=str(root.resolve()), speakers=["one", "two"], languages=["ru", "en"], requires_language=False)
    plan = plan_audition(request(root))
    assert [row["phrase"]["syllables"] for row in plan["rows"]] == [["ru/p ru/a"], ["en/p en/aa"]]
    assert plan["count"] == 2 and plan["audio_seconds"] > 0
    assert before == {p:p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_invalid_selections_and_phrase_paths_are_rejected_before_execution(tmp_path):
    root = bank_fixture(tmp_path / "bank")
    base = request(root).model_dump()
    for changes in (
        dict(selections=[dict(speaker="missing", language="ru")]),
        dict(selections=[dict(speaker="one", language="fr")]),
        dict(selections=[dict(speaker="one", language="ru")] * 2),
        dict(selections=[dict(speaker="one", language="ru")] * 17),
        dict(phrase=dict(mode="manual", phonemes="unknown")),
        dict(phrase=dict(model_dir=str(tmp_path))), dict(steps=True), dict(cpu_threads=0), dict(cache_models="yes"),
    ):
        with pytest.raises(ValueError): plan_audition(AuditionRequest.model_validate({**base, **changes}))
    (root / "dsconfig.yaml").write_text("speakers: [../escape]\nphonemes: phonemes.json", encoding="utf-8")
    with pytest.raises(ValueError): bank_inventory(str(root))


def test_metadata_free_language_must_be_selected_explicitly(tmp_path):
    root = bank_fixture(tmp_path / "bank")
    (root / "dsconfig.yaml").write_text("phonemes: phonemes.json", encoding="utf-8")
    (root / "phonemes.json").write_text('["SP", "a", "p"]', encoding="utf-8")
    info = bank_inventory(str(root))
    assert info["languages"] == [] and info["requires_language"]
    req = request(root).model_copy(update={"selections": []})
    # A normal validated request cannot omit a language, even for an untagged bank.
    with pytest.raises(ValueError): AuditionRequest.model_validate({**req.model_dump(), "selections":[{"speaker":"", "language":""}]})


def test_incompatible_or_missing_vocoder_is_rejected_during_planning(tmp_path):
    root = bank_fixture(tmp_path / "bank")
    with (root / "dsconfig.yaml").open("a", encoding="utf-8") as stream: stream.write("sample_rate: 44100\n")
    with (root / "dsvocoder/vocoder.yaml").open("a", encoding="utf-8") as stream: stream.write("\nsample_rate: 48000\n")
    with pytest.raises(ValueError, match="Incompatible vocoder"): plan_audition(request(root))
    (root / "dsvocoder/model.onnx").unlink()
    with pytest.raises(ValueError, match="Missing bank file"): plan_audition(request(root))


class FakeJob:
    def __init__(self): self.info = SimpleNamespace(result=None); self.cancelled = False
    def check_cancelled(self):
        if self.cancelled: raise JobCancelled()
    def progress(self, *args, **kwargs): pass


@pytest.mark.parametrize("finish", ["done", "cancelled", "failed"])
def test_batch_is_serial_and_persists_completed_results_on_stop_or_failure(tmp_path, finish):
    root = bank_fixture(tmp_path / "bank")
    req = request(root); plan = plan_audition(req); job = FakeJob(); calls = []; active = 0
    async def render(params):
        nonlocal active
        active += 1
        assert active == 1
        calls.append(params)
        await asyncio.sleep(0)
        active -= 1
        assert Path(params["ds"]).is_file()
        if len(calls) == 2 and finish == "failed": raise ValueError("fixture failure")
        if len(calls) == 1 and finish == "cancelled": job.cancelled = True
        return dict(files=[str(Path(params["out_dir"]) / "render.wav")], profile={"real_time_factor": .5})
    async def run(): return await run_audition(req, plan, tmp_path / "outputs", job, render)
    if finish == "done": assert asyncio.run(run())["completed"] == 2
    else:
        with pytest.raises(JobCancelled if finish == "cancelled" else ValueError): asyncio.run(run())
    report = json.loads((tmp_path / "outputs/audition-report.json").read_text(encoding="utf-8"))
    assert report == job.info.result and report["status"] == finish
    assert report["completed"] == (2 if finish == "done" else 1)
    assert len(calls) == (1 if finish == "cancelled" else 2)
    assert calls[0]["speaker"] == "one" and calls[0]["cpu_threads"] == 2


def test_api_planning_does_not_create_jobs_or_invoke_engines(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from mvocaltoolkit.server.app import create_app
    from mvocaltoolkit.settings import Home, Settings
    from mvocaltoolkit.toolkit import Toolkit
    root = bank_fixture(tmp_path / "bank")
    tk = Toolkit(Home(tmp_path / "home"), Settings(catalog_url="", auto_install_engines=False, auto_download_models=False))
    async def forbidden(*a, **k): raise AssertionError("Planning invoked an engine")
    monkeypatch.setattr(tk.engines, "call", forbidden)
    with TestClient(create_app(tk)) as client:
        assert client.post("/training/voicebank/audition/inventory", json={"bank":str(root)}).status_code == 200
        assert client.post("/training/voicebank/audition/plan", json=request(root).model_dump()).json()["count"] == 2
        for data in ({"bank":" "}, {"bank":str(root), "extra":1}, {"bank":True}):
            assert client.post("/training/voicebank/audition/inventory", json=data).status_code == 400
        bad = {**request(root).model_dump(), "phrase":{"mode":"manual", "phonemes":"unknown"}}
        assert client.post("/training/voicebank/audition/start", json=bad).status_code == 400
        (root / "dsconfig.yaml").write_text("[invalid", encoding="utf-8")
        assert client.post("/training/voicebank/audition/inventory", json={"bank":str(root)}).status_code == 400
        assert client.get("/jobs").json() == []


@pytest.mark.parametrize("cancel", [False, True])
def test_api_executes_series_in_shared_inference_slot_and_keeps_partial_result(tmp_path, monkeypatch, cancel):
    from contextlib import asynccontextmanager
    import threading
    from fastapi.testclient import TestClient
    from mvocaltoolkit.server.app import create_app
    from mvocaltoolkit.settings import Home, Settings
    from mvocaltoolkit.toolkit import Toolkit
    root = bank_fixture(tmp_path / "bank")
    tk = Toolkit(Home(tmp_path / "home"), Settings(catalog_url="", auto_install_engines=False, auto_download_models=False))
    calls, slot_active, aborted = [], False, []
    second_started = threading.Event()
    @asynccontextmanager
    async def slot():
        nonlocal slot_active
        assert not slot_active
        slot_active = True
        try: yield
        finally: slot_active = False
    async def engine_call(engine, method, params, **kwargs):
        assert slot_active and engine == "bank_render" and method == "render"
        calls.append(params)
        if len(calls) == 2:
            second_started.set()
            if cancel: await asyncio.sleep(10)
        return dict(files=[str(Path(params["out_dir"]) / "render.wav")])
    async def abort(engine): aborted.append(engine)
    monkeypatch.setattr(tk.training, "inference_slot", slot)
    monkeypatch.setattr(tk.engines, "call", engine_call)
    monkeypatch.setattr(tk.engines, "abort", abort)
    with TestClient(create_app(tk)) as client:
        response = client.post("/training/voicebank/audition/start", json=request(root).model_dump())
        assert response.status_code == 200
        job_id = response.json()["id"]
        if cancel:
            assert second_started.wait(timeout=3)
            assert client.post(f"/jobs/{job_id}/cancel").status_code == 200
        info = client.get(f"/jobs/{job_id}?wait=3").json()
        assert info["status"] == ("cancelled" if cancel else "done")
        assert info["result"]["completed"] == (1 if cancel else 2)
        saved = json.loads(Path(info["result"]["report"]).read_text(encoding="utf-8"))
        assert saved == info["result"] and saved["status"] == info["status"]
        assert len(calls) == 2 and not slot_active
        assert aborted == (["bank_render"] if cancel else [])

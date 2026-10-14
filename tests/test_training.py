"""Training runs with the fake engine: queue, trainer process, events, stop and resume, re-attach, the API."""

import asyncio
import json
import time

from conftest import FAKE_ENGINES, make_wav
from fastapi.testclient import TestClient

from mvocaltoolkit.server.app import create_app
from mvocaltoolkit.settings import Home
from mvocaltoolkit.toolkit import Toolkit
from mvocaltoolkit.training.runs import RunSpec, RunStatus


def _toolkit(tmp_path, name="home"):
    return Toolkit(Home(tmp_path / name), engine_dirs=[FAKE_ENGINES])


async def _wait_status(run, statuses, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if run.info.status in statuses:
            return run.info
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run.id} is {run.info.status}: {run.info.error}")


def test_run_trains_to_the_end(tmp_path):
    async def main():
        tk = _toolkit(tmp_path)
        await tk.start()
        try:
            run = tk.training.start(RunSpec(kind="fake_acoustic", config={"max_steps": 30, "val_every": 10}))
            assert run.info.status == RunStatus.queued
            info = await _wait_status(run, {RunStatus.done, RunStatus.failed})
            assert info.status == RunStatus.done, info.error
            assert info.step == 30 and info.max_steps == 30
            assert [c.step for c in info.checkpoints] == [10, 20, 30]
            assert sorted(int(s) for s in info.samples) == [10, 20, 30]
            assert info.val_loss is not None and info.best_val_step == 30
            assert info.speed is None and info.pid is None and info.finished_at
            # the events file and run.json are complete and re-readable
            events = run.read_events_since(0)
            kinds = [e["event"] for e in events]
            assert kinds[0] == "queued" and "launched" in kinds and kinds[-1] == "end"
            saved = json.loads((run.dir / "run.json").read_text(encoding="utf-8"))
            assert saved["status"] == "done"
            assert (run.dir / "train.log").exists() and (run.dir / "prepare.log").exists()
            # a second run waits for the first (the queue) and then runs
            second = tk.training.start(RunSpec(kind="fake_acoustic", config={"max_steps": 10, "val_every": 5}))
            info2 = await _wait_status(second, {RunStatus.done})
            assert info2.step == 10
        finally:
            await tk.stop()

    asyncio.run(main())


def test_stop_resume_and_failure(tmp_path):
    async def main():
        tk = _toolkit(tmp_path)
        await tk.start()
        try:
            run = tk.training.start(RunSpec(kind="fake_acoustic", config={"max_steps": 200, "val_every": 20, "step_seconds": 0.03}))
            await _wait_status(run, {RunStatus.training})
            await asyncio.sleep(0.5)
            await tk.training.stop(run.id)
            info = await _wait_status(run, {RunStatus.stopped, RunStatus.done, RunStatus.failed})
            assert info.status == RunStatus.stopped, info.error
            stopped_at = info.step
            assert 0 < stopped_at < 200 and info.checkpoints and info.checkpoints[-1].step >= stopped_at - 5
            assert not run.stop_file.exists()
            # resumes from the checkpoint it saved
            await tk.training.resume(run.id)
            info = await _wait_status(run, {RunStatus.done, RunStatus.failed}, timeout=60)
            assert info.status == RunStatus.done, info.error
            assert info.step == 200
            events = run.read_events_since(0)
            starts = [e for e in events if e["event"] == "start"]
            assert len(starts) == 2 and starts[1]["step"] >= stopped_at - 5
            # a trainer that dies with an error
            bad = tk.training.start(RunSpec(kind="fake_acoustic", config={"max_steps": 50, "val_every": 10, "fail_at": 7}))
            info = await _wait_status(bad, {RunStatus.done, RunStatus.failed})
            assert info.status == RunStatus.failed and info.error_kind == "oom", info
            assert "out of memory" in (info.error or "")
        finally:
            await tk.stop()

    asyncio.run(main())


def test_reattach_after_restart(tmp_path):
    async def main():
        tk = _toolkit(tmp_path)
        await tk.start()
        run = tk.training.start(RunSpec(kind="fake_acoustic", config={"max_steps": 60, "val_every": 20, "step_seconds": 0.05}))
        await _wait_status(run, {RunStatus.training})
        pid = run.info.pid
        # the server goes away while the trainer keeps running
        await tk.stop()
        run.save(force=True)
        assert pid
        tk2 = _toolkit(tmp_path)
        await tk2.start()
        try:
            again = tk2.training.get(run.id)
            assert again is not None and again.info.status == RunStatus.training and again.info.pid == pid
            info = await _wait_status(again, {RunStatus.done, RunStatus.failed}, timeout=60)
            assert info.status == RunStatus.done and info.step == 60
            assert [c.step for c in info.checkpoints] == [20, 40, 60]
        finally:
            await tk2.stop()

    asyncio.run(main())


def test_training_api(tmp_path):
    home = Home(tmp_path / "home")
    tk = Toolkit(home, engine_dirs=[FAKE_ENGINES])
    with TestClient(create_app(tk)) as client:
        env = client.get("/training/environment", params={"engine": "fake"}).json()
        assert env["installed"] is True and env["cuda"] is False
        r = client.post("/training/runs", json={"kind": "fake_acoustic", "config": {"max_steps": 40, "val_every": 20}})
        assert r.status_code == 200, r.text
        run_id = r.json()["id"]
        # events by long polling
        seen = 0
        status = ""
        deadline = time.time() + 30
        while time.time() < deadline:
            page = client.get(f"/training/runs/{run_id}/events", params={"since": seen, "wait": 2}).json()
            seen = page["next"]
            status = page["status"]
            if status in ("done", "failed", "stopped"):
                break
        assert status == "done"
        info = client.get(f"/training/runs/{run_id}").json()
        assert info["step"] == 40 and [c["step"] for c in info["checkpoints"]] == [20, 40]
        samples = client.get(f"/training/runs/{run_id}/samples").json()
        assert set(samples) == {"20", "40"}
        path = samples["40"][0]
        assert client.get("/files/download", params={"path": path}).status_code == 200
        assert client.get(f"/training/runs/{run_id}/log").json()["lines"]
        assert client.get(f"/training/runs/{run_id}/checkpoints").json()[-1]["step"] == 40
        job = client.post(f"/training/runs/{run_id}/export", json={"step": 40}).json()
        for _ in range(100):
            j = client.get(f"/jobs/{job['id']}", params={"wait": 1}).json()
            if j["status"] in ("done", "failed"):
                break
        assert j["status"] == "done" and j["result"]["files"]
        listed = client.get("/training/runs").json()
        assert [i["id"] for i in listed] == [run_id]
        assert client.post("/training/runs/nope/stop").status_code == 404
        assert client.delete(f"/training/runs/{run_id}", params={"files": "true"}).json() == {"ok": True}
        assert client.get("/training/runs").json() == []


def test_diffsinger_overlay_and_dataset(tmp_path):
    from mvocaltoolkit.training.dataset import SourceSpec, build_dataset, spec_from_result
    from mvocaltoolkit.training.diffsinger import build_overlay, resolve_assets

    src = tmp_path / "src"
    for i in range(5):
        make_wav(src / f"song{i}.wav", seconds=1.0)
        (src / f"song{i}.lab").write_text("0 2000000 SP\n2000000 6000000 a\n6000000 8000000 k\n8000000 10000000 SP\n", encoding="utf-8")
    (src / "nolabel.wav").write_bytes(make_wav(tmp_path / "x.wav").read_bytes())
    result = build_dataset([SourceSpec(folder=str(src), speaker="masha", language="ru")], tmp_path / "ds")
    assert result.files == 5 and result.phonemes == {"ru": ["a", "k"]}
    assert result.problems == [{"file": str(src / "nolabel.wav"), "problem": "no labels"}]
    assert (tmp_path / "ds" / "masha" / "transcriptions.csv").read_text(encoding="utf-8").startswith("name,ph_seq,ph_dur\n")
    assert (tmp_path / "ds" / "dictionary-ru.txt").read_text(encoding="utf-8") == "a\ta\nk\tk\n"
    assert len(result.speakers[0]["test_prefixes"]) == 3
    config = spec_from_result(result, preset={"goal": "check", "gpu_gb": 8})
    overlay = build_overlay("diffsinger_acoustic", config)
    assert overlay["max_updates"] == 200 and overlay["max_batch_frames"] == 30000 and overlay["pe"] == "rmvpe"
    assert overlay["datasets"][0]["speaker"] == "masha" and overlay["use_spk_id"] is False and overlay["num_lang"] == 1
    assert overlay["augmentation_args"]["random_pitch_shifting"]["range"] == [-3.0, 3.0]
    assert resolve_assets(config)["vocoder"] == "diffsinger-pc-nsf-hifigan-2025.02"
    variance = build_overlay("diffsinger_variance", {**config, "variance": {"dur": True, "pitch": True, "breathiness": True}})
    assert variance["predict_dur"] and variance["predict_breathiness"] and not variance["predict_energy"]
    assert variance["tension_logit_max"] == 8.0
    try:
        build_overlay("diffsinger_variance", {**config, "variance": {"energy": True, "breathiness": True, "voicing": True}})
    except ValueError as e:
        assert "coupled" in str(e)
    else:
        raise AssertionError("coupled parameters accepted")

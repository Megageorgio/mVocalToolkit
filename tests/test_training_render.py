import json
import asyncio

from fastapi.testclient import TestClient

from conftest import FAKE_ENGINES, make_wav
from mvocaltoolkit.server.app import create_app
from mvocaltoolkit.settings import Home
from mvocaltoolkit.toolkit import Toolkit
from mvocaltoolkit.training.runs import Run, RunInfo, RunSpec, RunStatus


def test_render_snapshots_input_and_persists_results_without_overwrite(tmp_path, monkeypatch):
    tk = Toolkit(Home(tmp_path / "home"), engine_dirs=[FAKE_ENGINES])
    app = create_app(tk)
    captured = []

    async def call(engine, method, params, **kwargs):
        from pathlib import Path
        captured.append(params)
        assert json.loads(Path(params["ds"]).read_text()) == [{"ph_seq": "a"}]
        output = make_wav(Path(params["out_dir"]) / "phrase.wav", seconds=0.05)
        return {"out_dir": str(output.parent), "files": [str(output)]}

    monkeypatch.setattr(tk.engines, "call", call)
    with TestClient(app) as client:
        folder = tmp_path / "run"
        run = Run(RunInfo(id="saved", spec=RunSpec(kind="fake_acoustic"), dir=str(folder), status=RunStatus.done))
        tk.training.runs[run.id] = run
        run.save(force=True)
        ds = tmp_path / "phrase.ds"
        ds.write_text('[{"ph_seq":"a"}]', encoding="utf-8")
        for _ in range(2):
            response = client.post("/training/runs/saved/render", json={"ds": str(ds), "step": 10, "seed": 42})
            assert response.status_code == 200, response.text
            job = response.json()
            for _ in range(30):
                result = client.get(f"/jobs/{job['id']}", params={"wait": 1}).json()
                if result["status"] in {"done", "failed"}:
                    break
            assert result["status"] == "done", result
        assert captured[0]["out_dir"] != captured[1]["out_dir"]
        saved = json.loads((folder / "run.json").read_text())
        assert len(saved["renders"]) == 2 and saved["renders"][0]["seed"] == 42
        assert not tk.training._inference
        run.info.status = RunStatus.training
        assert client.post("/training/runs/saved/render", json={"ds": str(ds)}).status_code == 409


def test_retain_api_and_invalid_render(tmp_path):
    tk = Toolkit(Home(tmp_path / "home"), engine_dirs=[FAKE_ENGINES])
    with TestClient(create_app(tk)) as client:
        folder = tmp_path / "run"
        run = Run(RunInfo(id="saved", spec=RunSpec(kind="diffsinger_acoustic"), dir=str(folder), status=RunStatus.done))
        tk.training.runs[run.id] = run
        run.save(force=True)
        (folder / "model_ckpt_steps_10.ckpt").write_bytes(b"weights")
        response = client.post("/training/runs/saved/checkpoints/10/retain", json={"starred": True})
        assert response.status_code == 200 and response.json()["checkpoints"][0]["starred"]
        assert client.get("/training/runs/saved/checkpoints").json()[0]["starred"]
        assert client.post("/training/runs/saved/checkpoints/99/retain", json={}).status_code == 409
        assert client.post("/training/runs/saved/render", json={"ds": str(tmp_path / "missing.ds")}).status_code == 400
        ds = tmp_path / "bad.ds"
        ds.write_text('{}')
        assert client.post("/training/runs/saved/render", json={"ds": str(ds)}).status_code == 400


def test_render_cancellation_aborts_worker_and_releases_training_queue(tmp_path, monkeypatch):
    tk = Toolkit(Home(tmp_path / "home"), engine_dirs=[FAKE_ENGINES])
    aborted = []
    async def call(*args, **kwargs):
        await asyncio.sleep(30)
    async def abort(engine):
        aborted.append(engine)
    monkeypatch.setattr(tk.engines, "call", call)
    monkeypatch.setattr(tk.engines, "abort", abort)
    with TestClient(create_app(tk)) as client:
        run = Run(RunInfo(id="saved", spec=RunSpec(kind="fake_acoustic"), dir=str(tmp_path / "run"), status=RunStatus.done))
        tk.training.runs[run.id] = run
        run.save(force=True)
        ds = tmp_path / "phrase.ds"
        ds.write_text('[{"ph_seq":"a"}]')
        job = client.post("/training/runs/saved/render", json={"ds": str(ds)}).json()
        client.post(f"/jobs/{job['id']}/cancel")
        for _ in range(10):
            result = client.get(f"/jobs/{job['id']}", params={"wait": 1}).json()
            if result["status"] == "cancelled":
                break
        assert result["status"] == "cancelled" and not tk.training._inference
        assert aborted == ["fake"]
        assert run.info.renders == []

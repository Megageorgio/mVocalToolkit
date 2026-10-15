"""Recovery must not discard tail events or replay an old end event after resume."""
import asyncio
import json

import pytest

from conftest import FAKE_ENGINES
from mvocaltoolkit.settings import Home
from mvocaltoolkit.toolkit import Toolkit
from mvocaltoolkit.training.runs import Run, RunInfo, RunSpec, RunStatus


def _toolkit(tmp_path):
    return Toolkit(Home(tmp_path / "home"), engine_dirs=[FAKE_ENGINES])


def test_existing_run_directory_is_never_replaced(tmp_path):
    tk = _toolkit(tmp_path)
    folder = tmp_path / "voice" / "runs" / "acoustic-1"
    first = tk.training.start(RunSpec(kind="fake_acoustic", dir=str(folder)))
    with pytest.raises(ValueError, match="not empty"):
        tk.training.start(RunSpec(kind="fake_acoustic", dir=str(folder)))
    assert json.loads((folder / "run.json").read_text(encoding="utf-8"))["id"] == first.id


def test_replays_unsaved_events_after_server_was_offline(tmp_path):
    tk = _toolkit(tmp_path)
    run = tk.training.start(RunSpec(kind="fake_acoustic"))
    run.info.status = RunStatus.training
    run.save(force=True)
    run.emit("step", step=50, losses={"total_loss": 0.3})
    run.emit("checkpoint", step=50, path=str(run.dir / "model_ckpt_steps_50.ckpt"))
    run.emit("end", status="done")
    restored = _toolkit(tmp_path).training.get(run.id)
    assert restored.info.status == RunStatus.done
    assert restored.info.step == 50
    assert restored.info.checkpoints[0].step == 50
    assert restored.info.events == 4
    assert restored.read_new_events() == []


def test_stopped_run_consumes_history_before_resume(tmp_path):
    async def main():
        tk = _toolkit(tmp_path)
        run = tk.training.start(RunSpec(kind="fake_acoustic", config={"max_steps": 15, "val_every": 5}))
        await tk.training.stop(run.id)  # stops while still queued
        await tk.stop()
        tk2 = _toolkit(tmp_path)
        await tk2.start()
        try:
            restored = tk2.training.get(run.id)
            assert restored.info.status == RunStatus.stopped
            await tk2.training.resume(run.id)
            for _ in range(300):
                if restored.info.status in {RunStatus.done, RunStatus.failed}:
                    break
                await asyncio.sleep(.05)
            assert restored.info.status == RunStatus.done, restored.info.error
            assert restored.info.step == 15
        finally:
            await tk2.stop()
    asyncio.run(main())


def test_overfit_hint_needs_sustained_validation_rise_and_training_fall(tmp_path):
    tk = _toolkit(tmp_path)
    run = Run(RunInfo(id="trend", spec=RunSpec(kind="fake_acoustic"), dir=str(tmp_path / "run")))
    for step, loss, train in [(10, 1., 2.), (20, 1.1, 1.8), (30, 1.2, 1.6), (40, 1.3, 1.4)]:
        run.info.losses = {"total_loss": train}
        tk.training._apply(run, {"event": "validation", "step": step, "val_loss": loss})
    assert run.info.warnings == []
    run.info.losses = {"total_loss": 1.2}
    tk.training._apply(run, {"event": "validation", "step": 50, "val_loss": 1.4})
    assert len(run.info.warnings) == 1 and run.info.warnings[0]["kind"] == "overfit"
    for event in run.read_new_events():
        tk.training._apply(run, event)
    assert len(run.info.warnings) == 1
    assert run.info.best_val_step == 10


def test_nonfinite_losses_do_not_poison_run_state(tmp_path):
    tk = _toolkit(tmp_path)
    run = Run(RunInfo(id="finite", spec=RunSpec(kind="fake_acoustic"), dir=str(tmp_path / "run")))
    tk.training._apply(run, {"event": "step", "step": 1, "losses": {"loss": float("nan"), "other": .5}})
    tk.training._apply(run, {"event": "validation", "step": 1, "val_loss": float("inf")})
    assert run.info.losses == {"other": .5} and run.info.val_loss is None


def test_queue_order_persists_and_rejects_stale_requests(tmp_path):
    tk = _toolkit(tmp_path)
    first = tk.training.start(RunSpec(kind="fake_acoustic"))
    second = tk.training.start(RunSpec(kind="fake_acoustic"))
    tk.training.reorder_queue([second.id, first.id])
    restored = _toolkit(tmp_path)
    queue = sorted(restored.training.list(), key=lambda info: info.queue_order)
    assert [info.id for info in queue] == [second.id, first.id]
    with pytest.raises(ValueError, match="every currently queued"):
        tk.training.reorder_queue([first.id])
    with pytest.raises(ValueError):
        tk.training.reorder_queue([first.id, first.id])

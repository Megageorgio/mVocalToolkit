import asyncio
from pathlib import Path

import pytest

from mvocaltoolkit.engines.diffsinger.checkpoint_store import listing, protect_removal, retain, restore, retain_best


def test_retention_survives_rotation_and_unstarring_preserves_the_checkpoint(tmp_path):
    original = tmp_path / "model_ckpt_steps_40.ckpt"
    original.write_bytes(b"weights")
    retain(tmp_path, 40, True)
    original.unlink()  # simulate an older trainer without the deletion hook
    assert listing(tmp_path)[0]["starred"]
    restore(tmp_path, 40)
    assert original.read_bytes() == b"weights"
    original.unlink()
    retain(tmp_path, 40, False)
    assert original.read_bytes() == b"weights"
    assert not listing(tmp_path)[0]["starred"]


def test_trainer_removes_only_unstarred_checkpoints(tmp_path):
    class Callback:
        def _remove_checkpoint(self, trainer, filepath):
            Path(filepath).unlink()

    first = tmp_path / "model_ckpt_steps_10.ckpt"
    second = tmp_path / "model_ckpt_steps_20.ckpt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    retain(tmp_path, 10, True)
    protect_removal(Callback, tmp_path)
    Callback()._remove_checkpoint(None, str(first))
    Callback()._remove_checkpoint(None, str(second))
    assert first.exists() and not second.exists()
    retain(tmp_path, 10, False)
    Callback()._remove_checkpoint(None, str(first))
    assert not first.exists()


def test_missing_checkpoint_cannot_be_starred(tmp_path):
    with pytest.raises(ValueError, match="no longer"):
        retain(tmp_path, 10, True)
    with pytest.raises(ValueError, match="non-negative"):
        retain(tmp_path, -1, True)


def test_best_checkpoint_backup_rotates_without_removing_manual_stars(tmp_path):
    for step in (10, 20):
        (tmp_path / f"model_ckpt_steps_{step}.ckpt").write_bytes(str(step).encode())
    retain(tmp_path, 10, True)
    retain_best(tmp_path, 10)
    retain_best(tmp_path, 20)
    assert len(list((tmp_path / "best").glob("*.ckpt"))) == 1
    (tmp_path / "model_ckpt_steps_10.ckpt").unlink()
    (tmp_path / "model_ckpt_steps_20.ckpt").unlink()
    assert [item["step"] for item in listing(tmp_path)] == [10, 20]
    restore(tmp_path, 20)
    assert (tmp_path / "model_ckpt_steps_20.ckpt").read_bytes() == b"20"

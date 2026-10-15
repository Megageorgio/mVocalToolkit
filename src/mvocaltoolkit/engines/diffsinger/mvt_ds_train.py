"""Starts DiffSinger training as its own process and reports what happens in <run dir>/events.jsonl.

    python mvt_ds_train.py --run-dir <run dir> [--reset]

Equivalent to `scripts/train.py --config <run dir>/config.yaml --exp_name <run dir> [--reset]` (the run folder is
the experiment folder: DiffSinger accepts an absolute path there), plus one Lightning callback that writes events:
    start      {step, max_steps}
    step       {step, max_steps, epoch, losses, lr, gpu_mb}       every log_interval steps
    validation {step, val_loss, metrics}
    checkpoint {step, path, size} / checkpoint_removed {step}
    samples    {step, files}   the validation audio DiffSinger logged to TensorBoard, as WAV files in samples/<step>/
    end        {status: done | stopped | failed, error, error_kind}
A STOP file in the run folder makes it save a checkpoint and end. Nothing upstream is modified: the callback is
added by wrapping lightning.pytorch.Trainer.__init__ in this process only."""

from __future__ import annotations

import argparse
import json
import os
import re
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SOURCE = Path(os.environ.get("MVT_SOURCE_DIR") or os.getcwd()).resolve()
CKPT_GLOB = "model_ckpt_steps_*.ckpt"


def _write_event(file: Path, event: str, **data: Any) -> None:
    """Appends one event; the first parameter is not called path because events carry a "path" of their own."""
    payload = {"event": event, "time": time.time(), **data}
    with file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _ckpt_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


class SampleExtractor:
    """Reads TensorBoard event files of the run and writes the audio summaries as WAV files."""

    def __init__(self, log_dir: Path, out_dir: Path, step_divisor: int = 1):
        self.log_dir = log_dir
        self.out_dir = out_dir
        self.loaders: dict[str, Any] = {}
        self.step_divisor = step_divisor

    def pull(self) -> dict[int, list[str]]:
        try:
            from tensorboard.backend.event_processing.event_file_loader import EventFileLoader  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return {}
        written: dict[int, list[str]] = {}
        if not self.log_dir.exists():
            return written
        for file in sorted(self.log_dir.glob("events.out.tfevents.*")):
            loader = self.loaders.get(str(file))
            if loader is None:
                loader = self.loaders[str(file)] = EventFileLoader(str(file))
            try:
                for event in loader.Load():
                    if not event.HasField("summary"):
                        continue
                    for value in event.summary.value:
                        # torch's SummaryWriter writes audio as a string tensor (plugin "audio"); older writers as
                        # an audio summary
                        data = b""
                        if value.HasField("tensor") and value.metadata.plugin_data.plugin_name == "audio":
                            if value.tensor.string_val:
                                data = value.tensor.string_val[0]
                        elif value.HasField("audio"):
                            data = value.audio.encoded_audio_string
                        if not data:
                            continue
                        step = int(event.step) // self.step_divisor
                        folder = self.out_dir / str(step)
                        folder.mkdir(parents=True, exist_ok=True)
                        name = value.tag.replace("/", "_") + ".wav"
                        if self.step_divisor == 2:
                            match = re.fullmatch(r"(HIFI|gt)_(\d+)_", value.tag)
                            if match:
                                name = ("diff" if match[1] == "HIFI" else "gt") + "_" + match[2] + ".wav"
                        (folder / name).write_bytes(data)
                        written.setdefault(step, []).append(f"samples/{step}/{name}")
            except Exception:  # noqa: BLE001  (a file being written: read again next time)
                continue
        return written


def make_callback(run_dir: Path, log_interval: int, step_divisor: int = 1, sample_log_dir: Path | None = None):
    import lightning.pytorch as pl  # noqa: PLC0415
    import torch  # noqa: PLC0415

    events = run_dir / "events.jsonl"
    stop_file = run_dir / "STOP"

    class MvtEvents(pl.Callback):
        def __init__(self) -> None:
            self.known: dict[int, Path] = {}
            self.samples = SampleExtractor(sample_log_dir or run_dir / "lightning_logs" / "latest", run_dir / "samples", step_divisor)
            self.stopping = False
            self.ended = False
            self.last_step_event = -1
            self.pending_scan = False
            self.best_step: int | None = None
            self.best_loss = float("inf")

        # ---- helpers ----
        def _losses(self, trainer: Any) -> dict[str, float]:
            out: dict[str, float] = {}
            metrics = getattr(trainer, "progress_bar_metrics", None) or {}
            for key, value in metrics.items():
                try:
                    out[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            return out

        def _gpu_mb(self) -> float | None:
            try:
                if torch.cuda.is_available():
                    return round(torch.cuda.max_memory_reserved() / 2**20)
            except Exception:  # noqa: BLE001
                pass
            return None

        def _scan_checkpoints(self) -> None:
            if self.best_step is not None:
                from checkpoint_store import retain_best

                retain_best(run_dir, self.best_step * step_divisor)
            current = {_ckpt_step(p) // step_divisor: p for p in run_dir.glob(CKPT_GLOB)}
            for step, path in current.items():
                if step >= 0 and step not in self.known:
                    try:
                        size = path.stat().st_size
                    except OSError:
                        size = 0
                    _write_event(events, "checkpoint", step=step, path=str(path), size=size)
            for step in list(self.known):
                if step not in current:
                    _write_event(events, "checkpoint_removed", step=step)
            self.known = current

        def _pull_samples(self, trainer: Any) -> None:
            try:
                logger = trainer.logger
                writer = getattr(logger, "all_rank_experiment", None) or getattr(logger, "experiment", None)
                if writer is not None and hasattr(writer, "flush"):
                    writer.flush()
            except Exception:  # noqa: BLE001
                pass
            for step, files in self.samples.pull().items():
                _write_event(events, "samples", step=step, files=files)

        def _end(self, status: str, error: str | None = None, error_kind: str | None = None) -> None:
            if self.ended:
                return
            self.ended = True
            _write_event(events, "end", status=status, error=error, error_kind=error_kind)

        # ---- hooks ----
        def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
            try:
                previous = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                self.best_step = previous.get("best_val_step")
                self.best_loss = float(previous["best_val_loss"]) if previous.get("best_val_loss") is not None else float("inf")
            except (OSError, ValueError, TypeError):
                pass
            self.known = {_ckpt_step(p) // step_divisor: p for p in run_dir.glob(CKPT_GLOB)}
            _write_event(events, "start", step=int(trainer.global_step) // step_divisor, max_steps=int(trainer.max_steps or 0) // step_divisor)

        def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
            step = int(trainer.global_step) // step_divisor
            if self.pending_scan:
                # Lightning runs checkpoint callbacks after every other callback, so the checkpoint of the last
                # validation is only on disk now, at the next batch
                self.pending_scan = False
                self._scan_checkpoints()
                self._pull_samples(trainer)
            if step != self.last_step_event and (step % log_interval == 0 or step <= 5):
                self.last_step_event = step
                lr = None
                try:
                    lr = float(trainer.optimizers[0].param_groups[0]["lr"])
                except Exception:  # noqa: BLE001
                    pass
                losses = self._losses(trainer)
                losses.pop("lr", None)
                _write_event(events, "step", step=step, max_steps=int(trainer.max_steps or 0) // step_divisor,
                             epoch=int(trainer.current_epoch), losses=losses, lr=lr, gpu_mb=self._gpu_mb())
            if not self.stopping and stop_file.exists():
                self.stopping = True
                path = run_dir / f"model_ckpt_steps_{int(trainer.global_step)}.ckpt"
                try:
                    trainer.save_checkpoint(str(path))
                    _write_event(events, "checkpoint", step=step, path=str(path), size=path.stat().st_size)
                    self.known[step] = path
                except Exception as e:  # noqa: BLE001
                    _write_event(events, "warning", kind="checkpoint", text=f"Couldn't save a checkpoint on stop: {e}")
                trainer.should_stop = True

        def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
            if trainer.sanity_checking:
                return
            step = int(trainer.global_step) // step_divisor
            metrics: dict[str, float] = {}
            val_loss = None
            for key, value in (trainer.callback_metrics or {}).items():
                try:
                    f = float(value)
                except (TypeError, ValueError):
                    continue
                if key == "val_loss":
                    val_loss = f
                elif str(key).startswith(("val", "metrics")):
                    metrics[str(key)] = f
            _write_event(events, "validation", step=step, val_loss=val_loss, metrics=metrics)
            if val_loss is not None and math.isfinite(val_loss) and val_loss < self.best_loss:
                self.best_loss, self.best_step = val_loss, step
            self.pending_scan = True

        def on_train_end(self, trainer: Any, pl_module: Any) -> None:
            self._scan_checkpoints()
            self._pull_samples(trainer)
            self._end("stopped" if self.stopping else "done")

        def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
            text = f"{type(exception).__name__}: {exception}"
            oom = "out of memory" in text.lower()
            self._end("failed", error=text, error_kind="oom" if oom else "crash")

    return MvtEvents()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--log-interval", type=int, default=0, help="steps between step events (default: the config's log_interval, at least 10)")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    events = run_dir / "events.jsonl"
    config = run_dir / "config.yaml"
    if not config.exists():
        _write_event(events, "end", status="failed", error=f"No config.yaml in {run_dir}", error_kind="prepare")
        return 2

    os.chdir(SOURCE)
    os.environ["PYTHONPATH"] = str(SOURCE)
    sys.path.insert(0, str(SOURCE))
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # as scripts/train.py: prevents slowdowns with 16-bit precision
    # the same command line scripts/train.py would see
    sys.argv = [str(SOURCE / "scripts" / "train.py"), "--config", str(config), "--exp_name", str(run_dir)] + (["--reset"] if args.reset else [])

    try:
        from utils.hparams import hparams, set_hparams  # noqa: PLC0415

        set_hparams()
        from checkpoint_store import protect_removal
        from utils.training_utils import DsModelCheckpoint

        protect_removal(DsModelCheckpoint, run_dir)
        if not hparams.get("nccl_p2p", True):
            os.environ["NCCL_P2P_DISABLE"] = "1"
        interval = args.log_interval or max(10, int(hparams.get("log_interval") or 100) // 10)
        callback = make_callback(run_dir, interval)

        import lightning.pytorch as pl  # noqa: PLC0415
        import torch.utils.data as tud  # noqa: PLC0415

        # ds_workers: 0 (the default on Windows, where every worker process loads torch and the CUDA libraries
        # again): torch refuses persistent_workers and prefetch_factor without workers, which upstream always passes
        loader_init = tud.DataLoader.__init__

        def loader_init_safe(self: Any, *a: Any, **kw: Any) -> None:
            if not kw.get("num_workers"):
                kw.pop("persistent_workers", None)
                kw["prefetch_factor"] = None
            loader_init(self, *a, **kw)

        tud.DataLoader.__init__ = loader_init_safe  # type: ignore[method-assign]

        original_init = pl.Trainer.__init__

        def init_with_events(self: Any, *a: Any, **kw: Any) -> None:
            callbacks = list(kw.get("callbacks") or [])
            callbacks.append(callback)
            kw["callbacks"] = callbacks
            original_init(self, *a, **kw)

        pl.Trainer.__init__ = init_with_events  # type: ignore[method-assign]

        import importlib  # noqa: PLC0415

        task_cls_name = hparams["task_cls"]
        pkg = ".".join(task_cls_name.split(".")[:-1])
        task_cls = getattr(importlib.import_module(pkg), task_cls_name.split(".")[-1])
        task_cls.start()
        callback._end("stopped" if callback.stopping else "done")
        return 0
    except KeyboardInterrupt:
        _write_event(events, "end", status="stopped", error="Interrupted")
        return 130
    except BaseException as e:  # noqa: BLE001
        text = f"{type(e).__name__}: {e}"
        lowered = text.lower()
        kind = "oom" if "out of memory" in lowered else "crash"
        traceback.print_exc()
        _write_event(events, "end", status="failed", error=text, error_kind=kind, trace=traceback.format_exc(limit=12))
        return 1


if __name__ == "__main__":
    sys.exit(main())

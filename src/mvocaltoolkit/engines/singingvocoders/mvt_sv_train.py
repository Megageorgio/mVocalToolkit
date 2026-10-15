"""Detached trainer wrapper; uses upstream training unchanged and reports real GAN steps (two optimizers)."""
import argparse
import importlib.util
import os
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    folder = Path(parser.parse_args().run_dir).resolve()
    helper_path = Path(__file__).resolve().parent.parent / "diffsinger" / "mvt_ds_train.py"
    sys.path.insert(0, str(helper_path.parent))
    spec = importlib.util.spec_from_file_location("mvt_training_events", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    events = folder / "events.jsonl"
    callback = None
    try:
        import lightning.pytorch as pl
        import torch.utils.data as tud

        original_loader = tud.DataLoader.__init__
        def loader(self, *args, **kwargs):
            if not kwargs.get("num_workers"):
                kwargs.pop("persistent_workers", None)
                kwargs["prefetch_factor"] = None
            original_loader(self, *args, **kwargs)
        tud.DataLoader.__init__ = loader
        callback = helper.make_callback(folder, 10, step_divisor=2, sample_log_dir=folder / "lightning_logs" / "lastest")
        from checkpoint_store import protect_removal
        from utils.training_utils import DsModelCheckpoint

        protect_removal(DsModelCheckpoint, folder)
        original_trainer = pl.Trainer.__init__
        def trainer(self, *args, **kwargs):
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), callback]
            original_trainer(self, *args, **kwargs)
        pl.Trainer.__init__ = trainer
        source = Path(os.environ["MVT_SOURCE_DIR"]).resolve()
        sys.path.insert(0, str(source))
        sys.argv = [str(source / "train.py"), "--config", str(folder / "config.yaml"),
                    "--exp_name", folder.name, "--work_dir", str(folder.parent)]
        runpy.run_path(str(source / "train.py"), run_name="__main__")
        callback._end("stopped" if callback.stopping else "done")
    except SystemExit as exc:
        if exc.code in (None, 0) and callback is not None:
            callback._end("stopped" if callback.stopping else "done")
        else:
            helper._write_event(events, "end", status="failed", error=f"Trainer exited: {exc.code}", error_kind="crash")
            raise
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        text = f"{type(exc).__name__}: {exc}"
        helper._write_event(events, "end", status="failed", error=text,
                            error_kind="oom" if "out of memory" in text.lower() else "crash")
        raise


if __name__ == "__main__":
    main()

"""A fake trainer: writes the events a real one would, honours the STOP file, resumes from its last checkpoint."""

import argparse
import json
import sys
import time
from pathlib import Path


def write(events, event, **data):
    with events.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, "time": time.time(), **data}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run = Path(args.run_dir)
    events = run / "events.jsonl"
    config = json.loads((run / "config.yaml").read_text(encoding="utf-8"))
    max_steps = config["max_updates"]
    val_every = config["val_check_interval"]
    ckpts = sorted(run.glob("model_ckpt_steps_*.ckpt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    step = 0 if args.reset or not ckpts else int(ckpts[-1].stem.rsplit("_", 1)[1])
    print(f"fake trainer from step {step}", flush=True)
    write(events, "start", step=step, max_steps=max_steps)
    stopping = False
    while step < max_steps:
        step += 1
        time.sleep(config.get("step_seconds", 0.02))
        if config.get("fail_at") and step >= int(config["fail_at"]):
            print("CUDA out of memory", flush=True)
            write(events, "end", status="failed", error="RuntimeError: CUDA out of memory", error_kind="oom")
            return 1
        if step % 5 == 0 or step <= 2:
            write(events, "step", step=step, max_steps=max_steps, epoch=step // 10, losses={"total_loss": 1.0 / step}, lr=0.001, gpu_mb=1234)
        if (run / "STOP").exists():
            stopping = True
        if step % val_every == 0 or stopping:
            write(events, "validation", step=step, val_loss=2.0 / step, metrics={})
            ckpt = run / f"model_ckpt_steps_{step}.ckpt"
            ckpt.write_text("fake", encoding="utf-8")
            write(events, "checkpoint", step=step, path=str(ckpt), size=4)
            sample = run / "samples" / str(step)
            sample.mkdir(parents=True, exist_ok=True)
            (sample / "diff_0.wav").write_bytes(b"RIFF")
            write(events, "samples", step=step, files=[f"samples/{step}/diff_0.wav"])
        if stopping:
            write(events, "end", status="stopped")
            return 0
    write(events, "end", status="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

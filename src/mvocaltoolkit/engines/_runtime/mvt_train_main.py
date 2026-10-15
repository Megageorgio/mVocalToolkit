"""Entry point shared by isolated training engines."""
import argparse
import json
import os
import sys
from pathlib import Path

from mvt_train_common import Stopped, emit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "train", "export"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--device", default=os.environ.get("MVT_DEVICE", "auto"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--step", type=int)
    args = parser.parse_args(); run = args.run.resolve()
    source = Path(os.environ["MVT_SOURCE_DIR"]).resolve()
    sys.path.insert(0, str(source)); sys.path.insert(0, str(source / "src"))
    try:
        if args.action == "prepare":
            from mvt_train_data import prepare
            prepare(run, source, args.kind, args.device)
        elif args.action == "export":
            from mvt_train_export import export
            export(run, source, args.kind, args.out.resolve(), args.step)
        else:
            if args.kind in {"tifa_aligner", "sofa_aligner", "game_notes"}:
                from mvt_lightning import train
            elif args.kind in {"wfl_asr_teacher", "mcfl_student", "refiner_boundary"}:
                from mvt_label_train import train
            elif args.kind == "qwen_tts_voice":
                from mvt_qwen_train import train
            elif args.kind == "voxcpm_voice":
                from mvt_voxcpm_train import train
            else: raise ValueError("Unknown training kind")
            train(run, source, args.kind, args.device, args.resume)
            emit(run, "end", status="done")
    except Stopped:
        emit(run, "end", status="stopped")
    except Exception as exc:
        if args.action == "train":
            emit(run, "end", status="failed", error=str(exc), error_kind="oom" if "out of memory" in str(exc).lower() else "crash")
        raise


if __name__ == "__main__": main()

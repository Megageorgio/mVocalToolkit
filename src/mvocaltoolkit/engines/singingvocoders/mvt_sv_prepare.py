"""Sequential preprocessing: bounded RAM, immutable audio snapshots, deterministic held-out recordings."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    folder = Path(parser.parse_args().run_dir).resolve()
    import numpy as np
    import soundfile as sf
    import yaml
    from process import wav2spec

    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((folder / "source-report.json").read_text(encoding="utf-8"))
    recordings = sorted(report["recordings"], key=lambda r: r["sha256"])
    binary = folder / "binary"
    audio_dir = folder / "audio"
    binary.mkdir(exist_ok=True); audio_dir.mkdir(exist_ok=True)
    paths = []
    for index, item in enumerate(recordings):
        source = Path(item["file"])
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Recording changed since the check: {source}")
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        if rate != 44100 or not np.isfinite(audio).all():
            raise ValueError(f"Expected finite 44.1 kHz audio: {source}")
        snapshot = audio_dir / f"{digest}.wav"
        sf.write(snapshot, audio.mean(axis=1), rate, subtype="PCM_24")
        destination = binary / f"{digest}.npz"
        okay, result = wav2spec(config, snapshot, destination)
        if not okay:
            raise ValueError(f"Could not prepare {source}: {result}")
        paths.append(destination.as_posix())
        print("MVT_PROGRESS " + json.dumps({"fraction": (index + 1) / len(recordings),
              "message": f"Prepared recording {index + 1} / {len(recordings)}"}), flush=True)
    validation_count = min(3, max(1, len(paths) // 10))
    if len(paths) < 2:
        raise ValueError("At least two recordings must survive preparation")
    for name, values in (("valid", paths[:validation_count]), ("train", paths[validation_count:])):
        (binary / name).write_text("\n".join(values) + "\n", encoding="utf-8")
    report["validation"] = [recordings[i]["sha256"] for i in range(validation_count)]
    report["training"] = [r["sha256"] for r in recordings[validation_count:]]
    (folder / "dataset-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

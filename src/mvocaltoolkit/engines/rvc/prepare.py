"""Sequential, bounded RVC feature preparation; no app UI or source-tree writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def progress(fraction, message):
    print("MVT_PROGRESS " + json.dumps({"fraction": fraction, "message": message}), flush=True)


def slice_recordings(run: Path, config: dict, recordings: list) -> list[Path]:
    import numpy as np
    import soundfile as sf
    import soxr
    from scipy.signal import butter, sosfilt
    rate = config["sample_rate"]
    target = run / "sliced_audios"
    target16 = run / "sliced_audios_16k"
    target.mkdir(exist_ok=True); target16.mkdir(exist_ok=True)
    output = []
    for index, recording in enumerate(recordings):
        source = Path(recording["file"])
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != recording["sha256"]:
                raise ValueError(f"Recording changed after preflight: {source}")
            stream.seek(0)
            # Decode from the verified handle; stream long recordings in chunks to bound memory.
            with sf.SoundFile(stream) as audio:
                source_rate = audio.samplerate
                length = round(config["chunk_seconds"] * source_rate)
                stride = round((config["chunk_seconds"] - config["overlap_seconds"]) * source_rate)
                for part, start in enumerate(range(0, len(audio), stride)):
                    audio.seek(start)
                    chunk = audio.read(length, dtype="float32", always_2d=True).mean(axis=1)
                    if len(chunk) < .8 * source_rate:
                        continue
                    chunk = soxr.resample(chunk, source_rate, rate).astype(np.float32)
                    if not np.isfinite(chunk).all():
                        raise ValueError(f"Non-finite audio: {source}")
                    chunk = sosfilt(butter(5, 48, fs=rate, output="sos"), chunk).astype(np.float32)
                    peak = float(np.max(np.abs(chunk)))
                    if peak < 1e-5:
                        continue
                    # Match the RVC blended peak normalization without amplifying silence.
                    chunk *= .75 * .9 / peak + .25
                    name = f"0_{index}_{part}.wav"
                    sf.write(target / name, chunk, rate, subtype="FLOAT")
                    sf.write(target16 / name, soxr.resample(chunk, rate, 16000), 16000, subtype="FLOAT")
                    output.append(target / name)
        progress(.2 * (index + 1) / len(recordings), f"Audio {index + 1}/{len(recordings)}")
    if len(output) < 3 * config["batch_size"]:
        raise ValueError("Not enough usable chunks for three training batches; add recordings or reduce batch size")
    return output


def features(run: Path, config: dict, paths: list[Path], device: str):
    import numpy as np
    import pyworld
    import soundfile as sf
    import torch
    from rvc.lib.utils import load_embedding
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_embedding("custom", config["embedder"]).to(device).float().eval()
    for name in ("f0", "f0_voiced", "extracted"):
        (run / name).mkdir(exist_ok=True)
    rows = []
    for index, path in enumerate(paths):
        audio, _ = sf.read(run / "sliced_audios_16k" / path.name, dtype="float32")
        f0, time_axis = pyworld.harvest(audio.astype(np.float64), 16000, f0_floor=50, f0_ceil=1100, frame_period=10)
        f0 = pyworld.stonemask(audio.astype(np.float64), f0, time_axis, 16000)
        mel = 1127 * np.log1p(f0 / 700)
        coarse = np.rint(np.clip((mel - 1127 * math.log1p(50 / 700)) * 254 /
                                (1127 * math.log1p(1100 / 700) - 1127 * math.log1p(50 / 700)) + 1, 1, 255)).astype(np.int64)
        with torch.inference_mode():
            content = model(torch.from_numpy(audio).to(device).unsqueeze(0))["last_hidden_state"].squeeze(0).float().cpu().numpy()
        if content.ndim != 2 or content.shape[1] != 768 or not np.isfinite(content).all():
            raise ValueError("Expected finite ContentVec features with 768 channels")
        pitch = run / "f0" / (path.name + ".npy")
        pitchf = run / "f0_voiced" / (path.name + ".npy")
        feature = run / "extracted" / (path.stem + ".npy")
        np.save(pitch, coarse, allow_pickle=False)
        np.save(pitchf, f0.astype(np.float32), allow_pickle=False)
        np.save(feature, content, allow_pickle=False)
        rows.append("|".join(map(str, (path, feature, pitch, pitchf, 0))))
        progress(.2 + .7 * (index + 1) / len(paths), f"Pitch and features {index + 1}/{len(paths)}")
    (run / "filelist.txt").write_text("\n".join(rows), encoding="utf-8")


def build_index(run: Path, limit: int, seed: int):
    if not limit:
        return
    import faiss
    import numpy as np
    files = sorted((run / "extracted").glob("*.npy"))
    lengths = [len(np.load(p, mmap_mode="r")) for p in files]
    total = sum(lengths)
    if not total:
        raise ValueError("No content features to index")
    # Uniform sampling across the corpus, with a fixed memory bound.
    chosen = np.sort(np.random.default_rng(seed).choice(total, min(limit, total), replace=False))
    vectors = np.empty((len(chosen), 768), dtype=np.float32)
    offset, position = 0, 0
    for file, length in zip(files, lengths):
        selection = chosen[(chosen >= offset) & (chosen < offset + length)] - offset
        if len(selection):
            vectors[position:position + len(selection)] = np.load(file, mmap_mode="r")[selection]
            position += len(selection)
        offset += length
    index = faiss.IndexFlatL2(768)
    index.add(vectors)
    faiss.write_index(index, str(run / "voice.index"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    import yaml
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    data = json.loads((run / "recordings.json").read_text(encoding="utf-8"))
    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("The selected CUDA device is unavailable")
    if device == "mps":
        raise ValueError("This RVC trainer supports CPU or CUDA")
    paths = slice_recordings(run, config, data["recordings"])
    features(run, config, paths, device)
    build_index(run, config["index_vectors"], config["seed"])
    source = Path(os.environ["MVT_SOURCE_DIR"])
    native = json.loads((source / "rvc" / "configs" / f"{config['sample_rate']}.json").read_text())
    native["train"].update(learning_rate=config["learning_rate"], seed=config["seed"])
    native["data"]["training_files"] = str(run / "filelist.txt")
    native["model"]["spk_embed_dim"] = 109 if config.get("base_generator") else 1
    (run / "config.json").write_text(json.dumps(native, indent=2), encoding="utf-8")
    (run / "model_info.json").write_text(json.dumps({"embedder_model": "contentvec", "speakers_id": 1,
                                                    "total_dataset_duration": data["seconds"]}), encoding="utf-8")
    (run / "prepared.json").write_text(json.dumps({"chunks": len(paths), "index": (run / "voice.index").exists()}), encoding="utf-8")
    progress(1, f"Prepared {len(paths)} chunks")


if __name__ == "__main__":
    main()

"""GAME engine worker: ONNX inference of the released GAME models (encoder -> segmenter -> estimator).

Mirrors GAME's `infer.py extract`: the audio is sliced at silences (GAME's Slicer), each slice goes through
encoder.onnx, segmenter.onnx (D3PM sampling loop if the model has one), estimator.onnx; notes of the slices are
merged. Model folder: config.json + encoder/segmenter/estimator/dur2bd/bd2dur .onnx (see GAME's ONNX.md).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import mvt_engine as rt

_models: dict[str, "GameModel"] = {}

DEFAULTS = {"seg_threshold": 0.2, "seg_radius": 0.02, "t0": 0.0, "nsteps": 8, "est_threshold": 0.2}


def _providers() -> list[str]:
    import onnxruntime as ort  # noqa: PLC0415

    available = ort.get_available_providers()
    if (os.environ.get("MVT_DEVICE") or "auto").lower() == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:  # noqa: BLE001
            rt.log(f"Could not preload CUDA libraries: {e}")
    order = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in available] or ["CPUExecutionProvider"]


class GameModel:
    def __init__(self, root: Path, layout: dict[str, Any]):
        import onnxruntime as ort  # noqa: PLC0415

        self.root = root
        self.config = json.loads((root / layout.get("config", "config.json")).read_text(encoding="utf-8"))
        self.samplerate = int(self.config["samplerate"])
        self.timestep = float(self.config["timestep"])
        self.languages: dict[str, int] | None = self.config.get("languages")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = _providers()
        self.sessions = {
            name: ort.InferenceSession(str(root / layout.get(name, f"{name}.onnx")), options, providers=providers)
            for name in ("encoder", "segmenter", "estimator", "dur2bd", "bd2dur")
        }
        self.segmenter_inputs = {i.name for i in self.sessions["segmenter"].get_inputs()}
        rt.log(f"GAME providers: {self.sessions['encoder'].get_providers()}")

    def run(self, name: str, feeds: dict[str, Any]) -> dict[str, Any]:
        session = self.sessions[name]
        wanted = {i.name for i in session.get_inputs()}
        outputs = [o.name for o in session.get_outputs()]
        return dict(zip(outputs, session.run(outputs, {k: v for k, v in feeds.items() if k in wanted})))

    def language_id(self, language: str | None) -> int:
        if not self.languages or not language:
            return 0
        if language in self.languages:
            return int(self.languages[language])
        base = language.lower().split("_")[0].split("-")[0]
        return int(next((v for k, v in self.languages.items() if k.lower() == base), 0))

    def infer_chunk(self, waveform, language_id: int, opts: dict[str, Any]):
        """-> (durations, presence, scores) for one slice."""
        import numpy as np  # noqa: PLC0415

        duration = np.array([len(waveform) / self.samplerate], dtype=np.float32)
        enc = self.run("encoder", {"waveform": waveform[None, :].astype(np.float32), "duration": duration})
        x_seg, x_est, mask_t = enc["x_seg"], enc["x_est"], enc["maskT"]
        known = self.run("dur2bd", {"durations": duration[None, :], "maskT": mask_t})["boundaries"]
        known = np.logical_and(known, mask_t)
        feeds: dict[str, Any] = {
            "x_seg": x_seg,
            "known_boundaries": known,
            "maskT": mask_t,
            "language": np.array([language_id], dtype=np.int64),
            "threshold": np.array(opts["seg_threshold"], dtype=np.float32),
            "radius": np.array(max(1, round(opts["seg_radius"] / self.timestep)), dtype=np.int64),
        }
        if "t" in self.segmenter_inputs:
            nsteps = int(opts["nsteps"])
            t0 = float(opts["t0"])
            step = (1 - t0) / nsteps
            boundaries = known
            for i in range(nsteps):
                boundaries = self.run("segmenter", {
                    **feeds, "prev_boundaries": boundaries, "t": np.array([t0 + i * step], dtype=np.float32),
                })["boundaries"]
        else:
            boundaries = self.run("segmenter", feeds)["boundaries"]
        conv = self.run("bd2dur", {"boundaries": boundaries, "maskT": mask_t})
        durations, mask_n = conv["durations"], conv["maskN"]
        est = self.run("estimator", {
            "x_est": x_est, "boundaries": boundaries, "maskT": mask_t, "maskN": mask_n,
            "threshold": np.array(opts["est_threshold"], dtype=np.float32),
        })
        return durations[0], est["presence"][0], est["scores"][0]


def _get_model(model: dict[str, Any]) -> GameModel:
    key = model["path"]
    if key not in _models:
        _models.clear()
        rt.progress(message="Loading GAME model")
        _models[key] = GameModel(Path(model["path"]), model.get("layout", {}))
    return _models[key]


def _slices(waveform, samplerate: int) -> list[dict[str, Any]]:
    from inference.slicer2 import Slicer  # noqa: PLC0415  (GAME)

    slicer = Slicer(sr=samplerate, threshold=-40.0, min_length=1000, min_interval=200, max_sil_kept=100)
    return slicer.slice(waveform)


def extract_notes(loaded: GameModel, audio: str, language: str | None, opts: dict[str, Any]) -> list[dict]:
    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    waveform, _ = librosa.load(audio, sr=loaded.samplerate, mono=True)
    language_id = loaded.language_id(language)
    chunks = _slices(waveform, loaded.samplerate)
    notes: list[list[float]] = []
    for index, chunk in enumerate(chunks):
        rt.progress(index / max(1, len(chunks)), f"Extracting notes {index + 1}/{len(chunks)}")
        wav = chunk["waveform"]
        length = len(wav) / loaded.samplerate
        offset = float(chunk["offset"])
        durations, presence, scores = loaded.infer_chunk(wav, language_id, opts)
        ends = np.minimum(np.cumsum(durations), length)
        starts = np.concatenate([[0.0], ends[:-1]])
        for start, end, pitch, voiced in zip(starts, ends, scores, presence):
            if end - start > 0 and bool(voiced):
                notes.append([float(start) + offset, float(end) + offset, float(pitch)])
    # same clean-up as GAME: sorted, no overlaps
    notes.sort()
    result, last = [], 0.0
    for start, end, pitch in notes:
        start = max(start, last)
        if end <= start:
            continue
        result.append({"start": round(start, 4), "end": round(end, 4), "pitch": pitch})
        last = end
    return result


def _pitch_text(pitch: float, pitch_format: str, round_pitch: bool) -> str:
    import librosa  # noqa: PLC0415

    if pitch_format == "name":
        return librosa.midi_to_note(round(pitch) if round_pitch else pitch, unicode=False, cents=not round_pitch)
    return str(round(pitch)) if round_pitch else f"{pitch:.3f}"


def _write(notes: list[dict], out: Path, stem: str, fmt: str, tempo: float, pitch_format: str,
           round_pitch: bool) -> Path:
    path = out / f"{stem}.{fmt}"
    if fmt == "mid":
        import mido  # noqa: PLC0415

        track = mido.MidiTrack()
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
        last = 0
        # 480 ticks per beat (mido default): seconds * tempo / 60 * 480 = seconds * tempo * 8
        for note in notes:
            on, off = round(note["start"] * tempo * 8), round(note["end"] * tempo * 8)
            if off <= on:
                continue
            pitch = min(127, max(0, round(note["pitch"])))
            track.append(mido.Message("note_on", note=pitch, time=on - last))
            track.append(mido.Message("note_off", note=pitch, time=off - on))
            last = off
        midi = mido.MidiFile(charset="utf8")
        midi.tracks.append(track)
        midi.save(path)
    elif fmt in ("csv", "txt"):
        with open(path, "w", encoding="utf-8", newline="") as f:
            if fmt == "csv":
                writer = csv.writer(f)
                writer.writerow(["onset", "offset", "pitch"])
            for note in notes:
                row = [f"{note['start']:.3f}", f"{note['end']:.3f}",
                       _pitch_text(note["pitch"], pitch_format, round_pitch)]
                if fmt == "csv":
                    writer.writerow(row)
                else:
                    f.write("\t".join(row) + "\n")
    else:
        raise ValueError(f"Unknown output format {fmt!r} (mid, csv, txt)")
    return path


@rt.method()
def extract(
    model: dict[str, Any],
    audio: str,
    output_dir: str,
    language: str | None = None,
    tempo: float = 120.0,
    options: dict[str, Any] | None = None,
    output_formats: list[str] | None = None,
    pitch_format: str = "name",
    round_pitch: bool = False,
) -> dict[str, Any]:
    opts = {**DEFAULTS, **{k: v for k, v in (options or {}).items() if v is not None}}
    try:
        loaded = _get_model(model)
        notes = extract_notes(loaded, audio, language, opts)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for fmt in output_formats or ["mid", "csv"]:
            files[fmt] = str(_write(notes, out, Path(audio).stem, fmt, tempo, pitch_format, round_pitch))
        for note in notes:
            note["note"] = _pitch_text(note["pitch"], pitch_format, round_pitch)
        rt.progress(1.0, "Done")
        return {"ok": True, "notes": notes, "files": files}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@rt.method()
def model_info(model: dict[str, Any]) -> dict[str, Any]:
    loaded = _get_model(model)
    return {"samplerate": loaded.samplerate, "timestep": loaded.timestep, "languages": loaded.languages,
            "loop": "t" in loaded.segmenter_inputs}


if __name__ == "__main__":
    rt.run()

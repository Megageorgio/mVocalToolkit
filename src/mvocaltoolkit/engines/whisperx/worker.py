"""WhisperX engine worker: batched faster-whisper with VAD."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import mvt_engine as rt

warnings.filterwarnings("ignore")

_pipelines: dict[tuple, Any] = {}
SAMPLE_RATE = 16000


def _load_audio(path: str):
    """Loads audio as 16 kHz mono float32 without requiring ffmpeg."""
    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def _pipeline(model: str, compute_type: str, vad: dict[str, Any], asr_options: dict[str, Any], device: str):
    import whisperx  # noqa: PLC0415

    key = (model, compute_type, device, tuple(sorted(vad.items())), tuple(sorted((k, str(v)) for k, v in asr_options.items())))
    if key in _pipelines:
        return _pipelines[key]
    # keep only one recognizer in memory
    _pipelines.clear()
    rt.free_memory()
    rt.progress(message=f"Loading Whisper {model}")
    kwargs: dict[str, Any] = {
        "compute_type": compute_type,
        "asr_options": asr_options,
        "download_root": os.path.join(os.environ.get("MVT_CACHE", "."), "whisper"),
    }
    if vad.get("enabled", True):
        kwargs["vad_method"] = vad.get("method", "silero")
        kwargs["vad_options"] = {
            "vad_onset": vad.get("onset", 0.5),
            "vad_offset": vad.get("offset", 0.363),
            "chunk_size": vad.get("chunk_size", 30),
        }
    whisper_device = "cuda" if device.startswith("cuda") else "cpu"
    try:
        pipeline = whisperx.load_model(model, whisper_device, **kwargs)
    except TypeError:
        # older whisperx without vad_method / some asr options
        kwargs.pop("vad_method", None)
        pipeline = whisperx.load_model(model, whisper_device, **kwargs)
    _pipelines[key] = pipeline
    return pipeline


@rt.method()
def transcribe(
    items: list[dict[str, Any]],
    model: str = "large-v3-turbo",
    language: str | None = None,
    batch_size: int = 8,
    compute_type: str = "auto",
    vad: dict[str, Any] | None = None,
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    suppress_numerals: bool = True,
    beam_size: int = 5,
    temperature: float = 0.0,
    model_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    device = rt.device()
    if compute_type == "auto":
        compute_type = "float16" if device.startswith("cuda") else "int8"
    asr_options: dict[str, Any] = {
        "beam_size": beam_size,
        "temperatures": [temperature],
        "suppress_numerals": suppress_numerals,
    }
    if initial_prompt:
        asr_options["initial_prompt"] = initial_prompt
    if hotwords:
        asr_options["hotwords"] = hotwords
    pipeline = _pipeline(model, compute_type, vad or {}, asr_options, device)
    results = []
    total = len(items)
    for index, item in enumerate(items):
        name = item.get("name") or Path(item["audio"]).stem
        rt.progress(index / total, f"Transcribing {name}")
        try:
            audio = _load_audio(item["audio"])
            result = _transcribe_with_retry(pipeline, audio, batch_size, language)
            segments = [
                {"start": float(s.get("start", 0)), "end": float(s.get("end", 0)), "text": s.get("text", "").strip()}
                for s in result.get("segments", [])
            ]
            text = " ".join(s["text"] for s in segments if s["text"]).strip()
            speech = sum(s["end"] - s["start"] for s in segments)
            duration = len(audio) / SAMPLE_RATE
            results.append(
                {
                    "ok": True,
                    "text": text,
                    "language": result.get("language") or language,
                    "segments": segments,
                    "duration": duration,
                    # nothing recognized in a long file, or very little speech: worth a manual check
                    "low_confidence": not text or (duration > 3 and speech / max(duration, 1e-6) < 0.15),
                }
            )
        except Exception as e:  # noqa: BLE001
            results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
    rt.progress(1.0, "Transcribed")
    return results


def _transcribe_with_retry(pipeline, audio, batch_size: int, language: str | None):
    size = max(1, int(batch_size))
    while True:
        try:
            return pipeline.transcribe(audio, batch_size=size, language=language, print_progress=False)
        except TypeError:
            return pipeline.transcribe(audio, batch_size=size, language=language)
        except Exception as e:  # noqa: BLE001
            if rt.is_oom(e) and size > 1:
                rt.free_memory()
                size = max(1, size // 2)
                rt.log(f"Out of GPU memory, retrying with batch size {size}")
                continue
            raise


@rt.method()
def detect_language(audio: str, model: str = "large-v3-turbo") -> dict[str, Any]:
    device = rt.device()
    pipeline = _pipeline(model, "float16" if device.startswith("cuda") else "int8", {}, {}, device)
    data = _load_audio(audio)[: SAMPLE_RATE * 30]
    return {"language": pipeline.detect_language(data)}


if __name__ == "__main__":
    rt.run()

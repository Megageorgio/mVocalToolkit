"""Cheap, bounded-memory checks of labelled training recordings. No model or GPU is loaded."""
from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

from ..labels import Label


def inspect_recording(path: Path, label: Label) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

    def finding(code: str, text: str, severity: str = "warning", **details: Any) -> None:
        findings.append({"file": str(path), "code": code, "severity": severity, "problem": text, **details})

    duration = None
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
                frames = wav.getnframes()
                duration = frames / rate
                metadata.update(sample_rate=rate, channels=channels, bit_depth=width * 8, seconds=duration)
                if channels != 1:
                    finding("channels", f"{channels} channels; training audio will be mixed to mono", "note")
                if rate != 44100:
                    finding("sample_rate", f"{rate} Hz; training audio will be resampled to 44100 Hz", "note")
                if not frames:
                    finding("empty_audio", "audio has no frames", "error")
                # Inspect at most about 8k samples/second, streaming blocks instead of loading a song in RAM.
                count = clipped = 0
                total = squares = 0.0
                frame_size = channels * width
                stride = max(1, rate // 8000) * frame_size
                read_frames = 0
                while block := wav.readframes(65536):
                    read_frames += len(block) // frame_size
                    for offset in range(0, len(block) - frame_size + 1, stride):
                        for channel in range(channels):
                            start = offset + channel * width
                            raw = block[start:start + width]
                            value = (raw[0] - 128) / 128 if width == 1 else int.from_bytes(raw, "little", signed=True) / (1 << (width * 8 - 1))
                            total += value
                            squares += value * value
                            clipped += abs(value) >= 0.999
                            count += 1
                if read_frames != frames:
                    finding("truncated_audio", "audio data ends before the WAV header's frame count", "error")
                if count:
                    rms = math.sqrt(squares / count)
                    metadata.update(rms_db=20 * math.log10(max(rms, 1e-10)), dc_offset=total / count, clipped_fraction=clipped / count)
                    if clipped / count > 0.001:
                        finding("clipping", f"possible clipping in {clipped / count:.1%} of inspected samples")
                    if abs(total / count) > 0.02:
                        finding("dc_offset", "large DC offset", value=round(total / count, 4))
                    if rms < 0.003:
                        finding("low_level", "very low recording level (below about -50 dB RMS)")
        except (OSError, EOFError, wave.Error, ValueError, ZeroDivisionError) as exc:
            # Non-PCM WAV formats may be supported by the engine's decoder. Keep the limit explicit.
            unsupported = "unknown format" in str(exc).lower()
            finding("audio_unchecked" if unsupported else "invalid_audio", f"PCM WAV inspection unavailable: {exc}", "warning" if unsupported else "error")
    else:
        finding("audio_unchecked", "audio signal checks currently cover PCM WAV; FLAC is checked during preparation", "note")

    if duration is not None and duration > 30:
        finding("long_recording", "recording is longer than 30 seconds; splitting it reduces training memory use")
    phones = label.phones
    cursor = 0.0
    short = 0
    for interval in phones:
        if not math.isfinite(interval.start) or not math.isfinite(interval.end) or interval.start < 0 or interval.end <= interval.start:
            finding("invalid_interval", "phoneme has invalid or non-finite times", "error", start=str(interval.start), end=str(interval.end))
            continue
        if interval.start < cursor - 0.001:
            finding("overlap", "phoneme intervals overlap or are out of order", "error", start=interval.start)
        elif interval.start > cursor + 0.02:
            finding("label_gap", "gap in phoneme labels; add an explicit SP interval", "error", start=cursor, end=interval.start)
        if interval.duration < 0.02 and interval.text not in {"AP", "SP"}:
            short += 1
        cursor = max(cursor, interval.end)
    if short:
        finding("short_phonemes", f"{short} phonemes shorter than 20 ms", count=short)
    if duration is not None and cursor > duration + 0.03:
        finding("labels_past_audio", "phoneme labels extend beyond the recording", "error", end=cursor, seconds=duration)
    elif duration is not None and duration - cursor > 0.05:
        finding("unlabelled_tail", "phoneme labels do not cover the end of the recording; add SP", "error", start=cursor, end=duration)
    if phones and (phones[0].text != "SP" or phones[-1].text != "SP"):
        finding("edge_silence", "no explicit SP at the start or end")
    return findings, metadata

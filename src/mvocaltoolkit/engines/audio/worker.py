"""Small audio operations, isolated from training and inference dependencies."""
from pathlib import Path

import mvt_engine as rt


@rt.method()
def slice_audio(audio: str, slices: list[dict]) -> dict:
    import soundfile as sf

    files = []
    with sf.SoundFile(audio) as source:
        for item in slices:
            start = max(0, round(float(item["start"]) * source.samplerate))
            end = min(len(source), round(float(item["end"]) * source.samplerate))
            if end <= start:
                raise ValueError("An audio slice must contain at least one frame")
            target = Path(item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"Output already exists: {target}")
            source.seek(start)
            data = source.read(end - start, dtype="float64", always_2d=True)
            # Requested by the preparation planner only within labelled silence.
            import numpy as np
            for key, tail in (("fade_in", False), ("fade_out", True)):
                frames = min(len(data) // 2, max(0, round(float(item.get(key, 0)) * source.samplerate)))
                if frames:
                    ramp = np.linspace(0.0, 1.0, frames)[:, None]
                    if tail:
                        data[-frames:] *= ramp[::-1]
                    else:
                        data[:frames] *= ramp
            subtype = source.subtype if source.subtype in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"} else "PCM_24"
            before = max(0, round(float(item.get("pad_before", 0)) * source.samplerate))
            after = max(0, round(float(item.get("pad_after", 0)) * source.samplerate))
            if before or after:
                data = np.pad(data, ((before, after), (0, 0)))
            sf.write(target, data, source.samplerate, subtype=subtype)
            files.append(str(target))
    return {"files": files}


@rt.method()
def inspect_audio(audio: str, options: dict | None = None) -> dict:
    """Bounded spectral/noise/tail heuristics; findings are advisory, never a reverb diagnosis."""
    import numpy as np
    import soundfile as sf
    opts = options or {}
    levels, rolloffs, tail_ratios = [], [], []
    with sf.SoundFile(audio) as source:
        rate = source.samplerate
        size = max(256, min(rate // 4, 16384))
        # Deterministic samples spread over the file, at most 128 windows.
        starts = np.linspace(0, max(0, len(source) - size), min(128, max(1, len(source) // size)), dtype=int)
        for start in starts:
            source.seek(int(start))
            data = source.read(size, dtype="float64", always_2d=True).mean(axis=1)
            if not np.all(np.isfinite(data)):
                raise ValueError("Audio contains non-finite samples")
            if not len(data):
                continue
            power = float(np.mean(data ** 2))
            levels.append(10 * np.log10(max(power, 1e-12)))
            spectrum = np.abs(np.fft.rfft(data * np.hanning(len(data)))) ** 2
            if power > 1e-7 and spectrum.sum() > 0:
                rolloffs.append(float(np.searchsorted(np.cumsum(spectrum), spectrum.sum() * .99) * rate / len(data)))
            mid = len(data) // 2
            if mid and power > 1e-7:
                tail_ratios.append(float(np.mean(data[mid:] ** 2) / max(np.mean(data[:mid] ** 2), 1e-12)))
    metrics = dict(noise_floor_estimate_db=float(np.percentile(levels, 10)) if levels else -120,
        bandwidth_99_hz=float(np.median(rolloffs)) if rolloffs else 0,
        tail_energy_ratio=float(np.median(tail_ratios)) if tail_ratios else 0, sampled_windows=len(levels))
    problems = []
    if metrics["noise_floor_estimate_db"] > float(opts.get("noise_warn_db", -40)):
        problems.append(dict(code="noise_estimate", severity="warning", problem="Quiet-window level is high; inspect noise or continuous singing", value=metrics["noise_floor_estimate_db"]))
    if rolloffs and metrics["bandwidth_99_hz"] < float(opts.get("bandwidth_warn_hz", 6000)):
        problems.append(dict(code="bandwidth", severity="warning", problem="Limited estimated high-frequency bandwidth; inspect the recording", value=metrics["bandwidth_99_hz"]))
    if opts.get("reverb_hint", False) and metrics["tail_energy_ratio"] > float(opts.get("tail_warn_ratio", 1.5)):
        problems.append(dict(code="tail_energy", severity="note", problem="High late-window energy; inspect sustained singing/reverb. This metric cannot identify reverb reliably."))
    return dict(metrics=metrics, problems=problems)


if __name__ == "__main__":
    rt.run()

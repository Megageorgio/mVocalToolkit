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
            subtype = source.subtype if source.subtype in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"} else "PCM_24"
            sf.write(target, data, source.samplerate, subtype=subtype)
            files.append(str(target))
    return {"files": files}


if __name__ == "__main__":
    rt.run()

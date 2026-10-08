"""Vocal separation worker (python-audio-separator).

Never called implicitly: separation can degrade clean recordings, so only explicit /separate requests use it.
"""

from __future__ import annotations

import inspect
import os
import re
import shutil
from pathlib import Path
from typing import Any

import mvt_engine as rt

_separators: dict[tuple, Any] = {}
_STEM_RE = re.compile(r"_\(([^)]+)\)")


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    try:
        import static_ffmpeg  # noqa: PLC0415

        static_ffmpeg.add_paths(weak=True)
    except Exception as e:  # noqa: BLE001
        rt.log(f"ffmpeg is not available: {e}")


def _staging_dir() -> Path:
    path = Path(os.environ.get("MVT_CACHE", ".")) / "separation" / "out"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _separator(model_file: str, model_dir: str, output_format: str, single_stem: str | None,
               sample_rate: int, options: dict[str, Any]):
    key = (model_file, model_dir, output_format, single_stem, sample_rate, tuple(sorted(map(str, options.items()))))
    if key in _separators:
        return _separators[key]
    _separators.clear()
    rt.free_memory()
    _ensure_ffmpeg()
    from audio_separator.separator import Separator  # noqa: PLC0415

    accepted = set(inspect.signature(Separator.__init__).parameters)
    kwargs: dict[str, Any] = {
        "model_file_dir": model_dir,
        "output_dir": str(_staging_dir()),
        "output_format": output_format.upper(),
        "output_single_stem": single_stem,
        "sample_rate": sample_rate,
        "use_soundfile": True,
    }
    if "mdxc_params" not in options:
        # Roformer / MDXC models: audio-separator overlaps every chunk 8 times by default; 2 sounds the same in
        # practice and is about four times faster (a bigger batch helps a GPU)
        options = {**options, "mdxc_params": {"segment_size": 256, "override_model_segment_size": False,
                                              "batch_size": 2 if rt.device().startswith("cuda") else 1, "overlap": 2, "pitch_shift": 0}}
    for arch in ("mdx_params", "vr_params", "demucs_params", "mdxc_params"):
        if arch in options:
            kwargs[arch] = options[arch]
    if "use_autocast" in options:
        kwargs["use_autocast"] = bool(options["use_autocast"])
    separator = Separator(**{k: v for k, v in kwargs.items() if k in accepted})
    rt.progress(message=f"Loading separation model {model_file} (downloaded on first use)")
    separator.load_model(model_filename=model_file)
    _separators[key] = separator
    return separator


def _as_wav(audio: str, staging: Path, name: str) -> tuple[str, Path | None]:
    """audio-separator writes its outputs with the input's encoding: an MP3/OGG/M4A input makes it try MP3 data in a
    WAV file ("unsupported encoding"). Anything that is not already PCM/float WAV is converted to a 16-bit WAV first."""
    import soundfile as sf  # noqa: PLC0415

    try:
        info = sf.info(audio)
        if info.format == "WAV" and info.subtype in ("PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"):
            return audio, None
    except Exception:  # noqa: BLE001
        pass
    target = staging / f"{name}.input.wav"
    try:
        data, sr = sf.read(audio, always_2d=True)
        sf.write(str(target), data, sr, subtype="PCM_16")
    except Exception:  # noqa: BLE001
        # formats libsndfile can't read (m4a, aac...): through ffmpeg
        import subprocess  # noqa: PLC0415

        _ensure_ffmpeg()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", audio, "-acodec", "pcm_s16le", str(target)], check=True)
    return str(target), target


def _stem_name(file_name: str) -> str:
    found = _STEM_RE.findall(file_name)
    return (found[0] if found else Path(file_name).stem).strip().lower().replace(" ", "_")


@rt.method()
def separate(items: list[dict[str, Any]], model_file: str, model_dir: str, stems: list[str] | None = None,
             output_format: str = "wav", sample_rate: int = 44100,
             options: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    wanted = [s.lower() for s in stems] if stems else None
    single = stems[0] if stems and len(stems) == 1 else None
    separator = _separator(model_file, model_dir, output_format, single, sample_rate, options or {})
    staging = _staging_dir()
    results = []
    for index, item in enumerate(items):
        name = item.get("name") or Path(item["audio"]).stem
        rt.progress(index / len(items), f"Separating {name}")
        out_dir = Path(item["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            source_audio, temp = _as_wav(item["audio"], staging, name)
            try:
                produced = separator.separate(source_audio)
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
            files: dict[str, str] = {}
            for produced_file in produced:
                source = Path(produced_file)
                if not source.is_absolute():
                    source = staging / source
                stem = _stem_name(source.name)
                if wanted and stem not in wanted and not any(w in stem for w in wanted):
                    source.unlink(missing_ok=True)
                    continue
                target = out_dir / f"{name}_{stem}{source.suffix}"
                shutil.move(str(source), str(target))
                files[stem] = str(target)
            results.append({"ok": True, "name": name, "files": files})
        except Exception as e:  # noqa: BLE001
            if rt.is_oom(e):
                rt.free_memory()
            results.append({"ok": False, "name": name, "error": f"{type(e).__name__}: {e}"})
    rt.progress(1.0, "Separated")
    return results


@rt.method()
def list_models() -> dict[str, Any]:
    """Models known to audio-separator (for catalogs and GUIs)."""
    from audio_separator.separator import Separator  # noqa: PLC0415

    separator = Separator(output_dir=str(_staging_dir()), info_only=True)
    for name in ("list_supported_model_files", "get_simplified_model_list"):
        if hasattr(separator, name):
            return {"models": getattr(separator, name)()}
    return {"models": {}}


if __name__ == "__main__":
    rt.run()

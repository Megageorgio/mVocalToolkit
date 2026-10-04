"""Pitch (f0) extraction worker: RMVPE, FCPE, Parselmouth."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import mvt_engine as rt

_cache: dict[str, Any] = {}
NATIVE_HOP = 0.01  # seconds; all methods run at 10 ms and are resampled if another hop is requested


def _rmvpe_module():
    """Imports RMVPE's src/inference.py without src/__init__.py (it pulls training-only dependencies)."""
    if "src" not in sys.modules:
        for entry in sys.path:
            candidate = Path(entry) / "src" / "inference.py"
            if candidate.exists():
                package = types.ModuleType("src")
                package.__path__ = [str(candidate.parent)]  # type: ignore[attr-defined]
                sys.modules["src"] = package
                break
    import src.inference as inference  # noqa: PLC0415

    return inference


def _rmvpe(model_path: str, device: str):
    key = f"rmvpe:{model_path}:{device}"
    if key in _cache:
        return _cache[key]
    import torch  # noqa: PLC0415

    inference = _rmvpe_module()
    from src.model import E2E, E2E0  # noqa: PLC0415

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = None
    errors = []
    for cls in (E2E0, E2E):
        candidate = cls(4, 1, (2, 2))
        try:
            candidate.load_state_dict(state, strict=True)
            model = candidate
            break
        except RuntimeError as e:
            errors.append(str(e).splitlines()[0])
    if model is None:
        raise RuntimeError("Unknown RMVPE checkpoint format: " + "; ".join(errors))
    model.eval()
    rmvpe = inference.RMVPE.__new__(inference.RMVPE)
    rmvpe.resample_kernel = {}
    rmvpe.hop_length = 160
    rmvpe.seg_length = 32 * 160
    rmvpe.model = model.to(device)
    rmvpe.mel_extractor = inference.MelSpectrogram(
        inference.N_MELS, inference.SAMPLE_RATE, inference.WINDOW_LENGTH, 160, None, inference.MEL_FMIN,
        inference.MEL_FMAX,
    )
    _cache.clear()
    _cache[key] = rmvpe
    return rmvpe


def _fcpe(device: str):
    key = f"fcpe:{device}"
    if key not in _cache:
        from torchfcpe import spawn_bundled_infer_model  # noqa: PLC0415

        _cache.clear()
        _cache[key] = spawn_bundled_infer_model(device=device)
    return _cache[key]


def _load(path: str, sr: int | None):
    import librosa  # noqa: PLC0415

    audio, rate = librosa.load(path, sr=sr, mono=True)
    return audio, rate


def _extract(method: str, path: str, model_path: str | None, f0_min: float, f0_max: float,
             threshold: float | None, device: str):
    import numpy as np  # noqa: PLC0415

    if method == "rmvpe":
        if not model_path:
            raise ValueError("RMVPE needs a model (catalog id rmvpe)")
        audio, sr = _load(path, 16000)
        f0 = _rmvpe(model_path, device).infer_from_audio(audio, sr, device=device,
                                                       thred=0.03 if threshold is None else threshold)
        f0 = np.asarray(f0, dtype=np.float64)
    elif method == "fcpe":
        import torch  # noqa: PLC0415

        audio, sr = _load(path, 16000)
        n_frames = len(audio) // 160 + 1
        tensor = torch.from_numpy(audio).float().to(device)[None, :, None]
        with torch.no_grad():
            f0 = _fcpe(device).infer(tensor, sr=sr, decoder_mode="local_argmax",
                                     threshold=0.006 if threshold is None else threshold,
                                     f0_min=f0_min, f0_max=f0_max, interp_uv=False,
                                     output_interp_target_length=n_frames)
        f0 = f0.squeeze().detach().cpu().numpy().astype(np.float64)
    elif method == "parselmouth":
        import parselmouth  # noqa: PLC0415

        audio, sr = _load(path, None)
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        n_frames = int(len(audio) / sr / NATIVE_HOP) + 1
        pitch = sound.to_pitch_ac(time_step=NATIVE_HOP, voicing_threshold=0.6 if threshold is None else threshold,
                                  pitch_floor=f0_min, pitch_ceiling=f0_max)
        values = pitch.selected_array["frequency"]
        # Praat frames start at x1 (half a window in); shift to a 0-based 10 ms grid
        times = pitch.xs()
        grid = np.arange(n_frames) * NATIVE_HOP
        f0 = np.interp(grid, times, values, left=0.0, right=0.0)
        f0[np.interp(grid, times, (values > 0).astype(float), left=0, right=0) < 0.5] = 0.0
    else:
        raise ValueError(f"Unknown pitch method {method!r} (rmvpe, fcpe, parselmouth)")
    f0[(f0 < f0_min * 0.5) | ~np.isfinite(f0)] = 0.0
    return f0


def _resample(f0, hop: float):
    import numpy as np  # noqa: PLC0415

    if abs(hop - NATIVE_HOP) < 1e-9:
        return f0
    duration = (len(f0) - 1) * NATIVE_HOP
    grid = np.arange(int(duration / hop) + 1) * hop
    src = np.arange(len(f0)) * NATIVE_HOP
    voiced = np.interp(grid, src, (f0 > 0).astype(float)) >= 0.5
    filled = f0.copy()
    if (f0 > 0).any():
        idx = np.where(f0 > 0)[0]
        filled = np.interp(src, src[idx], f0[idx])
    out = np.interp(grid, src, filled)
    out[~voiced] = 0.0
    return out


@rt.method()
def pitch(items: list[dict[str, Any]], method: str = "rmvpe", model_path: str | None = None, hop: float = 0.01,
          f0_min: float = 50.0, f0_max: float = 1100.0, threshold: float | None = None) -> list[dict[str, Any]]:
    device = rt.device()
    if device == "mps" and method != "parselmouth":
        device = "cpu"
    results = []
    for index, item in enumerate(items):
        name = item.get("name") or Path(item["audio"]).stem
        rt.progress(index / len(items), f"Extracting pitch of {name}")
        try:
            f0 = _resample(_extract(method, item["audio"], model_path, f0_min, f0_max, threshold, device), hop)
            results.append({"ok": True, "name": name, "hop": hop, "f0": [round(float(v), 3) for v in f0]})
        except Exception as e:  # noqa: BLE001
            if rt.is_oom(e):
                rt.free_memory()
            results.append({"ok": False, "name": name, "error": f"{type(e).__name__}: {e}"})
    rt.progress(1.0, "Done")
    return results


if __name__ == "__main__":
    rt.run()

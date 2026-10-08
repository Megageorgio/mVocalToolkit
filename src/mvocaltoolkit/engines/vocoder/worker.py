"""Resynthesis of a recording with another f0: WORLD, or the PC-NSF-HiFiGAN vocoder of DiffSinger (ONNX)."""

from __future__ import annotations

from typing import Any

import mvt_engine as rt

_sessions: dict[str, Any] = {}

# PC-NSF-HiFiGAN 44.1 kHz (vocoder.yaml of the release)
NSF_SR = 44100
NSF_HOP = 512
NSF_WIN = 2048
NSF_MELS = 128
NSF_FMIN = 40
NSF_FMAX = 16000


def _curve(f0: list[float], hop: float, times):
    """The given f0 (0 = no value) at [times]; gaps are filled from the neighbours (the vocoder needs a pitch
    everywhere, voicing comes from the spectrum)."""
    import numpy as np  # noqa: PLC0415

    f = np.asarray(f0, dtype=np.float64)
    t = np.arange(len(f)) * hop
    voiced = f > 0
    if not voiced.any():
        raise ValueError("the f0 curve has no voiced values")
    return np.interp(times, t[voiced], f[voiced])


def _world(x, sr: int, f0: list[float], hop: float):
    import numpy as np  # noqa: PLC0415
    import pyworld as pw  # noqa: PLC0415

    xd = x.astype(np.float64)
    period = 5.0
    f0o, tt = pw.dio(xd, sr, f0_floor=50.0, f0_ceil=1100.0, frame_period=period)
    f0o = pw.stonemask(xd, f0o, tt, sr)
    sp = pw.cheaptrick(xd, f0o, tt, sr)
    ap = pw.d4c(xd, f0o, tt, sr)
    given = np.asarray(f0, dtype=np.float64)
    idx = np.clip(np.round(tt / hop).astype(int), 0, len(given) - 1)
    g = given[idx]
    # where the voice sounds and a pitch is given, that pitch; unvoiced stays unvoiced
    newf = np.where((f0o > 0) & (g > 0), g, f0o)
    y = pw.synthesize(newf, sp, ap, sr, period)
    return y.astype(np.float32), sr


def _nsf(x, sr: int, f0: list[float], hop: float, model: dict[str, Any]):
    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    path = model["onnx"]
    session = _sessions.get(path)
    if session is None:
        _sessions.clear()
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _sessions[path] = session
    if sr != NSF_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=NSF_SR)
    # the mel spectrogram DiffSinger trains on: magnitude STFT, slaney mel, natural log
    pad = (NSF_WIN - NSF_HOP) // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    spec = np.abs(librosa.stft(xp, n_fft=NSF_WIN, hop_length=NSF_HOP, win_length=NSF_WIN, window="hann", center=False))
    basis = librosa.filters.mel(sr=NSF_SR, n_fft=NSF_WIN, n_mels=NSF_MELS, fmin=NSF_FMIN, fmax=NSF_FMAX)
    mel = np.log(np.clip(basis @ spec, 1e-5, None)).T.astype(np.float32)
    times = np.arange(mel.shape[0]) * NSF_HOP / NSF_SR
    f = _curve(f0, hop, times).astype(np.float32)
    names = [i.name for i in session.get_inputs()]
    y = session.run(None, {names[0]: mel[None], names[1]: f[None]})[0][0]
    return y.astype(np.float32), NSF_SR


@rt.method()
def resynth(audio: str, out: str, f0: list[float], hop: float, method: str = "world", start: float = 0.0,
            end: float | None = None, model: dict[str, Any] | None = None) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    rt.progress(0.05, "Reading the recording")
    x, sr = sf.read(audio, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    a = max(0, int(start * sr))
    b = len(x) if end is None else min(len(x), int(end * sr))
    if b - a < sr // 20:
        raise ValueError("the part is too short")
    x = x[a:b]
    # the curve is for the whole file; cut out the same part
    k0 = int(round(start / hop))
    part = list(f0[k0:]) if k0 < len(f0) else []
    rt.progress(0.2, "Resynthesis (" + method + ")")
    if method == "nsf":
        if not model or not model.get("onnx"):
            raise ValueError("NSF-HiFiGAN needs its model")
        y, out_sr = _nsf(x, sr, part, hop, model)
    else:
        y, out_sr = _world(x, sr, part, hop)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    sf.write(out, y, out_sr, subtype="PCM_16")
    rt.progress(1.0, "Done")
    return {"ok": True, "file": out, "sample_rate": out_sr, "seconds": len(y) / out_sr}


if __name__ == "__main__":
    rt.run()

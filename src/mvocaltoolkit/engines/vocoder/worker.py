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


def _nsf_session(path: str):
    import onnxruntime as ort  # noqa: PLC0415

    session = _sessions.get(path)
    if session is None:
        _sessions.clear()
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _sessions[path] = session
    return session


def _mel(x, sr: int):
    """The mel spectrogram DiffSinger trains on (magnitude STFT, slaney mel, natural log), frames x mels."""
    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if sr != NSF_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=NSF_SR)
    pad = (NSF_WIN - NSF_HOP) // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    spec = np.abs(librosa.stft(xp, n_fft=NSF_WIN, hop_length=NSF_HOP, win_length=NSF_WIN, window="hann", center=False))
    basis = librosa.filters.mel(sr=NSF_SR, n_fft=NSF_WIN, n_mels=NSF_MELS, fmin=NSF_FMIN, fmax=NSF_FMAX)
    return np.log(np.clip(basis @ spec, 1e-5, None)).T.astype(np.float32)


def _vocode(session, mel, f0):
    import numpy as np  # noqa: PLC0415

    names = [i.name for i in session.get_inputs()]
    y = session.run(None, {names[0]: mel[None].astype(np.float32), names[1]: np.asarray(f0, dtype=np.float32)[None]})[0][0]
    return y.astype(np.float32)


def _nsf(x, sr: int, f0: list[float], hop: float, model: dict[str, Any]):
    import numpy as np  # noqa: PLC0415

    session = _nsf_session(model["onnx"])
    mel = _mel(x, sr)
    times = np.arange(mel.shape[0]) * NSF_HOP / NSF_SR
    return _vocode(session, mel, _curve(f0, hop, times)), NSF_SR


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


# ---------------- an oto.ini entry sung as a note ----------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def tone_name(midi: float) -> str:
    """UTAU note name of a MIDI number: 60 -> C4."""
    m = int(round(midi))
    return NOTE_NAMES[m % 12] + str(m // 12 - 1)


def oto_span(offset: float, consonant: float, cutoff: float, preutterance: float, overlap: float,
             total_ms: float) -> tuple[float, float, float, float, float]:
    """Where the entry's sound is in the recording (start, end in ms) and its consonant, preutterance and overlap
    from that start, kept inside it. Cutoff as in oto.ini: negative = length from offset, else from the file end."""
    start = min(max(0.0, offset), total_ms)
    end = start - cutoff if cutoff < 0 else total_ms - cutoff
    end = min(max(end, start), total_ms)
    span = end - start
    return start, end, min(max(consonant, 0.0), span), min(max(preutterance, 0.0), span), min(overlap, span)


def time_map(times_ms, consonant: float, span: float, length: float):
    """Where in the entry's sound (ms from its start) each moment of a rendering [length] ms long is taken from:
    the fixed part (up to [consonant]) as it is, the rest stretched or squeezed to fill the length, as UTAU
    resamplers do."""
    import numpy as np  # noqa: PLC0415

    t = np.asarray(times_ms, dtype=np.float64)
    if length > consonant and span > consonant:
        src = np.where(t <= consonant, t, consonant + (t - consonant) * (span - consonant) / (length - consonant))
    else:
        src = t
    return np.clip(src, 0.0, max(span - 1e-6, 0.0))


def frames_at(frames, positions):
    """Rows of [frames] at fractional [positions] (linear between neighbours)."""
    import numpy as np  # noqa: PLC0415

    f = np.asarray(frames, dtype=np.float64)
    p = np.clip(np.asarray(positions, dtype=np.float64), 0, len(f) - 1)
    lo = np.floor(p).astype(int)
    hi = np.minimum(lo + 1, len(f) - 1)
    w = (p - lo).reshape((-1,) + (1,) * (f.ndim - 1))
    return f[lo] * (1 - w) + f[hi] * w


def note_layout(preutterance: float, overlap: float, length: float, notes: int) -> list[tuple[float, float, float, float]]:
    """The notes to mix, each (where its sound starts, how long it is, fade in, fade out), ms. One note: its sound
    from the offset, the note itself starting at the preutterance. Two notes of the same entry in a row: the
    second starts where the first ends, its sound begins a preutterance earlier and the first fades out over its
    overlap, as UTAU joins notes."""
    tail = 30.0
    o = max(overlap, 0.0)
    if notes < 2:
        return [(0.0, preutterance + length, o, tail)]
    second = length  # (preutterance + length) - preutterance
    return [(0.0, length + o, o, o if o > 0 else tail), (second, preutterance + length, o, tail)]


def mix(parts: list, sr: int, lead_ms: float = 50.0, trail_ms: float = 100.0):
    """Puts the rendered notes (where in ms, samples, fade in ms, fade out ms) on one track with silence around."""
    import numpy as np  # noqa: PLC0415

    lead = int(lead_ms * sr / 1000)
    end = max(int(at * sr / 1000) + len(y) for at, y, _fi, _fo in parts)
    track = np.zeros(lead + end + int(trail_ms * sr / 1000), dtype=np.float64)
    for at, y, fade_in, fade_out in parts:
        y = np.asarray(y, dtype=np.float64).copy()
        n = len(y)
        fi = min(int(fade_in * sr / 1000), n)
        fo = min(int(fade_out * sr / 1000), n)
        if fi > 0:
            y[:fi] *= np.linspace(0.0, 1.0, fi, endpoint=False)
        if fo > 0:
            y[n - fo:] *= np.linspace(1.0, 0.0, fo)
        a = lead + int(at * sr / 1000)
        track[a:a + n] += y
    return track.astype(np.float32)


def _world_renderer(seg, sr: int, consonant: float, span: float, hz: float):
    import numpy as np  # noqa: PLC0415
    import pyworld as pw  # noqa: PLC0415

    xd = np.ascontiguousarray(seg, dtype=np.float64)
    period = 5.0
    f0o, tt = pw.dio(xd, sr, f0_floor=50.0, f0_ceil=1100.0, frame_period=period)
    f0o = pw.stonemask(xd, f0o, tt, sr)
    sp = pw.cheaptrick(xd, f0o, tt, sr)
    ap = pw.d4c(xd, f0o, tt, sr)
    voiced = (f0o > 0).astype(np.float64)

    def render(length: float):
        n = int(np.ceil(length / period)) + 1
        pos = time_map(np.arange(n) * period, consonant, span, length) / period
        f0n = np.where(frames_at(voiced, pos) >= 0.5, hz, 0.0)
        y = pw.synthesize(np.ascontiguousarray(f0n), np.ascontiguousarray(frames_at(sp, pos)),
                          np.ascontiguousarray(frames_at(ap, pos)), sr, period)
        return y[:int(length * sr / 1000)].astype(np.float32)

    return render, sr


def _nsf_renderer(seg, sr: int, consonant: float, span: float, hz: float, model: dict[str, Any]):
    import numpy as np  # noqa: PLC0415

    session = _nsf_session(model["onnx"])
    mel = _mel(seg, sr)
    hop_ms = NSF_HOP / NSF_SR * 1000

    def render(length: float):
        n = int(np.ceil(length / hop_ms)) + 1
        pos = time_map(np.arange(n) * hop_ms, consonant, span, length) / hop_ms
        y = _vocode(session, frames_at(mel, pos).astype(np.float32), np.full(n, hz, dtype=np.float32))
        return y[:int(length * NSF_SR / 1000)]

    return render, NSF_SR


def resampler_args(exe: str, audio: str, out: str, tone: float, flags: str, offset: float, length: float,
                   consonant: float, cutoff: float) -> list[str]:
    """The command line UTAU gives a resampler: the note, its length rounded up to 50 ms, the oto values, volume
    100, no modulation, tempo 120 and a flat pitch bend."""
    import math  # noqa: PLC0415
    import os  # noqa: PLC0415

    required = int(math.ceil(length / 50.0) * 50)
    points = int(math.ceil(required / (60000.0 / 120 / 96))) + 1  # a pitch value every 1/96 beat
    args = [audio, out, tone_name(tone), "100", flags, f"{offset:.0f}", str(required), f"{consonant:.0f}",
            f"{cutoff:.0f}", "100", "0", "!120", f"AA#{points}#"]
    prefix = ["wine"] if os.name != "nt" and exe.lower().endswith(".exe") else []
    return [*prefix, exe, *args]


def _resampler_renderer(exe: str, audio: str, tone: float, flags: str, offset: float, consonant: float,
                        cutoff: float, folder: str):
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    import soundfile as sf  # noqa: PLC0415

    if not os.path.isfile(exe):
        raise ValueError(f"no resampler at {exe}")
    rate: list[int] = []
    made = [0]

    def render(length: float):
        made[0] += 1
        out = os.path.join(folder, f"_note_{made[0]}.wav")
        args = resampler_args(exe, audio, out, tone, flags, offset, length, consonant, cutoff)
        done = subprocess.run(args, cwd=os.path.dirname(exe) or None, capture_output=True, timeout=180)
        if not os.path.isfile(out):
            text = (done.stderr or done.stdout or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(f"the resampler made no sound (exit code {done.returncode})" + (": " + text[-400:] if text else ""))
        y, sr = sf.read(out, dtype="float32", always_2d=True)
        os.remove(out)
        rate[:] = [sr]
        return y.mean(axis=1)[:int(length * sr / 1000)]

    return render, rate


@rt.method()
def note(audio: str, out: str, offset: float, consonant: float, cutoff: float, preutterance: float, overlap: float,
         tone: float = 60.0, length: float = 600.0, notes: int = 1, method: str = "world",
         model: dict[str, Any] | None = None, resampler: str | None = None, flags: str = "") -> dict[str, Any]:
    """An oto.ini entry of [audio] sung as a note of MIDI pitch [tone], [length] ms long (once, or twice in a row
    to hear how the entry joins itself): WORLD, the NSF-HiFiGAN vocoder, or an UTAU resampler program."""
    import os  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    rt.progress(0.05, "Reading the recording")
    x, sr = sf.read(audio, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    start, end, cons, pre, ovl = oto_span(offset, consonant, cutoff, preutterance, overlap, len(x) * 1000.0 / sr)
    span = end - start
    if span < 20:
        raise ValueError("the entry is too short to sing")
    hz = 440.0 * 2.0 ** ((tone - 69.0) / 12.0)
    layout = note_layout(pre, ovl, max(length, 50.0), notes)
    rt.progress(0.2, "Singing the note (" + method + ")")
    if method == "resampler":
        if not resampler:
            raise ValueError("no resampler chosen")
        render, rate = _resampler_renderer(resampler, audio, tone, flags, offset, consonant, cutoff,
                                           os.path.dirname(os.path.abspath(out)))
        rendered = [render(n_len) for _at, n_len, _fi, _fo in layout]
        out_sr = rate[0]
    else:
        seg = x[int(start * sr / 1000):int(end * sr / 1000)]
        if method == "nsf":
            if not model or not model.get("onnx"):
                raise ValueError("NSF-HiFiGAN needs its model")
            render, out_sr = _nsf_renderer(seg, sr, cons, span, hz, model)
        else:
            render, out_sr = _world_renderer(seg, sr, cons, span, hz)
        rendered = [render(n_len) for _at, n_len, _fi, _fo in layout]
    y = mix([(at, r, fi, fo) for (at, _n, fi, fo), r in zip(layout, rendered)], out_sr)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    sf.write(out, y, out_sr, subtype="PCM_16")
    rt.progress(1.0, "Done")
    return {"ok": True, "file": out, "sample_rate": out_sr, "seconds": len(y) / out_sr, "tone": tone_name(tone)}


if __name__ == "__main__":
    rt.run()

"""
MCFL inference: phoneme labelling of singing without lyrics, on CPU, from an exported MCFL package.
Needs only numpy, onnxruntime, soundfile and soxr (no PyTorch). This one file is all the runtime.

  python mcfl_infer.py song.wav [more.wav ...] --lang en -m path/to/package [-o out_dir] [--threads 1]

  from mcfl_infer import MCFL
  m = MCFL("path/to/package", threads=1)        # load once
  segments = m.label_file("song.wav", lang="ja") # [(start_sec, end_sec, phoneme), ...]

Package (made by `python -m mcfl.export`): model.onnx / model_int8.onnx, labels.txt, langs.txt,
lang_phonemes.json, phoneme_merge_map.json, config.json.
Only the chosen language's own phonemes (and the shared pause labels) can come out; names are the language's own.
"""
import argparse
import json
import os
import time

import numpy as np

SR = 16000


# ---------------------------------------------------------------- decoding (numpy)

def log_softmax(x):
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


class BioViterbi:
    """Best B/I tag path: B-x may follow anything, I-x only B-x or I-x (with bonus log(bias)); O is not used.
    The Viterbi of WFL-ASR, in O(frames x phonemes)."""

    def __init__(self, labels):
        phones = [t[2:] for t in labels if t.startswith("B-") and f"I-{t[2:]}" in labels]
        self.phones = phones
        self.b_idx = np.array([labels.index(f"B-{p}") for p in phones])
        self.i_idx = np.array([labels.index(f"I-{p}") for p in phones])
        self.keep = np.concatenate([self.b_idx, self.i_idx])

    def __call__(self, logp, bias=5.0):
        T = logp.shape[0]
        if T == 0:
            return np.zeros(0, bool), np.zeros(0, np.int64)
        lp = log_softmax(logp[:, self.keep].astype(np.float64))
        n = len(self.phones)
        lb, li = lp[:, :n], lp[:, n:]
        logb = np.log(bias)
        sb = -np.log(n) + lb[0]
        si = np.full(n, -np.inf)
        back_any = np.zeros(T, dtype=np.int64)
        from_b = np.zeros((T, n), dtype=bool)
        for t in range(1, T):
            jb, ji = int(sb.argmax()), int(si.argmax())
            if sb[jb] >= si[ji]:
                m, back_any[t] = sb[jb], jb
            else:
                m, back_any[t] = si[ji], n + ji
            fb = sb > si
            from_b[t] = fb
            si = np.where(fb, sb, si) + logb + li[t]
            sb = m + lb[t]
        jb, ji = int(sb.argmax()), int(si.argmax())
        state = jb if sb[jb] >= si[ji] else n + ji
        path = np.empty(T, dtype=np.int64)
        for t in range(T - 1, -1, -1):
            path[t] = state
            if t == 0:
                break
            if state < n:
                state = int(back_any[t])
            elif not from_b[t, state - n]:
                pass
            else:
                state = state - n
        return path < n, path % n


def segments_from_path(is_b, ph, phones, offsets, fd):
    out, T, start = [], len(ph), 0
    for t in range(1, T + 1):
        if t == T or is_b[t] or ph[t] != ph[t - 1]:
            s = (start + float(offsets[start, 0])) * fd
            e = (t - 1 + float(offsets[t - 1, 1])) * fd
            out.append((s, e, phones[int(ph[start])]))
            start = t
    return out


def merge_adjacent(segs):
    out = []
    for s, e, p in segs:
        if out and out[-1][2] == p:
            out[-1] = (out[-1][0], e, p)
        else:
            out.append((s, e, p))
    return out


def hard_silence(segments, audio, threshold=0.005, min_duration=0.5, label="SP"):
    """Long quiet stretches become pauses (as WFL-ASR does without lyrics)."""
    fl = SR // 100
    a = np.pad(np.abs(audio), (0, (fl - len(audio) % fl) % fl))
    silent = a.reshape(-1, fl).max(1) < threshold
    iv, start = [], None
    for i, s in enumerate(list(silent) + [False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) * 0.01 >= min_duration:
                iv.append((start * 0.01, i * 0.01))
            start = None
    out = list(segments)
    for s0, e0 in iv:
        nxt = []
        for s, e, p in out:
            if e <= s0 or s >= e0:
                nxt.append((s, e, p))
                continue
            if s < s0:
                nxt.append((s, s0, p))
            if e > e0:
                nxt.append((e0, e, p))
        out = nxt + [(s0, e0, label)]
    return sorted(out, key=lambda x: x[0])


def continuous(segments, duration):
    starts = []
    for s, _, p in sorted(segments, key=lambda x: x[0]):
        s = max(0.0, float(s))
        if s >= duration or (starts and s <= starts[-1][0]):
            continue
        starts.append((s, p))
    if not starts:
        return [(0.0, duration, "SP")]
    starts[0] = (0.0, starts[0][1])
    return [(s, starts[i + 1][0] if i + 1 < len(starts) else duration, p) for i, (s, p) in enumerate(starts)]


def postprocess(logp, offsets, audio, labels, cfg, viterbi=None):
    """Frame log-probabilities -> continuous [(start, end, label)] covering the whole audio."""
    viterbi = viterbi or BioViterbi(labels)
    is_b, ph = viterbi(logp, cfg.get("viterbi_bias", 5.0))
    segs = merge_adjacent(segments_from_path(is_b, ph, viterbi.phones, offsets, cfg.get("frame_sec", 0.02)))
    segs = hard_silence(segs, audio, cfg.get("silence_threshold", 0.005), cfg.get("min_silence_sec", 0.5),
                        cfg.get("silence_label", "SP"))
    return continuous(merge_adjacent(segs), len(audio) / SR)


SHARED = ("AP", "SP", "cl", "vf", "exh")


class LanguageDecoders:
    """Per language: the label columns it may use ("O", its phonemes, the shared pauses) and a Viterbi over them."""

    def __init__(self, labels, langs, lang_phonemes, shared=SHARED):
        self.d = {}
        for lang, lid in langs.items():
            keep = set(lang_phonemes.get(lang, [])) | set(shared)
            idx = [i for i, lab in enumerate(labels) if lab == "O" or lab[2:] in keep]
            sub = [labels[i] for i in idx]
            self.d[lid] = (np.array(idx), sub, BioViterbi(sub))

    def __call__(self, logp, offsets, audio, lang_id, cfg):
        idx, sub, vit = self.d[lang_id]
        return postprocess(logp[:, idx], offsets, audio, sub, cfg, vit)


# ---------------------------------------------------------------- audio

def load_audio(path):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(a.mean(1), sr)


def resample(a, sr):
    if sr == SR:
        return a.astype(np.float32)
    try:
        import soxr
        return soxr.resample(a, sr, SR).astype(np.float32)
    except ImportError:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        return resample_poly(a, SR // g, sr // g).astype(np.float32)


def write_lab(path, segs):
    """HTK label file, times in 100 ns units."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s, e, p in segs:
            f.write(f"{int(round(s * 1e7))} {int(round(e * 1e7))} {p}\n")


# ---------------------------------------------------------------- model

class MCFL:
    def __init__(self, model_dir, threads=1, int8=None):
        import onnxruntime as ort
        j = lambda f: os.path.join(model_dir, f)  # noqa: E731
        self.cfg = json.load(open(j("config.json"), encoding="utf-8"))
        self.labels = [x.strip() for x in open(j("labels.txt"), encoding="utf-8") if x.strip()]
        self.langs = {l.split(",")[0]: int(l.split(",")[1]) for l in open(j("langs.txt"), encoding="utf-8") if l.strip()}
        self.merge = json.load(open(j("phoneme_merge_map.json"), encoding="utf-8"))
        self.decoders = LanguageDecoders(self.labels, self.langs, json.load(open(j("lang_phonemes.json"), encoding="utf-8")),
                                         self.cfg.get("shared_labels", SHARED))
        use8 = int8 or (int8 is None and os.path.exists(j("model_int8.onnx")) and self.cfg.get("prefer_int8", True))
        self.model_file = "model_int8.onnx" if use8 else "model.onnx"
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(j(self.model_file), so, providers=["CPUExecutionProvider"])
        self.fd = self.cfg.get("frame_sec", 0.02)

    def _run(self, audio, lid):
        lp, off = self.sess.run(None, {"audio": audio[None, :].astype(np.float32), "lang": np.array([lid], np.int64)})
        return lp[0], off[0]

    def frames(self, audio, lid):
        """Log-probabilities [T, labels] and offsets [T, 2] for the whole file (long files in overlapping windows)."""
        n_total = int(np.ceil(len(audio) / SR / self.fd))
        win = int(self.cfg.get("window_sec", 120) * SR)
        ov = int(self.cfg.get("overlap_sec", 16) * SR)
        if len(audio) <= win:
            lp, off = self._run(audio, lid)
            return lp[:n_total], off[:n_total]
        hop, fpf = win - ov, int(round(self.fd * SR))
        lps, offs, done, start = [], [], 0, 0
        while True:
            end = min(start + win, len(audio))
            lp, off = self._run(audio[start:end], lid)
            f0 = start // fpf
            last = end >= len(audio)
            keep_to = len(lp) if last else (start + win - ov // 2) // fpf - f0
            lps.append(lp[done - f0:keep_to])
            offs.append(off[done - f0:keep_to])
            done = f0 + keep_to
            if last:
                break
            start += hop
        return np.concatenate(lps)[:n_total], np.concatenate(offs)[:n_total]

    def label(self, audio, sr=SR, lang=None, native=True):
        lang = lang or next(iter(self.langs))
        if lang not in self.langs:
            raise ValueError(f"language {lang!r} is not in this model: {', '.join(sorted(self.langs))}")
        a = resample(np.asarray(audio, dtype=np.float32), sr)
        a = a / (np.abs(a).max() + 1e-8)
        lid = self.langs[lang]
        lp, off = self.frames(a, lid)
        segs = self.decoders(lp, off, a, lid, self.cfg)
        if native:  # shared (merged) label -> the language's own name
            segs = [(s, e, self.merge.get(p, {}).get(lang, p)) for s, e, p in segs]
        return segs

    def label_file(self, path, lang=None, native=True):
        return self.label(load_audio(path), SR, lang, native)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--lang", help="language code from langs.txt (default: the first one)")
    ap.add_argument("-m", "--model", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("-o", "--out", help="folder for the .lab files (default: next to the audio)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fp32", action="store_true", help="use model.onnx even if model_int8.onnx exists")
    args = ap.parse_args()
    m = MCFL(args.model, threads=args.threads, int8=False if args.fp32 else None)
    for p in args.audio:
        t0 = time.time()
        a = load_audio(p)
        segs = m.label(a, SR, args.lang)
        out = os.path.join(args.out or os.path.dirname(os.path.abspath(p)), os.path.splitext(os.path.basename(p))[0] + ".lab")
        if args.out:
            os.makedirs(args.out, exist_ok=True)
        write_lab(out, segs)
        dt = time.time() - t0
        print(f"{p}: {len(a) / SR:.1f} s of audio in {dt:.2f} s ({len(a) / SR / dt:.0f}x real time) -> {out}")


if __name__ == "__main__":
    main()

"""Render the actual exported graphs, with strict input contracts and no mock audio."""
from pathlib import Path
import json
import math
import re
import numpy as np
import soundfile as sf
import yaml
import onnxruntime as ort
import mvt_engine as rt
from mvt_mel import compatibility_errors


def local(root, ref):
    path = (root / str(ref)).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping bank reference: {ref}")
    return path


def config(folder):
    return yaml.safe_load(local(folder, "dsconfig.yaml").read_text(encoding="utf-8")) or {}


def vocab(folder, cfg, key):
    data = json.loads(local(folder, cfg.get(key, key + ".json")).read_text(encoding="utf-8"))
    return {s: i for i, s in enumerate(data)} if isinstance(data, list) else data


def session(path, threads=2):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def run_graph(sess, values):
    feed = {}
    for meta in sess.get_inputs():
        if meta.name not in values:
            raise ValueError(f"ONNX graph requires unsupported/missing input {meta.name}; provide it in DS/advanced inputs")
        dtype = {"tensor(int64)": np.int64, "tensor(float)": np.float32, "tensor(bool)": np.bool_, "tensor(double)": np.float64}.get(meta.type)
        if dtype is None:
            raise ValueError(f"Unsupported ONNX input type: {meta.type}")
        array = np.asarray(values[meta.name], dtype=dtype)
        if meta.name == "depth" and dtype in {np.float32, np.float64}:
            array = array / 1000
        if array.ndim != len(meta.shape) or any(isinstance(dim, int) and dim != actual for dim, actual in zip(meta.shape, array.shape)):
            raise ValueError(f"ONNX shape mismatch for {meta.name}: {array.shape}, expected {meta.shape}")
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite ONNX input: {meta.name}")
        feed[meta.name] = array
    result = sess.run(None, feed)
    if any(v.dtype.kind == "f" and not np.all(np.isfinite(v)) for v in result):
        raise ValueError("ONNX produced non-finite values")
    return {meta.name: result[i] for i, meta in enumerate(sess.get_outputs())}


def midi(note):
    if note.lower() in {"rest", "sp", "ap", "r"}:
        return 0
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", note)
    if not match:
        return float(note)
    name, accidental, octave = match.groups()
    return (int(octave) + 1) * 12 + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[name.upper()] + {"": 0, "#": 1, "b": -1}[accidental]


def frames(durations, timestep):
    numbers = np.asarray([float(x) for x in durations], dtype=np.float64)
    if not len(numbers) or not np.all(np.isfinite(numbers)) or np.any(numbers < 0):
        raise ValueError("Durations must be finite, non-negative and non-empty")
    return np.diff(np.rint(np.r_[0, np.cumsum(numbers)] / timestep)).astype(np.int64)


def curve(segment, key, count, timestep, default=None):
    values = segment.get(key)
    if values is None or values == "":
        if default is None:
            raise ValueError(f"DS is missing {key}; enable its bank prediction or provide a curve")
        return np.full((1, count), default, np.float32)
    samples = np.asarray([float(x) for x in str(values).split()], np.float32)
    hop = float(segment.get(key + "_timestep", segment.get("f0_timestep", timestep)))
    if not len(samples) or not math.isfinite(hop) or hop <= 0:
        raise ValueError(f"Invalid curve timestep/data: {key}")
    return np.interp(np.arange(count) * timestep, np.arange(len(samples)) * hop, samples)[None].astype(np.float32)


def embedding(folder, cfg, speaker, count):
    speakers = cfg.get("speakers") or []
    if not speakers:
        return None
    chosen = next((s for s in speakers if Path(s).name == speaker), None) if speaker else speakers[0]
    if chosen is None:
        raise ValueError(f"Speaker not found: {speaker}")
    vector = np.fromfile(local(folder, str(chosen) + ".emb"), dtype=np.float32)
    if not vector.size or not np.all(np.isfinite(vector)):
        raise ValueError("Invalid speaker embedding")
    return np.repeat(vector[None, None], count, axis=1)


def tokens_for(phones, folder, cfg, language):
    inventory = vocab(folder, cfg, "phonemes")
    languages = vocab(folder, cfg, "languages") if cfg.get("use_lang_id") else {}
    result, ids = [], []
    for phone in phones:
        lang, short = phone.split("/", 1) if "/" in phone else (language, phone)
        full = f"{lang}/{short}"
        selected = full if full in inventory else short
        if selected not in inventory:
            raise ValueError(f"Untrained phoneme: {phone}")
        result.append(inventory[selected])
        if languages:
            if short in {"AP", "SP"} or selected == short:
                ids.append(0)
            elif lang in languages:
                ids.append(languages[lang])
            else:
                raise ValueError(f"Bank has no language {lang}")
    return np.array([result], np.int64), np.array([ids], np.int64) if ids else None


@rt.method()
def render(bank: str, ds: str, out_dir: str, vocoder: str | None = None, speaker: str | None = None,
           language: str = "ru", steps: int = 20, depth: float = 1, predict: list[str] | None = None,
           gender: float = 0, velocity: float = 1, expression: float = 1, inputs: dict | None = None,
           cpu_threads: int = 2, cache_models: bool = True) -> dict:
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 32 or type(cache_models) is not bool:
        raise ValueError("cpu_threads must be an integer from 1 to 32 and cache_models a boolean")
    graphs = {}
    loads = hits = 0
    def load_graph(path):
        nonlocal loads, hits
        if cache_models and path in graphs:
            hits += 1
            return graphs[path]
        loaded = session(path, cpu_threads)
        loads += 1
        if cache_models:
            graphs[path] = loaded
        return loaded
    root, out = Path(bank).resolve(), Path(out_dir).resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Render output and bank must not overlap")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Render output must be empty")
    if not 1 <= steps <= 1000 or not 0 < depth <= 1 or velocity <= 0:
        raise ValueError("Invalid sampling steps, depth or velocity")
    cfg = config(root)
    acoustic_path = local(root, cfg["acoustic"])
    vocoder_root = Path(vocoder).resolve() if vocoder else root / "dsvocoder"
    vcfg_path = vocoder_root / "vocoder.yaml"
    if not vcfg_path.is_file():
        raise ValueError("Choose a local ONNX vocoder folder (vocoder.yaml); a training PTH is not an ONNX vocoder")
    vcfg = yaml.safe_load(vcfg_path.read_text(encoding="utf-8")) or {}
    issues = compatibility_errors(cfg, vcfg)
    if issues:
        raise ValueError("Incompatible vocoder: " + "; ".join(issues))
    vocoder_path = local(vocoder_root, vcfg.get("model", "vocoder.onnx"))
    rate = int(cfg.get("sample_rate", 44100))
    hop = int(cfg.get("hop_size", 512))
    if int(vcfg.get("sample_rate", rate)) != rate or int(vcfg.get("hop_size", hop)) != hop:
        raise ValueError("Vocoder and acoustic sample rate/hop differ")
    timestep = hop / rate
    segments = json.loads(Path(ds).read_text(encoding="utf-8-sig"))
    if not isinstance(segments, list) or not segments:
        raise ValueError("DS must be a non-empty segment list")
    predicted = set(predict or [])
    if predicted - {"dur", "pitch", "variance"}:
        raise ValueError("Supported bank predictions: dur, pitch, variance")
    acoustic = load_graph(acoustic_path)
    voc = load_graph(vocoder_path)
    chunks, report = [], []
    for index, segment in enumerate(segments):
        rt.progress(index / len(segments), "Rendering bank ONNX graphs", stage="bank_render")
        phones = segment["ph_seq"].split()
        durations = frames(str(segment["ph_dur"]).split(), timestep)
        if len(durations) != len(phones) or durations.sum() < 1:
            raise ValueError("ph_seq/ph_dur lengths differ or phrase is empty")
        count = int(durations.sum())
        if count * timestep > 600:
            raise ValueError("Render segments must be at most 10 minutes")
        notes = [midi(n) for n in str(segment.get("note_seq", "")).split()]
        note_durations = frames(str(segment.get("note_dur", "")).split(), timestep) if notes else durations.copy()
        if not notes or len(notes) != len(note_durations):
            if not segment.get("f0_seq"):
                raise ValueError("DS needs f0 or timed notes")
            notes, note_durations = [60.0], np.array([count])
        base_pitch = np.repeat(notes, note_durations).astype(np.float32)
        if len(base_pitch) != count:
            raise ValueError("Note and phoneme durations differ")
        f0 = curve(segment, "f0_seq", count, timestep) if segment.get("f0_seq") else (440 * 2 ** ((base_pitch[None] - 69) / 12)).astype(np.float32)
        if not segment.get("f0_seq"):
            f0[:, base_pitch == 0] = 0
        pitch = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1e-8) / 440), 0).astype(np.float32)
        values = dict(ph_dur=durations[None], durations=durations[None], f0=f0, pitch=pitch,
            speedup=np.array([max(1, 1000 // steps)]), steps=np.array([steps]), depth=np.array([round(depth * 1000)]),
            gender=np.full((1, count), gender), velocity=np.full((1, count), velocity), expr=np.full((1, count), expression),
            note_midi=np.array([notes], np.float32), note_rest=np.array([[n == 0 for n in notes]], bool), note_dur=note_durations[None],
            retake=np.ones((1, count), bool))
        if "dur" in predicted:
            duration_folder = root / "dsdur"
            sub = config(duration_folder)
            values["tokens"], langs = tokens_for(phones, duration_folder, sub, language)
            if langs is not None:
                values["languages"] = langs
            groups = [int(x) for x in str(segment.get("ph_num", " ".join("1" for _ in phones))).split()]
            if sum(groups) != len(phones) or any(g <= 0 for g in groups):
                raise ValueError("Duration prediction requires valid ph_num")
            positions = np.r_[0, np.cumsum(groups)]
            values["word_div"] = np.array([groups], np.int64)
            values["word_dur"] = np.array([[durations[a:b].sum() for a, b in zip(positions, positions[1:])]], np.int64)
            starts = np.r_[0, np.cumsum(durations)]
            values["ph_midi"] = np.array([[round(float(base_pitch[min(count - 1, int((a + b) / 2))])) for a, b in zip(starts, starts[1:])]], np.int64)
            values.update(run_graph(load_graph(local(duration_folder, sub["linguistic"])), values))
            spk = embedding(duration_folder, sub, speaker, len(phones))
            if spk is not None:
                values["spk_embed"] = spk
            values.update(inputs or {})
            raw = next(iter(run_graph(load_graph(local(duration_folder, sub["dur"])), values).values())).reshape(-1)
            if len(raw) != len(phones) or np.any(raw < 0):
                raise ValueError("Invalid predicted durations")
            # Keep word/note anchors while distributing each word across phones.
            for a, b in zip(positions, positions[1:]):
                total = int(durations[a:b].sum())
                if raw[a:b].sum() <= 0:
                    raise ValueError("Duration prediction has an empty word")
                cumulative = np.rint(np.r_[0, np.cumsum(raw[a:b])] / raw[a:b].sum() * total).astype(np.int64)
                durations[a:b] = np.diff(cumulative)
            values.update(ph_dur=durations[None], durations=durations[None])
        for component in ("pitch", "variance"):
            folder = root / ("dspitch" if component == "pitch" else "dsvariance")
            needed = component in predicted or (component == "pitch" and not segment.get("f0_seq")) or any(
                cfg.get("use_" + k + "_embed") and not segment.get(k) for k in ("energy", "breathiness", "voicing", "tension")) and component == "variance"
            if not needed:
                continue
            if not (folder / "dsconfig.yaml").is_file():
                if component in predicted:
                    raise ValueError(f"Bank has no {component} model")
                continue
            sub = config(folder)
            token_values, languages = tokens_for(phones, folder, sub, language)
            values["tokens"] = token_values
            if languages is not None:
                values["languages"] = languages
            groups = [int(x) for x in str(segment.get("ph_num", " ".join("1" for _ in phones))).split()]
            if sum(groups) != len(phones) or any(g <= 0 for g in groups):
                raise ValueError("Invalid ph_num")
            positions = np.r_[0, np.cumsum(groups)]
            values["word_div"] = np.array([groups], np.int64)
            values["word_dur"] = np.array([[durations[a:b].sum() for a, b in zip(positions, positions[1:])]], np.int64)
            linguistic = run_graph(load_graph(local(folder, sub["linguistic"])), values)
            values.update(linguistic)
            spk = embedding(folder, sub, speaker, count)
            if spk is not None:
                values["spk_embed"] = spk
            for key in ("energy", "breathiness", "voicing", "tension"):
                if segment.get(key):
                    values[key] = curve(segment, key, count, timestep)
                elif sub.get("predict_" + key):
                    values[key] = np.zeros((1, count), np.float32)
            if component == "variance":
                output_keys = [key for key in ("energy", "breathiness", "voicing", "tension") if sub.get("predict_" + key)]
                values["retake"] = np.ones((1, count, len(output_keys)), bool)
            values.update(inputs or {})
            output = run_graph(load_graph(local(folder, sub[component])), values)
            if component == "pitch":
                pitch = next(iter(output.values()))
                values["pitch"] = pitch
                values["f0"] = (440 * 2 ** ((pitch - 69) / 12)).astype(np.float32)
            else:
                for key, value in output.items():
                    values[key.removesuffix("_pred")] = value
        values["tokens"], langs = tokens_for(phones, root, cfg, language)
        if langs is not None:
            values["languages"] = langs
        spk = embedding(root, cfg, speaker, count)
        if spk is not None:
            values["spk_embed"] = spk
        elif "spk_embed" in values:
            del values["spk_embed"]
        for key in ("energy", "breathiness", "voicing", "tension"):
            if cfg.get("use_" + key + "_embed") and key not in values:
                values[key] = curve(segment, key, count, timestep)
        values.update(inputs or {})
        mel = next(iter(run_graph(acoustic, values).values()))
        mel_base = str(cfg.get("mel_base", "e"))
        vocoder_base = str(vcfg.get("mel_base", mel_base))
        if mel_base != vocoder_base:
            if {mel_base, vocoder_base} != {"e", "10"}:
                raise ValueError("Unsupported mel logarithm base conversion")
            mel = mel / math.log(10) if mel_base == "e" else mel * math.log(10)
        waveform = next(iter(run_graph(voc, dict(mel=mel, f0=values["f0"])).values())).reshape(-1)
        if not len(waveform):
            raise ValueError("Vocoder returned empty audio")
        offset = float(segment.get("offset", max((c[0] + len(c[1]) for c in chunks), default=0) / rate))
        if not math.isfinite(offset) or offset < 0 or offset > 3600:
            raise ValueError("Invalid DS offset")
        chunks.append((round(offset * rate), waveform))
        report.append(dict(segment=index, frames=count, graphs="exported ONNX", phones=len(phones)))
    length = max(start + len(samples) for start, samples in chunks)
    if length > rate * 3600:
        raise ValueError("Rendered phrase exceeds one hour")
    audio = np.zeros(length, np.float32)
    for start, samples in chunks:
        audio[start:start + len(samples)] += samples
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bank-preview.wav"
    sf.write(path, audio, rate, subtype="FLOAT")
    resources = dict(cpu_threads=cpu_threads, cache_models=cache_models, session_loads=loads, session_reuses=hits)
    (out / "render-report.json").write_text(json.dumps(dict(bank=str(root), ds=ds, sample_rate=rate, segments=report, resources=resources), indent=1), encoding="utf-8")
    return dict(files=[str(path)], out_dir=str(out), sample_rate=rate, seconds=len(audio) / rate, resources=resources)


if __name__ == "__main__":
    rt.run()

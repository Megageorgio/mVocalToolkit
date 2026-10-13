"""WFL-ASR (refactor branch) engine worker: the same steps as its infer.py main(), for a list of files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mvt_engine as rt
import yaml

_cache: dict[str, Any] = {}


def _prepare_config(model_root: Path, layout: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    config = yaml.safe_load((model_root / layout["config"]).read_text(encoding="utf-8")) or {}
    phonemes = next(iter(sorted(model_root.rglob("phonemes.txt"))), None)
    save_dir = phonemes.parent if phonemes else model_root
    config.setdefault("output", {})["save_dir"] = str(save_dir)
    return config, save_dir


def _load(model: dict[str, Any], device: str):
    import torch  # noqa: PLC0415
    from model import BIOPhonemeTagger  # noqa: PLC0415  (WFL-ASR)
    from utils import load_langs, load_phoneme_list, load_phoneme_merge_map  # noqa: PLC0415

    root = Path(model["path"])
    layout = model.get("layout", {})
    key = f"{root}|{device}"
    if key in _cache:
        return _cache[key]
    _cache.clear()
    rt.free_memory()
    cfg, save_dir = _prepare_config(root, layout)
    labels = load_phoneme_list(str(save_dir / "phonemes.txt"))
    merge_path = save_dir / "phoneme_merge_map.json"
    merge_map = load_phoneme_merge_map(str(merge_path)) if merge_path.exists() else None
    langs_path = save_dir / "langs.txt"
    id2lang = {v: k for k, v in load_langs(str(langs_path)).items()} if langs_path.exists() else {}
    lang_ph_path = save_dir / "lang_phonemes.json"
    lang_phonemes = json.loads(lang_ph_path.read_text(encoding="utf-8")) if lang_ph_path.exists() else {}
    # the network loads Whisper from "encoder" in the current folder (downloading it there when missing): use the
    # one saved next to the model by training when there is one, so it needn't be downloaded
    encoder_home = next((d for d in (save_dir, root, root.parent, save_dir.parent)
                         if (d / "encoder" / "config.json").is_file()), None)
    here = os.getcwd()
    try:
        if encoder_home is not None:
            os.chdir(encoder_home)
        net = BIOPhonemeTagger(cfg, labels)
    finally:
        os.chdir(here)
    net = net.to(device)
    net.eval()
    data = torch.load(str(root / layout["checkpoint"]), map_location=device, weights_only=False)
    state = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
    # Lightning checkpoints keep the network under "model."
    if any(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    net.load_state_dict(state)
    _cache[key] = (net, cfg, merge_map, id2lang, lang_phonemes)
    return _cache[key]


class _LanguageOnly:
    """The network with the logits of labels that the selected language never uses pushed down (for decoding
    without text): the language's phonemes from lang_phonemes.json, "O" and the shared AP SP cl vf exh."""

    def __init__(self, net, phonemes):
        import torch  # noqa: PLC0415

        keep = set(phonemes) | {"AP", "SP", "cl", "vf", "exh"}
        self.id2label = net.id2label
        self.net = net
        self.mask = torch.tensor([not (lab == "O" or lab[2:] in keep) for _, lab in sorted(net.id2label.items())],
                                 device=next(net.parameters()).device)

    def __call__(self, *args, **kwargs):
        logits, offsets = self.net(*args, **kwargs)
        return logits.masked_fill(self.mask, -1e4), offsets


_PAUSES = {"SP", "AP", "O"}


def _native(canon: str, lang: str, merge_map: dict | None, primary: str, prefix: str) -> str:
    """The model's (merged) label -> the language's own name, with "lang/" in front for other languages
    (prefix "auto"), always ("always") or never ("never"); pauses never get a prefix."""
    name = (merge_map or {}).get(canon, {}).get(lang, canon)
    if canon in _PAUSES or prefix == "never" or (prefix == "auto" and lang == primary):
        return name
    return f"{lang}/{name}"


def _multilang_blind(net, cfg, audio, sr, device, langs, lang_ids, lang_phonemes, merge_map, viterbi_bias, prefix):
    """No text, several languages: phrases (split at pauses of a first pass in the main language) each get the
    language whose own labels explain the frames best (mean log-probability of the best path inside that language's
    labels, with that language's embedding); no mixing inside a phrase. Returns (segments with native names,
    [(start, end, language) per phrase]).""" 
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from infer import continuous_segments, viterbi_decode, window_logits  # noqa: PLC0415  (WFL-ASR)
    from utils import decode_bio_tags  # noqa: PLC0415

    fd = cfg["data"]["frame_duration"]
    overlap = (cfg.get("postprocess") or {}).get("chunk_overlap", 4.0)
    duration = len(audio) / sr
    audio = audio / (np.max(np.abs(audio)) + 1e-8)  # as process_audio
    id2label = net.id2label
    labels = [id2label[i] for i in range(len(id2label))]
    label_index = {lab: i for i, lab in enumerate(labels)}
    allowed, logits, offsets = {}, {}, {}
    for lang in langs:
        keep = set(lang_phonemes.get(lang, [])) | {"AP", "SP", "cl", "vf", "exh"}
        allowed[lang] = torch.tensor([lab == "O" or lab[2:] in keep for lab in labels])
        lt = torch.tensor([lang_ids[lang]], dtype=torch.long, device=device)
        logits[lang], offsets[lang] = window_logits(net, audio, sr, cfg, device, lt, overlap_sec=overlap)
    primary = langs[0]
    n = len(logits[primary])
    masked = {l: logits[l].masked_fill(~allowed[l], -1e4) for l in langs}
    first = decode_bio_tags(viterbi_decode(masked[primary], id2label, viterbi_bias=viterbi_bias), fd, offsets[primary])
    # phrases = runs of non-pause segments; a pause of >= 0.3 s ends a phrase
    phrases, cur = [], None
    for s, e, p in first:
        if p in _PAUSES and e - s >= 0.3:
            if cur:
                phrases.append(cur)
            cur = None
        elif p not in _PAUSES:
            cur = [s, e] if cur is None else [cur[0], e]
    if cur:
        phrases.append(cur)
    frame_lang = [primary] * n
    choice = []
    for s, e in phrases:
        a, b = max(0, int(s / fd)), min(n, max(int(e / fd), int(s / fd) + 1))
        best, score = primary, -1e9
        for lang in langs:  # mean log-probability of the language's own best path (Viterbi inside its labels)
            lp = torch.log_softmax(logits[lang][a:b].float(), dim=-1)
            path = viterbi_decode(masked[lang][a:b], id2label, viterbi_bias=viterbi_bias)
            idx = torch.tensor([label_index[t] for t in path])
            sc = float(lp[torch.arange(len(idx)), idx].mean())
            if sc > score:
                best, score = lang, sc
        choice.append((s, e, best))
        for f in range(a, b):
            frame_lang[f] = best
    stitched = torch.stack([masked[frame_lang[f]][f] for f in range(n)])
    offs = None
    if offsets[primary] is not None:
        offs = torch.stack([offsets[frame_lang[f]][f] for f in range(n)])
    segs = decode_bio_tags(viterbi_decode(stitched, id2label, viterbi_bias=viterbi_bias), fd, offs)
    out = []
    for s, e, p in segs:
        lang = frame_lang[min(n - 1, max(0, int((s + e) / 2 / fd)))]
        out.append((s, e, _native(p, lang, merge_map, primary, prefix)))
    return continuous_segments(out, duration), choice


def _multilang_text(net, cfg, audio, sr, device, phones, phone_lang, primary, lang2id):
    """Text alignment where phonemes belong to different languages: a first alignment with the main language's
    embedding finds where each phoneme roughly is; every frame then takes the logits computed with the embedding of
    the language of the phoneme it belongs to (pauses: of the phoneme before), and the alignment is repeated."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from infer import continuous_segments, forced_align_viterbi, window_logits  # noqa: PLC0415  (WFL-ASR)
    from utils import decode_bio_tags  # noqa: PLC0415

    fd = cfg["data"]["frame_duration"]
    pp = cfg.get("postprocess") or {}
    fa = pp.get("forced_alignment_args", {})
    kw = dict(self_loop_penalty=fa.get("self_loop_penalty", -4.6), forward_penalty=fa.get("forward_penalty", -0.6),
              skip_penalty=fa.get("skip_penalty", -2.3))
    duration = len(audio) / sr
    audio = audio / (np.max(np.abs(audio)) + 1e-8)  # as process_audio
    logits, offsets = {}, {}
    for lang in dict.fromkeys([primary] + phone_lang):
        lt = torch.tensor([lang2id[lang]], dtype=torch.long, device=device)
        logits[lang], offsets[lang] = window_logits(net, audio, sr, cfg, device, lt,
                                                    overlap_sec=pp.get("chunk_overlap", 4.0))
    n = len(logits[primary])

    def align(lg, of):
        return continuous_segments(decode_bio_tags(forced_align_viterbi(lg, net.id2label, phones, **kw), fd, of),
                                   duration)

    first = align(logits[primary], offsets[primary])
    if len(first) != len(phones) or len(logits) == 1:
        return first
    frame_lang, last = [primary] * n, primary
    for (s, e, _), lang, ph in zip(first, phone_lang, phones):
        lang = last if ph in _PAUSES else lang
        last = lang
        for f in range(max(0, int(s / fd)), min(n, int(np.ceil(e / fd)))):
            frame_lang[f] = lang
    lg = torch.stack([logits[frame_lang[f]][f] for f in range(n)])
    of = torch.stack([offsets[frame_lang[f]][f] for f in range(n)]) if offsets[primary] is not None else None
    return align(lg, of)


def _confidences(net, cfg, audio, sr, device, lang_id, segments, to_model, lang2id) -> list[float | None]:
    """Per segment: the mean probability (B- + I- of its label) the network gives it over its frames, computed as
    process_audio computes the logits (28 s chunks, the language's embedding)."""
    import math  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    fd = cfg["data"]["frame_duration"]
    a = audio / (np.max(np.abs(audio)) + 1e-8)
    lang = torch.tensor([lang_id if lang_id is not None else 0], dtype=torch.long).to(device)
    step = int(28.0 * sr)
    parts = []
    for start in range(0, len(a), step):
        chunk = a[start:start + step]
        frames = math.ceil(len(chunk) / sr / fd)
        if len(chunk) < 1600:
            chunk = np.pad(chunk, (0, 1600 - len(chunk)))
        with torch.no_grad():
            lg, _ = net(torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device), lang,
                        lengths=torch.tensor([frames], device=device))
        parts.append(torch.softmax(lg.squeeze(0).float(), -1).cpu())
    if not parts:
        return [None] * len(segments)
    p = torch.cat(parts).numpy()
    index = {lab: i for i, lab in net.id2label.items()}
    out: list[float | None] = []
    for s, e, name in segments:
        nm = name.split("/", 1)[1] if "/" in name and name.split("/", 1)[0] in lang2id else name
        canon = to_model.get(nm, nm)
        cols = [index[c] for c in (f"B-{canon}", f"I-{canon}") if c in index]
        f0 = min(len(p) - 1, max(0, int(s / fd)))
        f1 = min(len(p), max(f0 + 1, int(round(e / fd))))
        out.append(round(float(p[f0:f1, cols].sum(1).mean()), 4) if cols else None)
    return out


@rt.method()
def segment(
    model: dict[str, Any],
    items: list[dict[str, Any]],
    lang_id: int | None = None,
    decoder: str = "viterbi",
    viterbi_bias: float = 5.0,
    silence_threshold: float = 0.005,
    min_silence_duration: float = 0.5,
    silence_phoneme: str = "SP",
    extra_lang_ids: list[int] | None = None,
    language_prefix: str = "auto",
    confidence: bool = False,
    **_ignored: Any,
) -> list[dict[str, Any]]:
    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415
    from infer import apply_hard_silence, continuous_segments, process_audio  # noqa: PLC0415  (WFL-ASR)
    from utils import merge_adjacent_segments  # noqa: PLC0415

    device = rt.device()
    net, cfg, merge_map, id2lang, lang_phonemes = _load(model, device)
    lang_name = id2lang.get(lang_id) if lang_id is not None else None
    # multilingual models: phonemes given in the language's own names -> the model's (merged) names; decoding
    # without text only among the language's own labels (no-op for one-language models)
    to_model = {native: canon for canon, per in (merge_map or {}).items() for lang, native in per.items()
                if lang == lang_name}
    blind_net = _LanguageOnly(net, lang_phonemes[lang_name]) if (
        len(id2lang) > 1 and lang_name in lang_phonemes) else net
    # several languages in one song (multilingual models only): the main one first, then the others
    lang2id = {v: k for k, v in id2lang.items()}
    extra = [id2lang[i] for i in (extra_lang_ids or []) if i in id2lang and id2lang[i] != lang_name]
    multi = [lang_name] + [lg for lg in dict.fromkeys(extra) if lg in lang_phonemes] if (
        lang_name in lang_phonemes and len(id2lang) > 1) else [lang_name]
    to_model_of = {lg: {native: canon for canon, per in (merge_map or {}).items() for l2, native in per.items()
                        if l2 == lg} for lg in lang2id}
    results = []
    for index, item in enumerate(items):
        name = item.get("name") or Path(item["audio"]).stem
        rt.progress(index / len(items), f"Segmenting {name}")
        try:
            audio, sr = sf.read(item["audio"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            target_sr = cfg["data"]["sample_rate"]
            if sr != target_sr:
                audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, target_sr).numpy()
                sr = target_sr
            tokens = [p for p in (item.get("phonemes") or []) if p]
            prefixed = any(p.split("/", 1)[0] in lang2id for p in tokens if "/" in p)
            extra_info: dict[str, Any] = {}
            if tokens and prefixed:
                # text with other-language phonemes written "lang/phoneme": each one goes to its language's label;
                # the output keeps exactly the input names (the prefix is dropped only with language_prefix="never")
                canon, tok_lang = [], []
                for p in tokens:
                    lg, _, ph = p.partition("/") if "/" in p and p.split("/", 1)[0] in lang2id else (lang_name, "", p)
                    canon.append(to_model_of.get(lg, {}).get(ph, ph))
                    tok_lang.append(lg)
                phones = canon
                if len(id2lang) > 1 and lang_name is not None:
                    segments = _multilang_text(net, cfg, audio, sr, device, canon, tok_lang, lang_name, lang2id)
                else:
                    segments = process_audio(net, audio, sr, cfg, device, lang_id=lang_id, phones=phones,
                                             decoder=decoder, viterbi_bias=viterbi_bias)
                if len(segments) == len(tokens):
                    names = [t.split("/", 1)[1] if language_prefix == "never" and "/" in t else t for t in tokens]
                    segments = [(s, e, nm) for (s, e, _), nm in zip(segments, names)]
                else:  # should not happen (every phoneme gets a frame); fall back to the main language's names
                    segments = [(s, e, _native(p, lang_name, merge_map, lang_name, "never")) for s, e, p in segments]
            elif not tokens and len(multi) > 1:
                phones = None
                segments, choice = _multilang_blind(net, cfg, audio, sr, device, multi, lang2id, lang_phonemes,
                                                    merge_map, viterbi_bias, language_prefix)
                extra_info["languages"] = [[float(s), float(e), lg] for s, e, lg in choice]
            else:
                phones = [to_model.get(p, p) for p in tokens] or None
                segments = process_audio(
                    net if phones else blind_net, audio, sr, cfg, device, lang_id=lang_id, merge_map=merge_map,
                    lang_name=lang_name,
                    phones=phones, decoder=decoder, viterbi_bias=viterbi_bias,
                )
            mode = (cfg.get("postprocess") or {}).get("merge_segments", "right")
            if mode != "none":
                segments = merge_adjacent_segments(segments, mode)
            if phones is None:
                segments = apply_hard_silence(segments, audio, sr, threshold=silence_threshold,
                                              min_duration=min_silence_duration, silence_phoneme=silence_phoneme)
            segments = continuous_segments(segments, len(audio) / sr)
            conf = _confidences(net, cfg, audio, sr, device, lang_id, segments, to_model, lang2id) if confidence else None
            results.append({"ok": True, "phones": [[float(s), float(e), str(p)] + ([conf[i]] if conf else [])
                                                   for i, (s, e, p) in enumerate(segments)], **extra_info})
        except Exception as e:  # noqa: BLE001
            if rt.is_oom(e):
                rt.free_memory()
            results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
    rt.progress(1.0, "Segmented")
    return results


if __name__ == "__main__":
    rt.run()

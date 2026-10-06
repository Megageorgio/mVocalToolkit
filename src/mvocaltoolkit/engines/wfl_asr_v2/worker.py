"""WFL-ASR (refactor branch) engine worker: the same steps as its infer.py main(), for a list of files."""

from __future__ import annotations

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
    _cache[key] = (net, cfg, merge_map, id2lang)
    return _cache[key]


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
    **_ignored: Any,
) -> list[dict[str, Any]]:
    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415
    from infer import apply_hard_silence, continuous_segments, process_audio  # noqa: PLC0415  (WFL-ASR)
    from utils import merge_adjacent_segments  # noqa: PLC0415

    device = rt.device()
    net, cfg, merge_map, id2lang = _load(model, device)
    lang_name = id2lang.get(lang_id) if lang_id is not None else None
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
            phones = [p for p in (item.get("phonemes") or []) if p] or None
            segments = process_audio(
                net, audio, sr, cfg, device, lang_id=lang_id, merge_map=merge_map, lang_name=lang_name,
                phones=phones, decoder=decoder, viterbi_bias=viterbi_bias,
            )
            mode = (cfg.get("postprocess") or {}).get("merge_segments", "right")
            if mode != "none":
                segments = merge_adjacent_segments(segments, mode)
            if phones is None:
                segments = apply_hard_silence(segments, audio, sr, threshold=silence_threshold,
                                              min_duration=min_silence_duration, silence_phoneme=silence_phoneme)
            segments = continuous_segments(segments, len(audio) / sr)
            results.append({"ok": True, "phones": [[float(s), float(e), str(p)] for s, e, p in segments]})
        except Exception as e:  # noqa: BLE001
            if rt.is_oom(e):
                rt.free_memory()
            results.append({"ok": False, "error": f"{type(e).__name__}: {e}"})
    rt.progress(1.0, "Segmented")
    return results


if __name__ == "__main__":
    rt.run()

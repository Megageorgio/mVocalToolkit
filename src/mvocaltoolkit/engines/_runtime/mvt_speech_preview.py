"""Local exported speech-package previews inside the matching isolated engine."""
import json
from pathlib import Path


def render(kind, model_dir, out_dir, text, speaker="", language="Auto", reference="", base_model="", device="auto", seed=42,
           steps=10, guidance=2., temperature=.9, max_tokens=1024):
    import torch
    import numpy as np
    import soundfile as sf
    from mvt_train_common import device as choose_device
    root, out = Path(model_dir).resolve(), Path(out_dir).resolve()
    if not root.is_dir() or out == root or root in out.parents or out in root.parents:
        raise ValueError("Choose a local exported package and separate output folder")
    if not text.strip() or len(text) > 4000 or not 1 <= max_tokens <= 8192 or not 1 <= steps <= 1000 or not 0 < temperature <= 2 or not 0 < guidance <= 10:
        raise ValueError("Invalid text or speech-generation limits")
    resolved = choose_device(device)
    references = json.loads((root / "speaker-references.json").read_text(encoding="utf-8")) if (root / "speaker-references.json").is_file() else {}
    with torch.random.fork_rng(devices=[torch.device(resolved)] if resolved.startswith("cuda") else []), torch.inference_mode():
        torch.manual_seed(seed)
        if kind == "qwen_tts_voice":
            from qwen_tts import Qwen3TTSModel
            model = Qwen3TTSModel.from_pretrained(str(root), device_map=resolved, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True)
            available = model.get_supported_speakers()
            selected = speaker.lower() if speaker else next(iter(available), "")
            if selected not in available: raise ValueError(f"Choose a trained speaker: {available}")
            samples, rate = model.generate_custom_voice(text=text, speaker=selected, language=language or "Auto", max_new_tokens=max_tokens, temperature=temperature)
            audio = samples[0]
        elif kind == "voxcpm_voice":
            from voxcpm import VoxCPM
            lora = root / "lora_config.json"
            kwargs = dict(enable_denoiser=False, optimize=False, device=resolved)
            if lora.is_file():
                config = json.loads(lora.read_text(encoding="utf-8"))
                base = Path(base_model or config.get("base_model", ""))
                if not base.is_dir(): raise ValueError("Choose the local VoxCPM base model used by this LoRA")
                from voxcpm.core import LoRAConfig
                kwargs.update(lora_config=LoRAConfig(**config["lora_config"]), lora_weights_path=str(root))
            else: base = root
            model = VoxCPM(str(base), **kwargs)
            if not reference and references:
                if speaker and speaker not in references: raise ValueError(f"Choose a trained speaker: {list(references)}")
                entry = references[speaker or next(iter(references))]
                path = (root / entry["audio"]).resolve()
                if not path.is_relative_to(root): raise ValueError("Reference path escaped the package")
                reference = str(path)
            audio = model.generate(text=text, reference_wav_path=reference or None, inference_timesteps=steps, cfg_value=guidance,
                max_len=max_tokens, normalize=False, denoise=False, retry_badcase=False)
            rate = model.tts_model.sample_rate
        else: raise ValueError("Choose Qwen3-TTS or VoxCPM2")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(audio) or not np.isfinite(audio).all(): raise ValueError("Speech model produced invalid audio")
    out.mkdir(parents=True, exist_ok=False)
    sf.write(out / "speech.wav", audio, rate, subtype="FLOAT")
    result = dict(files=[str(out / "speech.wav")], out_dir=str(out), sample_rate=rate, seconds=len(audio) / rate)
    (out / "render-report.json").write_text(json.dumps(dict(**result, text=text, speaker=speaker, model=str(root), seed=seed), ensure_ascii=False, indent=1), encoding="utf-8")
    return result

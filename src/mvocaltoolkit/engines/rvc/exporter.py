"""Export an RVC v2 generator and optional retrieval index, without optimizer state."""
from pathlib import Path
import json
import shutil
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffsinger"))
from checkpoint_store import listing


def export_model(run: Path, out: Path, step=None):
    import torch
    candidates = [c for c in listing(run) if step is None or c["step"] == step]
    if not candidates:
        raise ValueError("The selected RVC checkpoint is unavailable")
    state = torch.load(candidates[-1]["path"], map_location="cpu", weights_only=True)
    if state.get("format") != "rvc-applio-v1":
        raise ValueError("Not an RVC training checkpoint")
    native = state["native_config"]
    h, d = native["model"], native["data"]
    from rvc.train.process.extract_model import replace_keys_in_dict
    weights = {k: v.half().cpu() for k, v in state["g"].items() if "enc_q" not in k}
    # Both old RVC clients and Applio understand the original weight norm names.
    weights = replace_keys_in_dict(replace_keys_in_dict(weights, ".parametrizations.weight.original1", ".weight_v"),
                                   ".parametrizations.weight.original0", ".weight_g")
    model = {"weight": weights, "config": [d["filter_length"] // 2 + 1, 32, h["inter_channels"], h["hidden_channels"],
        h["filter_channels"], h["n_heads"], h["n_layers"], h["kernel_size"], h["p_dropout"], h["resblock"],
        h["resblock_kernel_sizes"], h["resblock_dilation_sizes"], h["upsample_rates"], h["upsample_initial_channel"],
        h["upsample_kernel_sizes"], int(state["g"]["emb_g.weight"].shape[0]), h["gin_channels"], d["sample_rate"]],
        "sr": d["sample_rate"], "f0": 1, "version": "v2", "vocoder": "HiFi-GAN", "embedder_model": "contentvec",
        "speakers_id": 1, "model_name": run.name, "step": state["step"], "epoch": state["epoch"]}
    out = out.resolve(); out.mkdir(parents=True, exist_ok=True)
    temporary = out / "voice.pth.tmp"
    try:
        torch.save(model, temporary)
        os.replace(temporary, out / "voice.pth")
    finally:
        temporary.unlink(missing_ok=True)
    if (run / "voice.index").is_file():
        shutil.copy2(run / "voice.index", out / "voice.index")
    data = json.loads((run / "recordings.json").read_text(encoding="utf-8"))
    provenance = {"speaker": data["speaker"], "recording_licenses": sorted({r["license"] for r in data["recordings"]}),
                  "base_generator": data["settings"]["base_generator"] or ("RVC HiFi-GAN pretrained" if data["settings"]["pretrained"] else "random initialization"),
                  "source": "IAHispano/Applio@d36bde17aede51e0379cfb46d13e75e380b3a0be", "step": state["step"]}
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.txt").write_text("RVC v2 / ContentVec / HiFi-GAN voice conversion model. Use speaker ID 0.\n"
        "voice.pth contains inference weights; voice.index is optional retrieval data.\n"
        "Use with a compatible RVC client such as Applio. This is not an OpenUtau DiffSinger voicebank.\n"
        "heldout-original/generated previews use recordings excluded from training when validation is enabled.\n"
        "Other original/generated previews use training examples; spectral error does not measure perceived quality.\n"
        "Check the recording and selected base-model licenses before distribution.\n", encoding="utf-8")
    return {"out_dir": str(out), "files": [str(p) for p in out.iterdir() if p.is_file()]}

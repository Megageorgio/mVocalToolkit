"""SOFA weights and a stateless OpenUtau-style greedy ONNX transducer."""
import json
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "sofa"))
sys.path.insert(0, str(HERE.parent / "diffsinger"))
from g2p_model import _instantiate
from checkpoint_store import listing


def load_model(run: Path, step=None):
    data = json.loads((run / "dictionary-data.json").read_text(encoding="utf-8"))
    choices = [c for c in listing(run) if step is None or c["step"] == step]
    if not choices:
        raise ValueError("No matching G2P checkpoint")
    selected = max(choices, key=lambda c: c["step"])
    model = _instantiate(data["model"])
    model.load_state_dict(torch.load(selected["path"], map_location="cpu", weights_only=False)["state_dict"])
    return model.eval(), data, selected


class Greedy(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src, tgt, t):
        memory = self.model.encoder(src)[:, t[0]].unsqueeze(1)
        decoded, _ = self.model.decoder.lstm(self.model.decoder.emb(tgt))
        logits = self.model.decoder.joint(memory, self.model.decoder.fc(decoded[:, -1:]))
        return logits[0, 0].argmax(dim=-1).int()


def export_model(run: Path, out: Path, step=None):
    model, data, selected = load_model(run, step)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "config.yaml", out / "cfg.yaml")
    torch.save(model.state_dict(), out / "model.ptsd")
    (out / "dict.txt").write_text("".join(w + "  " + " ".join(p) + "\n" for w, p in data["entries"]), encoding="utf-8")
    metadata = {"graphemes": data["model"]["encoder"]["graphemes"], "phonemes": data["model"]["decoder"]["phonemes"],
                "lowercase": data["lowercase"], "normalization": "NFC", "blank": 2, "bos": 2,
                "language": data["language"], "step": selected["step"], "dictionary_sha256": data["dictionary_sha256"]}
    (out / "vocabulary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1), encoding="utf-8")
    graph = Greedy(model).eval()
    sample = (torch.full((1, 4), 4, dtype=torch.int32), torch.tensor([[2, 4]], dtype=torch.int32), torch.tensor([0], dtype=torch.int32))
    torch.onnx.export(graph, sample, str(out / "g2p.onnx"), input_names=["src", "tgt", "t"], output_names=["pred"],
                      dynamic_axes={"src": {1: "T"}, "tgt": {1: "U"}}, opset_version=17, dynamo=False)
    import onnx
    onnx.checker.check_model(str(out / "g2p.onnx"))
    import onnxruntime as ort
    session = ort.InferenceSession(str(out / "g2p.onnx"), providers=["CPUExecutionProvider"])
    # Exercise dynamic lengths, the frame cursor and non-empty decoder histories.
    for length in (1, 3, 6):
        src, tgt, t = torch.full((1, length), 4, dtype=torch.int32), torch.tensor([[2]], dtype=torch.int32), torch.tensor([length - 1], dtype=torch.int32)
        actual = session.run(None, {"src": src.numpy(), "tgt": tgt.numpy(), "t": t.numpy()})[0]
        with torch.no_grad():
            expected = graph(src, tgt, t).numpy()
        if not (actual == expected).all():
            raise ValueError("Exported ONNX prediction differs from the training model")
    (out / "README.txt").write_text("SOFA: cfg.yaml + model.ptsd + dict.txt.\nONNX: src/tgt/t int32 -> pred int32, blank/BOS=2.\n"
        "Use vocabulary.json in the consuming application. OpenUtau built-in phonemizers have fixed vocabularies; a custom vocabulary requires phonemizer integration.\n"
        "Phoneme vowel/consonant types must be supplied by the language author.\n", encoding="utf-8")
    if (run / "validation.json").is_file():
        shutil.copy2(run / "validation.json", out / "validation-latest.json")
    if (run / "validations" / f"{selected['step']}.json").is_file():
        shutil.copy2(run / "validations" / f"{selected['step']}.json", out / "validation-checkpoint.json")
    return {"out_dir": str(out), "step": selected["step"], "files": sorted(str(p) for p in out.iterdir() if p.is_file())}

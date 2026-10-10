"""Exports a SOFA model's G2P (OpenUtau-style, RNN-T; g2p/cfg.yaml + g2p/model.ptsd) for g2pflow's lstm converter,
so a TIFA model can carry it: encoder.onnx, decoder.onnx, char.json, phonemes.json in one folder.

    python tools/export_sofa_g2p.py <sofa model>/g2p <out folder>

Needs torch, onnx and pyyaml. Greedy decoding (beam_size: 1 in the lstm converter) gives exactly what the SOFA G2P
gives.

g2pflow calls encoder.onnx once with input_ids = [bos] + chars + [eos] -> (encoder_outputs, hidden, cell), then
decoder.onnx per step with (decoder_input, hidden, cell, encoder_outputs) -> (logits, hidden, cell, extra), stopping
at <eos>. The RNN-T search (skip encoder frames while the joint predicts the blank) is done inside one decoder step:
the frame index travels in an extra column of `cell`.
"""
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mvocaltoolkit" / "engines" / "sofa"))
from g2p_model import BOS_IDX, EOS_IDX, load_g2p  # noqa: E402

src_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
g2p = load_g2p(src_dir / "cfg.yaml", src_dir / "model.ptsd")
enc, dec = g2p.encoder, g2p.decoder
L, H = dec.num_layers, dec.d_hidden
NEG = -1e9


g2p.requires_grad_(False)


class EncoderWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = enc

    def forward(self, input_ids):
        mem = self.enc(input_ids[:, 1:-1])  # the SOFA model reads the letters without bos/eos
        h = torch.zeros(L, 1, H)
        c = torch.zeros(L, 1, H + 1)  # last column of layer 0: the encoder frame
        return mem, h, c


class DecoderWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = dec

    def forward(self, decoder_input, hidden, cell, encoder_outputs):
        dec = self.dec
        t = cell[0, 0, H]
        c = cell[:, :, :H]
        x, (h2, c2) = dec.lstm(dec.emb(decoder_input), (hidden, c))
        d = dec.fc(x)[0, 0]  # [d_model]
        mem = encoder_outputs[0]  # [T, d_model]
        n = mem.shape[0]
        j = dec.joint.project_layer(dec.joint.tanh(dec.joint.forward_layer(
            torch.cat([mem, d.unsqueeze(0).expand(n, -1)], dim=-1))))  # [T, V]
        pred = torch.argmax(j, dim=-1)
        frames = torch.arange(n, dtype=torch.float32)
        valid = (frames >= t) & (pred != BOS_IDX)
        first = torch.min(torch.where(valid, frames, torch.full_like(frames, 1e6)))
        found = first < 1e5
        f = torch.clamp(first, max=n - 1).long()
        emit = j[f].clone()
        emit[BOS_IDX] = NEG
        stop = torch.full_like(emit, NEG)
        stop[EOS_IDX] = 0.0
        logits = torch.where(found, emit, stop).reshape(1, 1, -1)
        new_h = torch.where(found, h2, hidden)
        new_t = torch.where(found, first, t)
        new_c = torch.cat([torch.where(found, c2, c), cell[:, :, H:]], dim=-1).clone()
        new_c[0, 0, H] = new_t
        return logits, new_h, new_c, j


with torch.no_grad():
    ids = torch.tensor([[BOS_IDX, 5, 6, 7, EOS_IDX]], dtype=torch.int64)
    torch.onnx.export(EncoderWrap(), (ids,), str(out_dir / "encoder.onnx"), input_names=["input_ids"],
                      output_names=["encoder_outputs", "hidden", "cell"],
                      dynamic_axes={"input_ids": {1: "n"}, "encoder_outputs": {1: "t"}}, opset_version=17, dynamo=False)
    mem, h, c = EncoderWrap()(ids)
    torch.onnx.export(DecoderWrap(), (torch.tensor([[BOS_IDX]]), h, c, mem), str(out_dir / "decoder.onnx"),
                      input_names=["decoder_input", "hidden", "cell", "encoder_outputs"],
                      output_names=["logits", "hidden_out", "cell_out", "joint"],
                      dynamic_axes={"encoder_outputs": {1: "t"}, "joint": {0: "t"}}, opset_version=17, dynamo=False)

(out_dir / "char.json").write_text(json.dumps({g: i for i, g in enumerate(enc.graphemes)}, ensure_ascii=False, indent=1), encoding="utf-8")
(out_dir / "phonemes.json").write_text(json.dumps({p: i for i, p in enumerate(dec.phonemes)}, ensure_ascii=False, indent=1), encoding="utf-8")
print("exported to", out_dir)

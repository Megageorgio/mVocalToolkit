"""OpenUtau-style G2P model (RNN-T like LSTM encoder/decoder), used for words missing in a SOFA dictionary.

Model folder layout (as in LabelMakr models):
    g2p/cfg.yaml     training config with _target_ entries for G2p / Encoder / Decoder
    g2p/model.ptsd   state dict

Architecture follows OpenUtau's G2P (MIT License).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


class Joint(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, d_output: int):
        super().__init__()
        self.forward_layer = nn.Linear(d_model * 2, d_hidden, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(d_hidden, d_output, bias=True)

    def forward(self, enc_state: Tensor, dec_state: Tensor) -> Tensor:
        t, u = enc_state.size(1), dec_state.size(1)
        enc_state = enc_state.unsqueeze(2).expand([-1, -1, u, -1])
        dec_state = dec_state.unsqueeze(1).expand([-1, t, -1, -1])
        x = self.tanh(self.forward_layer(torch.cat((enc_state, dec_state), dim=-1)))
        return self.project_layer(x)


class Encoder(nn.Module):
    def __init__(self, graphemes: list, d_model: int, d_hidden: int, num_layers: int, dropout: float, **_: Any):
        super().__init__()
        self.graphemes = graphemes
        self.emb = nn.Embedding(len(graphemes), d_model)
        self.lstm = nn.LSTM(d_model, d_hidden // 2, num_layers=num_layers, batch_first=True, dropout=dropout,
                            bidirectional=True)
        self.fc = nn.Linear(d_hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.lstm(self.emb(x))
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, phonemes: list, d_model: int, d_hidden: int, num_layers: int, dropout: float, **_: Any):
        super().__init__()
        self.phonemes = phonemes
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.emb = nn.Embedding(len(phonemes), d_model)
        self.lstm = nn.LSTM(d_model, d_hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(d_hidden, d_model)
        self.joint = Joint(d_model, d_hidden, len(phonemes))

    def step(self, tgt: Tensor, memory: Tensor, t: Tensor, h: Tensor, c: Tensor):
        x = self.emb(tgt[:, -1:])
        mem = memory[:, t[0]].unsqueeze(1)
        x, (h, c) = self.lstm(x, (h, c))
        x = self.joint(mem, self.fc(x))
        x = torch.argmax(F.softmax(x, dim=-1), dim=-1).reshape(1, -1).int()
        return x, h, c


class G2p(nn.Module):
    def __init__(self, max_len: int, encoder: Encoder, decoder: Decoder, **_: Any):
        super().__init__()
        self.max_len = max_len
        self.encoder = encoder
        self.decoder = decoder
        self._index = {g: i for i, g in enumerate(encoder.graphemes)}

    @torch.no_grad()
    def predict_str(self, word: str) -> list[str]:
        device = next(self.parameters()).device
        src = torch.tensor([[self._index.get(ch, UNK_IDX) for ch in word]], device=device)
        tgt = torch.tensor([[BOS_IDX]], device=device)
        mem = self.encoder(src)
        t = torch.zeros(1, dtype=torch.int, device=device)
        h = torch.zeros((self.decoder.num_layers, 1, self.decoder.d_hidden), device=device)
        c = torch.zeros((self.decoder.num_layers, 1, self.decoder.d_hidden), device=device)
        while t[0] < src.shape[-1] and tgt.shape[1] < self.max_len:
            pred, new_h, new_c = self.decoder.step(tgt, mem, t, h, c)
            if pred.item() != BOS_IDX:
                tgt = torch.cat([tgt, pred], dim=-1)
                h, c = new_h, new_c
            else:
                t[0] += 1
        phonemes = [self.decoder.phonemes[i] for i in tgt[0, 1:].tolist()]
        return [p for p in phonemes if p not in _SPECIAL]


_SPECIAL = ["<unk>", "<pad>", "<bos>", "<eos>"]
_CLASSES = {"G2p": G2p, "Encoder": Encoder, "Decoder": Decoder}


def _instantiate(node: Any) -> Any:
    if isinstance(node, dict):
        target = node.get("_target_")
        args = {k: _instantiate(v) for k, v in node.items() if k != "_target_"}
        if target:
            name = str(target).split(".")[-1]
            if name not in _CLASSES:
                raise ValueError(f"Unsupported G2P class {target}")
            # configs use either encoder/decoder or Encoder/Decoder keys
            args = {k.lower() if k in ("Encoder", "Decoder") else k: v for k, v in args.items()}
            return _CLASSES[name](**args)
        return args
    if isinstance(node, list):
        return [_instantiate(v) for v in node]
    return node


def load_g2p(config: Path, weights: Path, device: str = "cpu") -> G2p:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    if "model" in cfg and isinstance(cfg["model"], dict) and "_target_" in cfg["model"]:
        cfg = cfg["model"]
    model = _instantiate(cfg)
    if not isinstance(model, G2p):
        raise ValueError("cfg.yaml doesn't describe a G2p model")
    state = torch.load(str(weights), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):  # a pickled module
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()

"""Pronunciation-model preflight and reproducible splits, without importing torch."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
import copy
import uuid
from pathlib import Path

from .dictionaries import read_entries

SPECIAL = ["<unk>", "<pad>", "<bos>", "<eos>"]


def prepare_dictionary(config: dict) -> dict:
    path = Path(str(config.get("dictionary") or ""))
    entries = read_entries(path, config.get("language") or None)
    if len(entries) > 200_000:
        raise ValueError("Use at most 200000 pronunciation entries per run")
    lowercase = bool(config.get("lowercase", True))
    grouped: dict[str, list[list[str]]] = {}
    for word, phones in entries:
        word = unicodedata.normalize("NFC", word)
        word = word.lower() if lowercase else word
        phones = [unicodedata.normalize("NFC", p) for p in phones]
        if not 1 <= len(word) <= 128 or not 1 <= len(phones) <= 128 or any(c.isspace() for c in word):
            raise ValueError("G2P needs single words and pronunciations of 1–128 symbols")
        if set(phones) & set(SPECIAL):
            raise ValueError("Dictionary phonemes must not use reserved <unk>/<pad>/<bos>/<eos> tokens")
        variants = grouped.setdefault(word, [])
        if phones not in variants:
            variants.append(phones)
    if len(grouped) < 2:
        raise ValueError("At least two distinct words are needed for a held-out validation split")
    fraction = float(config.get("validation_fraction", .1))
    if not math.isfinite(fraction) or not .01 <= fraction <= .5:
        raise ValueError("validation_fraction must be between 0.01 and 0.5")
    seed = int(config.get("seed", 42))
    ranked = sorted(grouped, key=lambda w: hashlib.sha256(f"{seed}:{w}".encode()).digest())
    val_count = min(len(grouped) - 1, max(1, round(len(grouped) * fraction)))
    validation_words = set(ranked[:val_count])
    train = [[w, p] for w, variants in sorted(grouped.items()) if w not in validation_words for p in variants]
    validation = [[w, grouped[w]] for w in sorted(validation_words)]
    epochs, batch = int(config.get("epochs", 50)), int(config.get("batch_size", 16))
    hidden, layers = int(config.get("hidden_size", 128)), int(config.get("layers", 2))
    lr, patience = float(config.get("learning_rate", .001)), int(config.get("patience", 8))
    if not 1 <= epochs <= 10000 or not 1 <= batch <= 256 or hidden not in (64, 128, 256, 512) or not 1 <= layers <= 4:
        raise ValueError("Invalid epochs, batch size, hidden size or layer count")
    if not math.isfinite(lr) or not 0 < lr <= .1 or not 0 <= patience <= 1000:
        raise ValueError("Invalid learning rate or early-stopping patience")
    graphemes = SPECIAL + sorted({c for w in grouped for c in w})
    phonemes = SPECIAL + sorted({p for v in grouped.values() for pron in v for p in pron})
    shared = {"d_model": hidden // 2, "d_hidden": hidden, "num_layers": layers, "dropout": .1 if layers > 1 else 0.0}
    model = {"_target_": "models.g2p_model.G2p", "max_len": 129,
             "encoder": {"_target_": "models.g2p_model.Encoder", "graphemes": graphemes, **shared},
             "decoder": {"_target_": "models.g2p_model.Decoder", "phonemes": phonemes, **shared}}
    base_weights = None
    added_chars, added_phones = [], []
    if config.get("base_model"):
        import yaml
        base = Path(config["base_model"]).expanduser().resolve()
        cfg = yaml.safe_load((base / "cfg.yaml").read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("Base cfg.yaml must contain a model object")
        cfg = cfg.get("model", cfg)
        if not isinstance(cfg, dict):
            raise ValueError("Base model configuration must be an object")
        model = copy.deepcopy(cfg)
        for upper, lower in (("Encoder", "encoder"), ("Decoder", "decoder")):
            if upper in model: model[lower] = model.pop(upper)
        if str(model.get("_target_", "")).split(".")[-1] != "G2p":
            raise ValueError("The base cfg.yaml must describe a compatible G2p RNN transducer")
        for name, field, required, class_name in (("encoder", "graphemes", graphemes, "Encoder"), ("decoder", "phonemes", phonemes, "Decoder")):
            node = model.get(name, {})
            inventory = node.get(field, [])
            if (str(node.get("_target_", "")).split(".")[-1] != class_name or
                    not isinstance(inventory, list) or inventory[:4] != SPECIAL or
                    not all(isinstance(v, str) for v in inventory) or len(inventory) != len(set(inventory))):
                raise ValueError("The base model must use the compatible grapheme/phoneme vocabulary")
            missing_tokens = sorted(set(required) - set(inventory))
            if missing_tokens and not config.get("expand_vocabulary", False):
                raise ValueError(f"Base {field} lacks: {' '.join(missing_tokens)}. Enable vocabulary expansion or use a matching dictionary")
            node[field] = [*inventory, *missing_tokens]
            if name == "encoder": added_chars = missing_tokens
            else: added_phones = missing_tokens
            if (not isinstance(node.get("d_hidden"), int) or not 2 <= node["d_hidden"] <= 2048 or node["d_hidden"] % 2
                    or not isinstance(node.get("d_model"), int) or not 1 <= node["d_model"] <= 2048
                    or not isinstance(node.get("num_layers"), int) or not 1 <= node["num_layers"] <= 8):
                raise ValueError("Invalid base-model recurrent dimensions")
        if model["encoder"]["d_model"] != model["decoder"]["d_model"]:
            raise ValueError("Base encoder and decoder must use the same d_model")
        model["max_len"] = max(int(model.get("max_len", 129)), 129)
        weight_file = base / "model.ptsd"
        if not weight_file.is_file():
            raise ValueError("Base folder must contain cfg.yaml and model.ptsd")
        with weight_file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        base_weights = {"path": str(weight_file), "sha256": digest}
    train_phones = {p for _, pron in train for p in pron}
    train_chars = {c for w, _ in train for c in w}
    warnings = []
    if len(grouped) < 100:
        warnings.append("Fewer than 100 distinct words: suitable for a smoke test, not reliable pronunciation")
    missing = sorted(set(phonemes[4:]) - train_phones)
    if missing:
        warnings.append("Phonemes only in validation: " + " ".join(missing))
    missing_chars = sorted(set(graphemes[4:]) - train_chars)
    if missing_chars:
        warnings.append("Graphemes only in validation: " + " ".join(missing_chars))
    if added_chars or added_phones:
        warnings.append("New vocabulary rows start with random weights; the old rows and recurrent layers are preserved")
    return {"model": model, "train": train, "validation": validation, "entries": [[w, p] for w, v in sorted(grouped.items()) for p in v],
            "epochs": epochs, "batch_size": batch, "learning_rate": lr, "patience": patience, "seed": seed,
            "lowercase": lowercase, "language": config.get("language", ""), "warnings": warnings,
            "words": len(grouped), "train_words": len(grouped) - val_count, "validation_words": val_count,
            "max_updates": epochs * math.ceil(len(train) / batch),
            "base_weights": base_weights, "added_graphemes": added_chars, "added_phonemes": added_phones,
            "dictionary_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def report(config: dict) -> dict:
    data = prepare_dictionary(config)
    return {k: v for k, v in data.items() if k not in {"train", "validation", "entries", "model"}}


def snapshot(config: dict, folder: Path) -> dict:
    data = prepare_dictionary(config)
    content = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (digest + ".json")
    if not path.is_file():
        temporary = folder / (digest + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return {"prepared_file": str(path), "prepared_sha256": digest,
            "data_report": {k: v for k, v in data.items() if k not in {"train", "validation", "entries", "model"}}}

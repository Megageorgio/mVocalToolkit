"""Offline preparation preflight. Does not start workers or download weights."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .dataset import AUDIO, _csv_rows, _label_for, _label_index
from .phonemizers import convert_label, profile


def languages(req) -> list[str]:
    codes = list(dict.fromkeys([req.language, *req.extra_languages]))
    if any(not c or c != c.lower().strip() or "/" in c or any(x.isspace() for x in c) for c in codes):
        raise ValueError("Use explicit lowercase language codes, for example ru, en, de")
    return codes


def model_languages(model) -> list[str]:
    return list(model_language_ids(model))


def model_language_ids(model) -> dict[str, int]:
    root = Path(model.path)
    path = (root / model.layout.get("config", "config.yaml")).parent / "langs.txt"
    if not path.is_file():
        path = root / "langs.txt"
    if not path.is_file():
        raise ValueError("WFL model has no langs.txt; language IDs cannot be verified")
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        code, separator, number = line.strip().partition(",")
        if not separator or not number.strip().isdigit() or code.strip() in result or int(number) in result.values():
            raise ValueError("WFL langs.txt must contain unique language,id rows")
        result[code.strip()] = int(number)
    return result


def validate_model(model, req, *, installed: bool = True) -> list[str]:
    codes = languages(req)
    available = model_languages(model) if installed else list(model.languages)
    if model.engine != "wfl_asr":
        raise ValueError("Automatic preparation requires a WFL-ASR model")
    if not available or set(codes) - set(available):
        raise ValueError(f"Selected model does not declare languages {codes}; supported: {available}")
    if installed and req.extra_languages and int(model.layout.get("generation", 1)) < 2:
        raise ValueError("Additional languages require a WFL-ASR generation 2 model (for example multi13)")
    return available


def preparation_check(tk, req) -> dict:
    root, out = Path(req.folder).expanduser().resolve(), Path(req.out_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Recording folder does not exist")
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Preparation output and source folders must not overlap")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Preparation output must be empty")
    for lang, identifier in req.phonemizer_profiles.items():
        profile(identifier, lang)
    if req.label_mode != "existing":
        if not req.label_model:
            raise ValueError("Select a WFL model for automatic labelling")
        languages(req)
        installed = tk.models.get_installed(req.label_model)
        entry = installed or tk.models.catalog.get(req.label_model)
        if entry is None:
            raise ValueError("WFL model is not installed or listed in the catalog")
        validate_model(entry, req, installed=installed is not None)
    files = sorted(p for p in (root.rglob("*") if req.recursive else root.glob("*")) if p.is_file() and p.suffix.lower() in AUDIO)
    if not files:
        raise ValueError("No WAV or FLAC recordings found")
    index, csv_cache = _label_index(root), {}
    problems, labelled, pending, inventory = [], 0, 0, Counter()
    for audio in files:
        if audio.parent not in csv_cache:
            csv_cache[audio.parent] = _csv_rows(audio.parent) or _csv_rows(root)
        label = _label_for(audio, csv_cache[audio.parent], index)
        if req.label_mode == "replace" or not label or not label.phones:
            if req.label_mode != "existing":
                pending += 1
            else:
                problems.append(dict(file=str(audio), code="no_labels", severity="error", problem="No phoneme labels; select automatic labelling for raw WAV"))
            continue
        labelled += 1
        try:
            converted, report = convert_label(label, req.language, req.phonemizer_profiles,
                alphabet=req.source_alphabet, prefix=req.language_prefix, overrides=req.phoneme_mappings,
                unknown_policy=req.unknown_phonemes)
            inventory.update(p.text for p in converted.phones)
            for field in ("unknown", "lossy", "extensions"):
                if report[field]:
                    problems.append(dict(file=str(audio), code=field, severity="warning", problem=str(report[field])))
        except ValueError as error:
            problems.append(dict(file=str(audio), code="phoneme_conversion", severity="error", problem=str(error)))
    return dict(ok=not any(p["severity"] == "error" for p in problems), files=len(files), labelled=labelled,
                pending_auto_labels=pending, phonemes=dict(inventory), profiles=req.phonemizer_profiles, problems=problems)

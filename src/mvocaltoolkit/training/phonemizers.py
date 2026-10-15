"""Versioned phoneme inventories and explicit WFL-to-voicebank conversion.

Model output is already in its native language alphabet. Canonical multi13 input
must be requested explicitly: guessing it would reverse German e/eh twice.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from ..labels import Interval, Label

DATA = Path(__file__).parent
PROFILES = {p["id"]: p for p in json.loads((DATA / "phonemizer_profiles.json").read_text(encoding="utf-8")) if p["vowels"]}
MULTI13 = json.loads((DATA / "wfl_multi13_map.json").read_text(encoding="utf-8"))["mapping"]
SPECIALS = {"SP", "AP", "cl", "vf", "exh", "vtrash", "ctrash"}
PAUSES = {"sp", "sil", "pau", "<sp>", "silence", ""}

# Native multi13 -> the selected phonemizer. Identity is only accepted when the
# symbol is in that phonemizer's declared alphabet. Reductions are reported.
RU_SOFT = {p + "y": p + p for p in "bdfgklmnprstvz"}
RU_STOCK = {**RU_SOFT, "ts": "c", "shy": "sch", "hy": "hh", "e": "ee", "o": "oo"}
RU_REDUCED = {"ax": "a", "x": "i", "ex": "ee"}
NATIVE_MAP = {
    "ru-openutau": {**RU_STOCK, **RU_REDUCED},
    "it-openutau": {"ee": "EE", "oo": "OO", "gl": "LL", "gn": "JJ", "ch": "tSS", "sh": "SS", "y": "j", "j": "dZZ"},
    "pt-openutau": {"an": "a~", "en": "e~", "in": "i~", "on": "o~", "un": "u~", "eh": "E", "oh": "O", "nh": "J", "lh": "L", "j": "Z", "sh": "S", "ch": "tS", "dj": "dZ", "y": "j"},
    "es-openutau": {"J": "gn", "H": "Y"},
}
# A stock Russian G2P can also drive an hhskt bank. Vowel allophones are
# necessarily folded; this is a compatibility option, never a training rewrite.
RU_STOCK_TO_HHSKT = {**{v: k for k, v in RU_STOCK.items()},
    "a": "a", "aa": "a", "ay": "ax", "ee": "e", "i": "x", "ii": "i", "ja": "a", "je": "e", "jo": "o", "ju": "u",
    "oo": "o", "u": "u", "uj": "u", "uu": "u", "y": "y", "yy": "y"}


def profile(identifier: str, language: str | None = None) -> dict:
    if identifier not in PROFILES:
        raise ValueError(f"Unknown phonemizer profile: {identifier}")
    result = PROFILES[identifier]
    if language and result["language"] != language:
        raise ValueError(f"Profile {identifier} is for {result['language']}, not {language}")
    return result


def default_profile(language: str) -> str:
    return next((p["id"] for p in PROFILES.values() if p["language"] == language and p["preferred"]),
                next((p["id"] for p in PROFILES.values() if p["language"] == language), ""))


def convert_label(label: Label, language: str, targets: dict[str, str], *,
                  alphabet: str = "target", prefix: bool = False,
                  overrides: dict[str, list[str]] | None = None,
                  unknown_policy: str = "error", merge_map: dict | None = None) -> tuple[Label, dict]:
    """Copy all tiers; preserve durations, confidences and existing word groups.

    Overrides use a full language/token key or token and may split a phone. Such
    boundaries are evenly allocated and explicitly marked as approximations.
    """
    if alphabet not in {"target", "wfl-native", "wfl-multi13-canonical"}:
        raise ValueError(f"Unknown source alphabet: {alphabet}")
    if unknown_policy not in {"error", "keep"}:
        raise ValueError("unknown_policy must be error or keep")
    for lang, identifier in targets.items():
        profile(identifier, lang)
    mapping = MULTI13 if merge_map is None else merge_map
    result = label.model_copy(deep=True)
    converted, unknown, lossy, extensions = Counter(), Counter(), Counter(), Counter()
    phones = []
    for phone in label.phones:
        lang, token = phone.text.split("/", 1) if "/" in phone.text else (language, phone.text)
        original = token
        if token.lower() in PAUSES:
            token = "SP"
        elif token.lower() in {"ap", "br", "<ap>"}:
            token = "AP"
        target = profile(targets[lang], lang) if lang in targets else None
        if alphabet == "wfl-multi13-canonical" and token not in SPECIALS:
            token = mapping.get(token, {}).get(lang, token)
        custom = (overrides or {}).get(phone.text, (overrides or {}).get(original))
        if custom is not None:
            if not custom or any(not p or any(c.isspace() for c in p) or "/" in p for p in custom):
                raise ValueError(f"Mapping for {phone.text} must contain non-empty unprefixed phonemes")
            tokens = list(custom)
            if len(tokens) > 1:
                lossy[f"{phone.text}: equally divided duration"] += 1
        else:
            tokens = [NATIVE_MAP.get(target["id"], {}).get(token, token) if target and alphabet != "target" else token]
            if target and target["id"] == "ru-openutau" and token in RU_REDUCED and alphabet != "target":
                lossy[f"{lang}/{token} -> {tokens[0]}: reduced vowel folded"] += 1
        for n, value in enumerate(tokens):
            if value not in SPECIALS and target and value not in target["vowels"] + target["consonants"]:
                unknown[f"{lang}/{value}"] += 1
            elif value not in SPECIALS and not target and alphabet != "target":
                unknown[f"{lang}/{value} (select target profile or keep custom alphabet)"] += 1
            if value in SPECIALS - {"SP", "AP"}:
                extensions[value] += 1
            name = f"{lang}/{value}" if (prefix or "/" in phone.text) and lang and value not in SPECIALS else value
            if name != phone.text:
                converted[f"{phone.text} -> {name}"] += 1
            data = dict(phone.data or {})
            data.setdefault("source_phoneme", phone.text)
            data["language"] = lang
            phones.append(Interval(start=phone.start + phone.duration * n / len(tokens),
                                   end=phone.start + phone.duration * (n + 1) / len(tokens), text=name,
                                   confidence=phone.confidence, data=data))
    result.tiers["phones"] = phones
    report = dict(converted=dict(converted), unknown=dict(unknown), lossy=dict(lossy), extensions=dict(extensions),
                  profiles=targets, source_alphabet=alphabet)
    if unknown and unknown_policy == "error":
        raise ValueError("Unmapped phonemes: " + ", ".join(unknown) + ". Add explicit mappings or select Keep custom phonemes.")
    return result, report


def vowel_inventory(targets: dict[str, str], prefix: bool) -> list[str]:
    return [f"{lang}/{p}" if prefix else p for lang, identifier in targets.items() for p in profile(identifier, lang)["vowels"]]


def compatibility_aliases(source_id: str, target_id: str) -> tuple[dict[str, str], list[str]]:
    """G2P source -> trained target. Only single-phone remaps OpenUtau supports."""
    source, target = profile(source_id), profile(target_id)
    if source["language"] != target["language"]:
        raise ValueError("Phonemizers must use the same language")
    aliases, missing = {}, []
    inventory = set(target["vowels"] + target["consonants"])
    for token in source["vowels"] + source["consonants"]:
        if source_id == "ru-openutau" and target_id == "ru-hhskt":
            value = RU_STOCK_TO_HHSKT.get(token, token)
        elif source_id == "ru-hhskt" and target_id == "ru-openutau":
            value = NATIVE_MAP[target_id].get(token, token)
        else:
            value = token
        if value in inventory:
            aliases[token] = value
        else:
            missing.append(token)
    return aliases, missing

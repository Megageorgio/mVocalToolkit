"""Post-processing rules for phoneme labels.

Rules are plain data, so GUIs can show them as toggles and users can write their own:

    {"op": "replace", "from": "pau", "to": "SP"}
    {"op": "merge_pair", "first": "uh", "second": "r", "into": "er"}
    {"op": "merge_duplicates"}
    {"op": "merge_short", "phones": ["hh", "h"], "max_dur": 0.009}           # merge into the previous phoneme
    {"op": "contextual", "phones": ["t", "d"], "prev": "vowel", "next": "vowel", "max_dur": 0.05, "to": "dx"}
    {"op": "min_duration", "min_dur": 0.005}                                # tiny phonemes are merged into neighbours

A rule set is {"vowels": [...], "rules": [...]}. Built-in sets are in RULE_SETS.
"""

from __future__ import annotations

from typing import Any

from ..labels import Interval, Label

ARPABET_VOWELS = [
    "a", "aa", "ae", "ah", "ao", "aw", "ax", "ay", "e", "eh", "er", "en", "eng", "ey", "el", "em", "ee",
    "i", "ih", "iy", "ix", "ii", "N", "nn", "o", "ow", "ox", "oy", "oo", "u", "uh", "uw", "uu", "ux",
]

RULE_SETS: dict[str, dict[str, Any]] = {
    # the label fixes LabelMakr applies to English
    "en_fixes": {
        "vowels": ARPABET_VOWELS,
        "rules": [
            {"op": "contextual", "phones": ["t", "d"], "prev": "vowel", "next": "vowel", "max_dur": 0.05, "to": "dx"},
            {"op": "contextual", "phones": ["t", "d"], "prev": ["r"], "next": "vowel", "max_dur": 0.05, "to": "dx"},
            {"op": "merge_pair", "first": "uh", "second": "r", "into": "er"},
            {"op": "merge_short", "phones": ["hh", "h"], "max_dur": 0.009},
            {"op": "merge_duplicates"},
        ],
    },
    "merge_duplicates": {"rules": [{"op": "merge_duplicates"}]},
    "cleanup": {"rules": [{"op": "replace", "from": "pau", "to": "SP"}, {"op": "replace", "from": "sil", "to": "SP"},
                          {"op": "replace", "from": "br", "to": "AP"}, {"op": "merge_duplicates", "phones": ["SP"]}]},
}


def apply_rules(label: Label, rules: list[dict[str, Any]], vowels: list[str] | None = None, tier: str = "phones") -> Label:
    phones = [iv.model_copy() for iv in label.tiers.get(tier, [])]
    vowel_set = set(vowels or ARPABET_VOWELS)
    for rule in rules:
        op = rule.get("op")
        handler = _OPS.get(op or "")
        if handler is None:
            raise ValueError(f"Unknown rule op: {op}")
        phones = handler(phones, rule, vowel_set)
    result = label.model_copy(deep=True)
    result.tiers[tier] = phones
    return result


def apply_rule_sets(label: Label, names: list[str], custom: list[dict[str, Any]] | None = None) -> Label:
    for name in names:
        if name not in RULE_SETS:
            raise ValueError(f"Unknown rule set: {name}")
        rule_set = RULE_SETS[name]
        label = apply_rules(label, rule_set["rules"], rule_set.get("vowels"))
    if custom:
        label = apply_rules(label, custom)
    return label


def _matches(phone: str | None, spec: Any, vowels: set[str]) -> bool:
    if spec is None:
        return True
    if phone is None:
        return False
    if spec == "vowel":
        return phone in vowels
    if spec == "consonant":
        return phone not in vowels and phone not in ("SP", "AP")
    if isinstance(spec, str):
        return phone == spec
    return phone in spec


def _merge(a: Interval, b: Interval, text: str) -> Interval:
    return Interval(start=a.start, end=b.end, text=text, confidence=_min_conf(a.confidence, b.confidence))


def _min_conf(a: float | None, b: float | None) -> float | None:
    values = [v for v in (a, b) if v is not None]
    return min(values) if values else None


def _op_replace(phones: list[Interval], rule: dict, _v: set[str]) -> list[Interval]:
    for iv in phones:
        if iv.text == rule["from"]:
            iv.text = rule["to"]
    return phones


def _op_merge_pair(phones: list[Interval], rule: dict, _v: set[str]) -> list[Interval]:
    out: list[Interval] = []
    i = 0
    while i < len(phones):
        if i + 1 < len(phones) and phones[i].text == rule["first"] and phones[i + 1].text == rule["second"]:
            out.append(_merge(phones[i], phones[i + 1], rule["into"]))
            i += 2
        else:
            out.append(phones[i])
            i += 1
    return out


def _op_merge_duplicates(phones: list[Interval], rule: dict, _v: set[str]) -> list[Interval]:
    only = set(rule.get("phones") or [])
    out: list[Interval] = []
    for iv in phones:
        if out and out[-1].text == iv.text and (not only or iv.text in only):
            out[-1] = _merge(out[-1], iv, iv.text)
        else:
            out.append(iv)
    return out


def _op_merge_short(phones: list[Interval], rule: dict, _v: set[str]) -> list[Interval]:
    targets = set(rule.get("phones") or [])
    max_dur = float(rule.get("max_dur", 0.01))
    out: list[Interval] = []
    for iv in phones:
        if out and (not targets or iv.text in targets) and iv.duration <= max_dur and out[-1].text not in ("SP", "AP"):
            out[-1] = _merge(out[-1], iv, out[-1].text)
        else:
            out.append(iv)
    return out


def _op_contextual(phones: list[Interval], rule: dict, vowels: set[str]) -> list[Interval]:
    targets = rule.get("phones") or []
    max_dur = rule.get("max_dur")
    for i, iv in enumerate(phones):
        if iv.text not in targets:
            continue
        prev = phones[i - 1].text if i > 0 else None
        nxt = phones[i + 1].text if i + 1 < len(phones) else None
        if not _matches(prev, rule.get("prev"), vowels) or not _matches(nxt, rule.get("next"), vowels):
            continue
        if max_dur is not None and iv.duration > float(max_dur):
            continue
        iv.text = rule["to"]
    return phones


def _op_min_duration(phones: list[Interval], rule: dict, _v: set[str]) -> list[Interval]:
    min_dur = float(rule.get("min_dur", 0.005))
    out: list[Interval] = []
    for iv in phones:
        if iv.duration < min_dur and out:
            out[-1] = Interval(start=out[-1].start, end=iv.end, text=out[-1].text, confidence=out[-1].confidence)
        else:
            out.append(iv)
    if len(out) > 1 and out[0].duration < min_dur:
        out[1] = Interval(start=out[0].start, end=out[1].end, text=out[1].text, confidence=out[1].confidence)
        out.pop(0)
    return out


_OPS = {
    "replace": _op_replace,
    "merge_pair": _op_merge_pair,
    "merge_duplicates": _op_merge_duplicates,
    "merge_short": _op_merge_short,
    "contextual": _op_contextual,
    "min_duration": _op_min_duration,
}

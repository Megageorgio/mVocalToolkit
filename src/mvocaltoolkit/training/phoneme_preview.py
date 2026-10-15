"""Read-only conversion examples with source times for targeted listening."""
from pathlib import Path
from .. import formats
from .dataset import AUDIO, _label_for, _label_index, _csv_rows
from .phonemizers import convert_label


def preview(req: dict) -> dict:
    path = Path(req["path"]).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Choose an audio file with labels or a label file")
    if path.suffix.lower() in AUDIO:
        label = _label_for(path, _csv_rows(path.parent), _label_index(path.parent))
        audio = path
    else:
        label = formats.read(path)
        audio = path.with_suffix(".wav")
    if label is None or not label.phones:
        raise ValueError("No timed phoneme labels; first prepare raw audio with WFL")
    converted, report = convert_label(label, req["language"], req.get("profiles", {}), alphabet=req.get("alphabet", "target"),
        prefix=True, overrides=req.get("mappings", {}), unknown_policy="keep")
    examples = [dict(source=p.data["source_phoneme"], target=p.text, start=p.start, end=p.end,
                     language=p.data["language"], confidence=p.confidence) for p in converted.phones[:2000]]
    return dict(report=report, examples=examples, total=len(converted.phones), truncated=len(converted.phones) > 2000,
                audio=str(audio) if audio.is_file() else None)


def suggest_merges(entries: list[dict]) -> dict:
    """Only exact user-specified IPA + feature matches; never applies a merge."""
    groups = {}
    for entry in entries:
        ipa, phone = str(entry.get("ipa", "")).strip(), str(entry.get("phone", "")).strip()
        if not ipa or "/" not in phone:
            continue
        # Palatalization, aspiration, length and other distinctions must remain
        # explicit in IPA/features. No transliteration/fuzzy matching is used.
        features = tuple(sorted((str(k), str(v)) for k, v in entry.get("features", {}).items()))
        groups.setdefault((ipa, features), []).append(phone)
    return dict(groups=[sorted(set(phones)) for phones in groups.values() if len({p.split('/')[0] for p in phones}) > 1],
                warning="Suggestions require phonetic review; they are not applied automatically")


def suggest_mappings(entries: list[dict], targets: list[dict]) -> dict:
    import unicodedata
    def key(row):
        return (unicodedata.normalize("NFC", str(row.get("ipa", "")).strip()),
            tuple(sorted((str(k), str(v)) for k, v in row.get("features", {}).items())))
    inventory = {}
    for row in targets:
        phone = str(row.get("phone", "")).split("/", 1)[-1]
        if not phone or any(c.isspace() for c in phone) or "/" in phone: raise ValueError("Target phones must be single symbols")
        if key(row)[0]: inventory.setdefault(key(row), set()).add(phone)
    mappings, ambiguous, missing = {}, {}, []
    for row in entries:
        phone = str(row.get("phone", ""))
        if not phone or any(c.isspace() for c in phone): raise ValueError("Source phones must be single symbols")
        matches = sorted(inventory.get(key(row), [])) if key(row)[0] else []
        if len(matches) == 1: mappings[phone] = matches
        elif matches: ambiguous[phone] = matches
        else: missing.append(phone)
    return dict(mappings=mappings, ambiguous=ambiguous, missing=missing,
        warning="Exact supplied IPA/features only; no approximate substitutions or automatic application")

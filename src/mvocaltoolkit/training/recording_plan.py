"""Phoneme-only recording exercises from a frozen dataset coverage report."""
from collections import Counter
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .dictionaries import read_entries
from .phonemizers import PROFILES, SPECIALS, default_profile, profile


class RecordingPlanRequest(BaseModel):
    model_config = {"extra": "forbid"}
    report: str = Field(min_length=1)
    language: str = Field(min_length=1)
    speaker: str = ""
    min_count: int = Field(5, strict=True, ge=1, le=100)
    repetitions: int = Field(3, strict=True, ge=1, le=20)
    limit: int = Field(32, strict=True, ge=1, le=128)
    patterns: list[Literal["cv", "vc", "v", "cvc"]] = Field(default_factory=lambda: ["cv", "vc", "v"], min_length=1)
    include_profile: bool = True
    profile: str | None = None
    targets: list[str] | None = None
    vowels: list[str] | None = None
    consonants: list[str] | None = None


def recording_plan(req: RecordingPlanRequest) -> dict:
    path = Path(req.report).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict): raise ValueError("Dataset report must contain an object")

    def tokens(values, strict=False):
        result = set()
        for token in values:
            if not isinstance(token, str) or not token or any(c.isspace() for c in token): raise ValueError("Invalid phoneme symbol")
            if "/" in token:
                language, token = token.split("/", 1)
                if language != req.language:
                    if strict: raise ValueError("Use phonemes from the selected language")
                    continue
            if not token or "/" in token: raise ValueError("Invalid language-qualified phoneme")
            if token in SPECIALS:
                if strict: raise ValueError("Silence/breath/control symbols are not recording targets")
                continue
            result.add(token)
        return result

    coverage = data.get("coverage")
    if not isinstance(coverage, dict) or req.language not in coverage: raise ValueError("No coverage for the selected language; rebuild the dataset report")
    if req.speaker:
        speakers = data.get("speaker_coverage") or {}
        if not isinstance(speakers, dict) or not isinstance(speakers.get(req.speaker), dict) or req.language not in speakers[req.speaker]:
            raise ValueError("No coverage for this singer/language; counts from other singers are not substituted")
        source_counts = speakers[req.speaker][req.language]
    else: source_counts = coverage[req.language]
    if not isinstance(source_counts, dict): raise ValueError("Invalid phoneme counts in the dataset report")
    counts = Counter()
    for token, count in source_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0: raise ValueError("Phoneme counts must be nonnegative integers")
        for phone in tokens([token]): counts[phone] += count
    profiles = data.get("phonemizer_profiles") or {}
    if not isinstance(profiles, dict): raise ValueError("Invalid alphabet profiles in the dataset report")
    inferred = req.profile is None and req.language not in profiles
    identifier = req.profile if req.profile is not None else profiles.get(req.language, "")
    selected = profile(identifier, req.language) if identifier else {}
    if req.targets is not None:
        expected = tokens(req.targets, True)
        if not expected: raise ValueError("Choose at least one target phoneme")
    else:
        expected = set(counts)
        dictionaries = data.get("dictionaries") or {}
        if not isinstance(dictionaries, dict): raise ValueError("Invalid dictionaries in the dataset report")
        if dictionary := dictionaries.get(req.language):
            dictionary_path = Path(dictionary).expanduser()
            if not dictionary_path.is_absolute(): dictionary_path = path.parent / dictionary_path
            expected.update(tokens(p for _, phones in read_entries(dictionary_path, req.language) for p in phones))
    if inferred:
        candidates = [p for p in PROFILES.values() if p["language"] == req.language]
        if candidates:
            selected = max(candidates, key=lambda p: (len(expected & set(p["vowels"] + p["consonants"])), p["id"] == default_profile(req.language)))
    expand = req.include_profile and not inferred
    if req.targets is None and expand: expected.update(tokens(selected.get("vowels", []) + selected.get("consonants", [])))
    if len(expected) > 2048: raise ValueError("Use at most 2048 target phonemes")
    vowels = tokens(req.vowels, True) if req.vowels is not None else set(selected.get("vowels", []))
    consonants = tokens(req.consonants, True) if req.consonants is not None else set(selected.get("consonants", []))
    if vowels & consonants: raise ValueError("Vowel and consonant lists must not overlap")
    available = expected | set(counts)
    if expand: available.update(selected.get("vowels", []) + selected.get("consonants", []))
    if req.vowels is not None: available.update(vowels)
    if req.consonants is not None: available.update(consonants)
    vowels &= available; consonants &= available
    needed = {p: req.min_count - counts[p] for p in sorted(expected) if counts[p] < req.min_count}
    remaining = needed.copy()
    anchor_vowels = sorted(vowels, key=lambda p: (-counts[p], p))[:3]
    anchor_consonants = sorted(consonants, key=lambda p: (-counts[p], p))[:3]
    candidates, candidate_phones = {}, set()

    def candidate(phones, pattern):
        candidates.setdefault(tuple(phones), pattern)
        candidate_phones.update(phones)

    for phone in needed:
        if phone in vowels:
            if "v" in req.patterns: candidate([phone], "v")
            for consonant in anchor_consonants:
                if "cv" in req.patterns: candidate([consonant, phone], "cv")
                if "vc" in req.patterns: candidate([phone, consonant], "vc")
                if "cvc" in req.patterns: candidate([consonant, phone, consonant], "cvc")
        elif phone in consonants:
            for vowel in anchor_vowels:
                if "cv" in req.patterns: candidate([phone, vowel], "cv")
                if "vc" in req.patterns: candidate([vowel, phone], "vc")
                if "cvc" in req.patterns: candidate([phone, vowel, phone], "cvc")
        if phone not in candidate_phones: candidate([phone], "phone")
    contributions = {phones: Counter(phones) for phones in candidates}
    rows = []
    planned = Counter()
    while candidates and len(rows) < req.limit and any(remaining.values()):
        def score(phones):
            return sum(min(remaining.get(p, 0), n * req.repetitions) * (2 if counts[p] == 0 else 1) for p, n in contributions[phones].items())
        phones = max(sorted(candidates), key=score)
        if score(phones) == 0: break
        rows.append(dict(id=f"{len(rows) + 1:03d}", phonemes=list(phones), pattern=candidates.pop(phones),
            repetitions=req.repetitions, targets=[p for p in dict.fromkeys(phones) if p in needed]))
        for phone, count in contributions[phones].items():
            planned[phone] += count * req.repetitions
            if phone in remaining: remaining[phone] = max(0, remaining[phone] - count * req.repetitions)
    unclassified = sorted(set(needed) - vowels - consonants)
    incomplete = {p: n for p, n in remaining.items() if n}
    lines = [f"Language: {req.language}; singer: {req.speaker or 'all'}; target occurrences: {req.min_count}",
        "Phoneme exercises; repeat each row as indicated. Counts are planned, not measured recordings.", ""]
    lines.extend(row["id"] + "\t" + " ".join(row["phonemes"]) + f"\t x{row['repetitions']}" for row in rows)
    if incomplete: lines.extend(["", "Remaining planned deficits: " + ", ".join(f"{p}: {n}" for p, n in incomplete.items())])
    return dict(report=str(path), language=req.language, speaker=req.speaker, profile=selected.get("id"), profile_inferred=inferred,
        settings=req.model_dump(exclude={"report"}), rows=rows, counts=dict(counts),
        needed=[dict(phone=p, count=counts[p], deficit=n) for p, n in needed.items()],
        planned_occurrences=dict(sorted(planned.items())), remaining=incomplete, unclassified=unclassified,
        text="\n".join(lines) + "\n")

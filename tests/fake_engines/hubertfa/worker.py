"""Fake HubertFA worker: dictionary per language from vocab.json, evenly spread phonemes, AP at the start."""

import json
import wave
from pathlib import Path

import mvt_engine as rt


@rt.method()
def align(model, items, language=None, g2p="auto", non_lexical_phonemes=None, pad_times=1, pad_length=5.0,
          skip_unknown_words=False, dictionary=None, extra_words=None):
    root = Path(model["path"])
    vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    dictionaries = vocab["dictionaries"]
    if language not in dictionaries:
        raise ValueError(f"Language {language!r} is not supported, available: {', '.join(dictionaries)}")
    table = {}
    for line in (root / dictionaries[language]).read_text(encoding="utf-8").splitlines():
        word, phones = line.split("\t")
        table[word] = phones.split()
    table.update(extra_words or {})
    results = []
    for item in items:
        with wave.open(item["audio"]) as w:
            duration = w.getnframes() / w.getframerate()
        words = item.get("words") or []
        unknown = [w for w in words if w not in table]
        if unknown and not skip_unknown_words:
            results.append({"ok": False, "error": "Unknown words: " + " ".join(unknown), "unknown_words": unknown})
            continue
        phones = [p for w in words if w in table for p in table[w]]
        lead = ["AP"] if "AP" in (non_lexical_phonemes or []) else ["SP"]
        seq = lead + phones + ["SP"]
        step = duration / len(seq)
        out = [[i * step, (i + 1) * step, p] for i, p in enumerate(seq)]
        results.append({"ok": True, "duration": duration, "confidence": 0.8, "phones": out,
                        "words": [], "unknown_words": unknown, "language": language})
    return results


if __name__ == "__main__":
    rt.run()

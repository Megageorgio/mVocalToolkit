"""Fake TIFA worker: takes the raw text, one phoneme per letter, evenly spread, SP at the edges."""

import wave

import mvt_engine as rt


@rt.method()
def align(model, items, language=None, extra_languages=None, g2p="auto", skip_unknown_words=False,
          extra_words=None, optional_breaths=False, split_silence=False, **_):
    results = []
    for item in items:
        with wave.open(item["audio"]) as w:
            duration = w.getnframes() / w.getframerate()
        words = item.get("words") or (item.get("text") or "").split()
        if not words:
            results.append({"ok": False, "error": "Empty text"})
            continue
        seq = ["SP"] + [ch for word in words for ch in (extra_words or {}).get(word, list(word))] + ["SP"]
        step = duration / len(seq)
        phones = [[i * step, (i + 1) * step, p] for i, p in enumerate(seq)]
        results.append({"ok": True, "duration": duration, "confidence": 0.5, "phones": phones,
                        "words": [[step, duration - step, " ".join(words)]], "texts": [[step, duration - step, "T"]],
                        "language": language, "diagnosis": {"agreement": 0.9, "confidence": 0.5,
                                                            "extra_languages": extra_languages,
                                                            "optional_breaths": optional_breaths,
                                                            "split_silence": split_silence}})
    return results


@rt.method()
def phonemize(model, texts, language=None, extra_languages=None):
    return [{"ok": True, "words": [{"text": w, "candidates": [{"scripts": [w], "phonemes": list(w)}]
                                    + ([{"scripts": [w], "phonemes": ["x"]}] if w == "read" else [])}
                                   for w in t.split()]} for t in texts]


if __name__ == "__main__":
    rt.run()

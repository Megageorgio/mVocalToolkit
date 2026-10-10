"""Fake SOFA worker for tests: spreads phonemes evenly over the audio duration."""

import wave
from pathlib import Path

import mvt_engine as rt


def _dictionary(model):
    path = Path(model["path"]) / model["layout"]["dictionary"]
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        word, phones = line.split("\t")
        entries[word] = phones.split()
    return entries


@rt.method()
def align(model, items, mode="force", g2p="auto", ap_detector="none", skip_unknown_words=False,
          dictionary=None, extra_words=None, split_silence=False, split_max_length=25.0, split_min_silence=0.3):
    dictionary = {**_dictionary(model), **(extra_words or {})}
    results = []
    for index, item in enumerate(items):
        rt.progress(index / len(items), "aligning")
        with wave.open(item["audio"]) as w:
            duration = w.getnframes() / w.getframerate()
        unknown = [w for w in (item.get("words") or []) if w not in dictionary]
        if unknown and not skip_unknown_words:
            results.append({"ok": False, "error": "Unknown words: " + " ".join(unknown), "unknown_words": unknown})
            continue
        if item.get("phonemes"):
            words = [(p, [p]) for p in item["phonemes"]]
        else:
            words = [(w, dictionary[w]) for w in item.get("words") or [] if w in dictionary]
        phones = [p for _, ps in words for p in ps]
        step = duration / (len(phones) + 2)
        out_ph = [[0.0, step, "SP"]]
        out_w = []
        t = step
        for word, ps in words:
            start = t
            for p in ps:
                out_ph.append([t, t + step, p, 0.9])
                t += step
            out_w.append([start, t, word])
        out_ph.append([t, duration, "SP"])
        results.append({"ok": True, "duration": duration, "confidence": 0.9, "phones": out_ph, "words": out_w,
                        "unknown_words": unknown})
    return results


@rt.method()
def g2p(model, words):
    return {w: ["x"] for w in words}


if __name__ == "__main__":
    rt.run()

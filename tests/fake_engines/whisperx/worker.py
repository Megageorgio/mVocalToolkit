"""Fake WhisperX worker for tests: "recognizes" the text stored in <audio>.expected next to the file."""

from pathlib import Path

import mvt_engine as rt


@rt.method()
def transcribe(items, model="x", language=None, **_options):
    results = []
    for item in items:
        expected = Path(item["audio"]).with_suffix(".expected")
        text = expected.read_text(encoding="utf-8") if expected.exists() else "Hello, World!"
        results.append({"ok": True, "text": text, "language": language or "en",
                        "segments": [{"start": 0.0, "end": 1.0, "text": text}]})
    return results


if __name__ == "__main__":
    rt.run()

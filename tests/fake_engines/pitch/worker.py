"""Fake pitch worker: 220 Hz, the first frame unvoiced."""

import wave

import mvt_engine as rt


@rt.method()
def pitch(items, method="rmvpe", model_path=None, hop=0.01, f0_min=50.0, f0_max=1100.0, threshold=None):
    results = []
    for item in items:
        with wave.open(item["audio"]) as w:
            duration = w.getnframes() / w.getframerate()
        n = int(duration / hop) + 1
        results.append({"ok": True, "name": item["name"], "hop": hop, "f0": [0.0] + [220.0] * (n - 1),
                        "method": method})
    return results


if __name__ == "__main__":
    rt.run()

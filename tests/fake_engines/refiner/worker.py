"""Fake boundary refiner for tests: moves every boundary between touching phonemes 10 ms later."""

import mvt_engine as rt


@rt.method()
def refine(model, items, mode="auto", phone_map=None, min_confidence=None, max_shift_ms=None):
    results = []
    for item in items:
        phones = [list(p) for p in item["phones"]]
        for i in range(1, len(phones)):
            if abs(phones[i][0] - phones[i - 1][1]) < 1e-9:
                phones[i][0] += 0.01
                phones[i - 1][1] = phones[i][0]
        results.append({"ok": True, "phones": phones, "info": {"mode": mode, "moved": len(phones) - 1}})
    return results


if __name__ == "__main__":
    rt.run()

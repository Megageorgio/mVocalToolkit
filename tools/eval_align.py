"""Aligns HubertFA's evaluation sets through the mVocalToolkit API and compares with the ground truth.

usage: python eval_align.py <dataset dir with transcriptions.csv and wavs/> <model id> <language> [max files] [A=B,...]
Gives phonemes (without AP/SP) to /align and measures boundary errors against ph_dur.
"""

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8765"
SILENT = {"AP", "SP", "EP", "pau", "sil"}


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    root, model, language = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    # phoneme renames between the dataset and the model's dictionary, e.g. "E=ie,En=ian"
    mapping = dict(pair.split("=") for pair in sys.argv[5].split(",")) if len(sys.argv) > 5 else {}
    rows = list(csv.DictReader(open(root / "transcriptions.csv", encoding="utf-8")))[:limit]
    items, truth = [], {}
    for row in rows:
        seq, dur = [mapping.get(p, p) for p in row["ph_seq"].split()], [float(d) for d in row["ph_dur"].split()]
        t, gt = 0.0, []
        for ph, d in zip(seq, dur):
            gt.append((t, t + d, ph))
            t += d
        truth[row["name"]] = gt
        items.append({"path": str(root / "wavs" / f"{row['name']}.wav"),
                      "phonemes": [p for p in seq if p not in SILENT]})
    started = time.time()
    job = call("POST", "/align", {"input": {"items": items}, "model": model, "language": language,
                                   "transcribe": None, "output": {"formats": ["json"], "dir": "/tmp/mvt-eval-out"}})
    while True:
        info = call("GET", f"/jobs/{job['id']}?wait=20")
        if info["status"] in ("done", "failed", "cancelled"):
            break
        print(f"  {info['progress'] * 100:.0f}% {info['stage']} {info['message']}", flush=True)
    if info["status"] != "done":
        print(json.dumps(info, indent=1)[:3000])
        return
    errors, worst = [], []
    for item in info["result"]["items"]:
        if not item["ok"]:
            print("FAILED", item["name"], item.get("error"))
            continue
        pred = [(p["start"], p["end"], p["text"]) for p in item["label"]["tiers"]["phones"] if p["text"] not in SILENT]
        gt = [p for p in truth[item["name"]] if p[2] not in SILENT]
        if [p[2] for p in pred] != [p[2] for p in gt]:
            print("phoneme mismatch", item["name"], [p[2] for p in pred][:10], [p[2] for p in gt][:10])
            continue
        for (ps, pe, _), (gs, ge, _) in zip(pred, gt):
            errors += [abs(ps - gs), abs(pe - ge)]
    errors.sort()
    if errors:
        n = len(errors)
        print(f"{model}: {len(info['result']['items'])} files, {n} boundaries, {time.time() - started:.1f}s")
        print(f"  mean {sum(errors) / n * 1000:.1f} ms, median {errors[n // 2] * 1000:.1f} ms, "
              f"<20ms {sum(e < 0.02 for e in errors) / n:.0%}, <50ms {sum(e < 0.05 for e in errors) / n:.0%}")


if __name__ == "__main__":
    main()

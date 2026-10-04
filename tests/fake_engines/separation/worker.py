"""Fake separation worker: copies the input as the vocals stem."""

import shutil
from pathlib import Path

import mvt_engine as rt


@rt.method()
def separate(items, model_file, model_dir, stems=None, output_format="wav", sample_rate=44100, options=None):
    results = []
    for item in items:
        out = Path(item["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for stem in stems or ["vocals", "instrumental"]:
            target = out / f"{item['name']}_{stem}.wav"
            shutil.copy(item["audio"], target)
            files[stem] = str(target)
        results.append({"ok": True, "name": item["name"], "files": files, "model_file": model_file})
    return results


if __name__ == "__main__":
    rt.run()

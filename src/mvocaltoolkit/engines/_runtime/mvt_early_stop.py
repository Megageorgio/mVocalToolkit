"""Validation-check patience persisted independently of checkpoint callback order."""
import json
import math
import os


class EarlyStop:
    def __init__(self, folder, options):
        self.path = folder / "early-stop.json"
        self.options = options
        self.state = dict(best=None, bad=0, last_step=-1)
        if self.path.is_file():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("options") == options: self.state.update(saved["state"])

    def update(self, step, score):
        if not self.options.get("patience") or step < self.options.get("min_steps", 0) or step <= self.state["last_step"]:
            return False
        if score is None or not math.isfinite(score): return False
        better = self.state["best"] is None or score < self.state["best"] - self.options.get("min_delta", 0)
        self.state.update(best=score if better else self.state["best"], bad=0 if better else self.state["bad"] + 1, last_step=step)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(options=self.options, state=self.state)), encoding="utf-8"); os.replace(tmp, self.path)
        return self.state["bad"] >= self.options["patience"]

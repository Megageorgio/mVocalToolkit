"""Post-processing rules applied to existing label files (e.g. labels made earlier or by other tools)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from .. import formats
from ..api_models import FixLabelsRequest
from ..text.rules import apply_rule_sets

SUFFIXES = {"htk": [".lab"], "textgrid": [".textgrid"], "json": [".json"]}


def find_label_files(req: FixLabelsRequest) -> list[Path]:
    wanted = {s for f in req.formats for s in SUFFIXES.get(formats.normalize_format(f), [])}
    files = [Path(p).expanduser().absolute() for p in req.paths]
    if req.folder:
        root = Path(req.folder).expanduser().absolute()
        if not root.is_dir():
            raise FileNotFoundError(f"Folder not found: {root}")
        candidates = root.rglob("*") if req.recursive else root.glob("*")
        files += [p for p in candidates if p.is_file() and p.suffix.lower() in wanted
                  and not any(part.startswith(("_backup", ".")) for part in p.relative_to(root).parts)]
    return sorted(set(files))


def fix_labels(req: FixLabelsRequest) -> dict[str, Any]:
    if not req.rule_sets and not req.rules:
        raise ValueError("No fixes selected")
    files = find_label_files(req)
    backup_root = None
    if req.backup and not req.dry_run and files:
        base = Path(req.folder).expanduser().absolute() if req.folder else files[0].parent
        backup_root = base / "_backup" / time.strftime("%Y%m%d-%H%M%S")
    results = []
    for path in files:
        try:
            fmt = formats.detect_format(path)
            label = formats.read(path, fmt)
            before = [iv.text for iv in label.tiers.get("phones", [])]
            fixed = apply_rule_sets(label, req.rule_sets, req.rules)
            after = [iv.text for iv in fixed.tiers.get("phones", [])]
            changed = before != after or any(
                (a.start, a.end) != (b.start, b.end)
                for a, b in zip(label.tiers.get("phones", []), fixed.tiers.get("phones", []))
            )
            if changed and not req.dry_run:
                if backup_root is not None:
                    root = Path(req.folder).expanduser().absolute() if req.folder else path.parent
                    target = backup_root / (path.relative_to(root) if path.is_relative_to(root) else path.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                formats.write(fixed, path, fmt)
            results.append({"path": str(path), "ok": True, "changed": changed,
                            "phones_before": len(before), "phones_after": len(after)})
        except Exception as e:  # noqa: BLE001
            results.append({"path": str(path), "ok": False, "error": f"{type(e).__name__}: {e}"})
    return {
        "files": results,
        "changed": sum(1 for r in results if r.get("changed")),
        "backup": str(backup_root) if backup_root and any(r.get("changed") for r in results) else None,
    }

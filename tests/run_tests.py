"""Minimal test runner for environments without pytest (supports the tmp_path fixture only).

Usage: python tests/run_tests.py [name-filter]
With pytest installed, just run `pytest`.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    failed = passed = 0
    for module_path in sorted(HERE.glob("test_*.py")):
        module = importlib.import_module(module_path.stem)
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_") or pattern not in f"{module_path.stem}.{name}":
                continue
            kwargs = {}
            with tempfile.TemporaryDirectory() as tmp:
                if "tmp_path" in inspect.signature(func).parameters:
                    kwargs["tmp_path"] = Path(tmp)
                try:
                    func(**kwargs)
                    passed += 1
                    print(f"PASS {module_path.stem}.{name}")
                except Exception:  # noqa: BLE001
                    failed += 1
                    print(f"FAIL {module_path.stem}.{name}")
                    traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

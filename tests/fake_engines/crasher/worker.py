"""Fake engine: prints a traceback to stderr and exits when called (like a crash while loading a model)."""

import os
import sys

import mvt_engine as rt


@rt.method()
def boom() -> None:
    print("Traceback (most recent call last):\n  File \"model.py\", line 1\nRuntimeError: no encoder weights", file=sys.stderr)
    sys.stderr.flush()
    os._exit(3)


if __name__ == "__main__":
    rt.run()

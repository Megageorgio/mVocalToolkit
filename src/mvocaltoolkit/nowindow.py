"""No console windows on Windows.

Libraries started from the toolkit (ffmpeg through audio loaders, nvidia-smi, git, …) call subprocess without
CREATE_NO_WINDOW; when the toolkit itself has no visible console, each such call flashes a console window.
This adds the flag to every subprocess that doesn't ask for a console of its own.
"""

from __future__ import annotations

import os

CREATE_NO_WINDOW = 0x08000000
_OWN_CONSOLE = 0x00000010 | 0x00000008  # CREATE_NEW_CONSOLE | DETACHED_PROCESS


def hide_child_consoles() -> None:
    if os.name != "nt":
        return
    import subprocess  # noqa: PLC0415

    if getattr(subprocess.Popen, "_mvt_no_window", False):
        return
    original = subprocess.Popen.__init__

    def init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        flags = kwargs.get("creationflags", 0) or 0
        if not flags & _OWN_CONSOLE:
            kwargs["creationflags"] = flags | CREATE_NO_WINDOW
        original(self, *args, **kwargs)

    subprocess.Popen.__init__ = init  # type: ignore[method-assign]
    subprocess.Popen._mvt_no_window = True  # type: ignore[attr-defined]

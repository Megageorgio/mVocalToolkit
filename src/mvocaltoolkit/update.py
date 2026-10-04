"""Update check and self-update.

The toolkit is installed as a uv tool (`uv tool install mvocaltoolkit`), so updating is `uv tool upgrade`.
Engines are updated separately: when an engine's engine.toml changes, its environment is rebuilt on next use.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any

import httpx

from . import __version__

REPO = "Megageorgio/mVocalToolkit"


def _parse(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version)[:3])


async def check_update() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            response = await client.get(f"https://api.github.com/repos/{REPO}/releases/latest")
            response.raise_for_status()
            data = response.json()
    except Exception as e:  # noqa: BLE001
        return {"current": __version__, "error": str(e)}
    latest = str(data.get("tag_name", "")).lstrip("v")
    return {
        "current": __version__,
        "latest": latest,
        "update_available": bool(latest) and _parse(latest) > _parse(__version__),
        "url": data.get("html_url"),
        "notes": data.get("body", ""),
    }


def self_update() -> int:
    """Upgrades the toolkit installed with `uv tool install`."""
    from .engines.env import find_uv  # noqa: PLC0415

    return subprocess.call([find_uv(), "tool", "upgrade", "mvocaltoolkit"])

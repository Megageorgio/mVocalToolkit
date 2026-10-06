"""Updating itself when `mvt serve` starts.

The toolkit is installed as a uv tool from a source: a GitHub branch (archive URL or git URL) or a local folder.
When the source has a commit newer than the installation, `mvt serve` hands over to a small detached updater that
waits for it to exit, runs `uv tool install --force --reinstall <source>` (files of a running environment can't be
replaced on Windows) and starts `mvt serve` again with the same arguments. The old process exits with
EXIT_UPDATING, so a program that started it knows to wait for the server to come back instead of giving up.
Checks happen at most every few hours; with no network, nothing changes.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

EXIT_UPDATING = 75
CHECK_EVERY = 3 * 3600
NO_WINDOW = 0x08000000


def receipt() -> dict[str, Any] | None:
    """uv's record of how this tool was installed (only present for `uv tool install`)."""
    path = Path(sys.prefix) / "uv-receipt.toml"
    if not path.is_file():
        return None
    try:
        import tomllib  # noqa: PLC0415

        return {"path": path, **tomllib.loads(path.read_text("utf-8"))}
    except Exception:  # noqa: BLE001
        return None


def install_source(rec: dict[str, Any]) -> str | None:
    """What `uv tool install` was given: a URL, git URL or folder."""
    for req in rec.get("tool", {}).get("requirements", []):
        if str(req.get("name", "")).lower().replace("_", "-") != "mvocaltoolkit":
            continue
        if req.get("directory"):
            return str(req["directory"])
        if req.get("path"):
            return str(req["path"])
        if req.get("url"):
            return str(req["url"])
        if req.get("git"):
            git = str(req["git"])
            ref = req.get("branch") or req.get("rev") or req.get("tag")
            return f"git+{git}" + (f"@{ref}" if ref else "")
    return None


_GITHUB = re.compile(r"github\.com/([^/]+)/([^/@#?]+?)(?:\.git)?(?:/archive/refs/heads/(.+?)\.zip|@([^#?]+)|[/?#].*|$)")


def source_time(source: str, github_token: str = "") -> float | None:
    """Time of the newest commit of the source (seconds since epoch), or None when unknown."""
    m = _GITHUB.search(source)
    if m:
        owner, repo = m.group(1), m.group(2)
        branch = m.group(3) or m.group(4) or "main"
        try:
            import httpx  # noqa: PLC0415

            headers = {"Accept": "application/vnd.github+json", "User-Agent": "mVocalToolkit"}
            if github_token:
                headers["Authorization"] = f"Bearer {github_token}"
            r = httpx.get(f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}", headers=headers, timeout=8)
            if r.status_code != 200:
                return None
            date = r.json()["commit"]["committer"]["date"]
            return datetime.fromisoformat(date.replace("Z", "+00:00")).timestamp()
        except Exception:  # noqa: BLE001
            return None
    folder = Path(source)
    git = folder / ".git"
    if git.is_dir():
        # the branch ref changes on every commit and pull
        try:
            head = (git / "HEAD").read_text("utf-8").strip()
            if head.startswith("ref:"):
                ref = git / head[4:].strip()
                if ref.is_file():
                    return ref.stat().st_mtime
                packed = git / "packed-refs"
                return packed.stat().st_mtime if packed.is_file() else None
            return (git / "HEAD").stat().st_mtime
        except OSError:
            return None
    if folder.is_dir():
        # a plain folder: its newest source file
        try:
            return max(p.stat().st_mtime for p in folder.rglob("*.py"))
        except ValueError:
            return None
    return None


def _state_file(home_root: Path) -> Path:
    return home_root / "update.json"


def _load_state(home_root: Path) -> dict[str, Any]:
    try:
        return json.loads(_state_file(home_root).read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_state(home_root: Path, state: dict[str, Any]) -> None:
    try:
        home_root.mkdir(parents=True, exist_ok=True)
        _state_file(home_root).write_text(json.dumps(state, indent=2), "utf-8")
    except OSError:
        pass


def pending_update(home_root: Path, github_token: str = "", force: bool = False) -> str | None:
    """The source to update from when it is newer than this installation, else None."""
    if os.environ.get("MVT_NO_UPDATE"):
        return None
    rec = receipt()
    if rec is None:
        return None  # not a uv tool (a development checkout, or used as a library): nothing to do
    source = install_source(rec)
    if not source:
        return None  # installed from PyPI by name: `uv tool upgrade` is the way
    state = _load_state(home_root)
    now = time.time()
    if not force and now - float(state.get("checked", 0)) < CHECK_EVERY:
        return None
    state["checked"] = now
    _save_state(home_root, state)
    newest = source_time(source, github_token)
    installed = rec["path"].stat().st_mtime
    if newest is None or newest <= installed + 60:
        return None
    return source


def _find_uv() -> str | None:
    from .engines.env import find_uv  # noqa: PLC0415

    try:
        return find_uv()
    except Exception:  # noqa: BLE001
        return None


def hand_over(source: str, home_root: Path, argv: list[str]) -> bool:
    """Starts the detached updater (update, then `mvt` with [argv] again). The caller then exits with EXIT_UPDATING."""
    uv = _find_uv()
    if uv is None:
        return False
    log = home_root / "logs" / "update.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    mvt = Path(sys.argv[0]).resolve() if Path(sys.argv[0]).exists() else Path("mvt")
    install = [uv, "tool", "install", "--force", "--reinstall", "--python", "3.12", source]
    again = [str(mvt), *argv]
    pid = os.getpid()
    if os.name == "nt":
        # wait until this process is gone (its files are in use until then), update, start again
        script = (
            f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; if ($p) {{ $p.WaitForExit(30000) }}; "
            f"& {_ps(install)} *>> {_ps([str(log)])}; "
            f"Start-Process -WindowStyle Hidden -FilePath {_ps([again[0]])} -ArgumentList {_ps_args(again[1:])}"
        )
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-Command", script]
        # a hidden console of its own (not DETACHED_PROCESS: then uv and mvt would each open a visible one)
        flags = 0x00000200 | NO_WINDOW  # CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        subprocess.Popen(cmd, creationflags=flags, close_fds=True)
    else:
        script = (
            f"while kill -0 {pid} 2>/dev/null; do sleep 0.3; done; "
            f"{shlex.join(install)} >> {shlex.quote(str(log))} 2>&1; "
            f"exec {shlex.join(again)} >> {shlex.quote(str(log))} 2>&1"
        )
        subprocess.Popen(["sh", "-c", script], start_new_session=True, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def _ps(parts: list[str]) -> str:
    return " ".join("'" + p.replace("'", "''") + "'" for p in parts)


def _ps_args(parts: list[str]) -> str:
    return "@(" + ", ".join("'" + p.replace("'", "''") + "'" for p in parts) + ")" if parts else "@()"

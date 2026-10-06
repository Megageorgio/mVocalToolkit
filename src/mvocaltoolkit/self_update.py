"""Updating itself when `mvt serve` starts.

The toolkit is installed as a uv tool from a source: a GitHub branch (archive URL or git URL) or a local folder.
When the source has a commit newer than the installation, `mvt serve` hands over to a small detached updater that
waits for it to exit, runs `uv tool install --force --reinstall <source>` (files of a running environment can't be
replaced on Windows) and starts `mvt serve` again with the same arguments. The old process exits with
EXIT_UPDATING, so a program that started it knows to wait for the server to come back instead of giving up.
Every start checks (one short request to GitHub); with no network the toolkit starts as it is.
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
# the last line of logs/update.log once uv is done (a program waiting for the toolkit may start it itself then)
FINISHED = "mVocalToolkit update finished"
# seconds between checks; 0 = at every start
CHECK_EVERY = 0
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


def source_head(source: str, github_token: str = "") -> tuple[str | None, float | None]:
    """The newest commit of the source: (id, time in seconds since epoch); either is None when unknown.
    GitHub branches have an id; local folders only a time."""
    m = _GITHUB.search(source)
    if m:
        owner, repo = m.group(1), m.group(2)
        branch = m.group(3) or m.group(4) or "main"
        try:
            import httpx  # noqa: PLC0415

            headers = {"Accept": "application/vnd.github+json", "User-Agent": "mVocalToolkit"}
            if github_token:
                headers["Authorization"] = f"Bearer {github_token}"
            r = httpx.get(f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}", headers=headers, timeout=5)
            if r.status_code != 200:
                return None, None
            data = r.json()
            date = data["commit"]["committer"]["date"]
            return str(data["sha"]), datetime.fromisoformat(date.replace("Z", "+00:00")).timestamp()
        except Exception:  # noqa: BLE001
            return None, None
    return None, source_time(source)


def source_time(source: str, github_token: str = "") -> float | None:
    """Time of the newest change of the source (seconds since epoch), or None when unknown."""
    if _GITHUB.search(source):
        return source_head(source, github_token)[1]
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


def is_newer(state: dict[str, Any], commit: str | None, commit_time: float | None, installed: float, now: float) -> bool:
    """Whether the source has something the installation doesn't, updating [state].

    A commit's own time says when it was written, not when it reached the branch (it can be pushed hours later),
    so for branches the time the commit was first seen there counts: an installation made after that has it.
    """
    if commit:
        seen = state.get("seen") or {}
        if seen.get("commit") != commit:
            seen = {"commit": commit, "at": now}
            state["seen"] = seen
        return installed < float(seen["at"]) - 5
    return commit_time is not None and commit_time > installed + 60


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
    commit, newest = source_head(source, github_token)
    installed = rec["path"].stat().st_mtime
    newer = is_newer(state, commit, newest, installed, now)
    _save_state(home_root, state)
    return source if newer else None


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
    log = update_log(home_root)
    log.parent.mkdir(parents=True, exist_ok=True)
    # a fresh log for this update (programs that started us can follow it)
    try:
        log.write_text("", encoding="utf-8")
    except OSError:
        pass
    again = [*_mvt_command(), *argv]
    # only the toolkit itself is rebuilt; its dependencies come from uv's cache unless they changed (much faster
    # than --reinstall, which reinstalled every package)
    install = [uv, "tool", "install", "--force", "--reinstall-package", "mvocaltoolkit", "--python", "3.12", source]
    pid = os.getpid()
    if os.name == "nt":
        # cmd's redirection keeps uv's UTF-8 output as it is (PowerShell's would write UTF-16 and wrap errors);
        # /s with outer quotes: cmd keeps the quotes inside the line as they are
        cmd_line = ('"' + _cmd_line(install) + " >> " + _cmd_quote(str(log)) + " 2>&1 & echo " + FINISHED
                    + " >> " + _cmd_quote(str(log)) + '"')
        # wait until this process is gone (its files are in use until then), update, start again
        script = (
            f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; if ($p) {{ $p.WaitForExit(30000) }}; "
            f"& cmd.exe /d /s /c {_ps([cmd_line])}; "
            f"Start-Process -WindowStyle Hidden -FilePath {_ps([again[0]])} -ArgumentList {_ps_args(again[1:])}"
        )
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-Command", script]
        # a hidden console of its own (not DETACHED_PROCESS: then uv and mvt would each open a visible one)
        flags = 0x00000200 | NO_WINDOW  # CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        subprocess.Popen(cmd, creationflags=flags, close_fds=True)
    else:
        script = (
            f"while kill -0 {pid} 2>/dev/null; do sleep 0.3; done; "
            f"{shlex.join(install)} >> {shlex.quote(str(log))} 2>&1; echo {FINISHED} >> {shlex.quote(str(log))}; "
            f"exec {shlex.join(again)} >> {shlex.quote(str(log))} 2>&1"
        )
        subprocess.Popen(["sh", "-c", script], start_new_session=True, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def _mvt_command() -> list[str]:
    """How to start this toolkit again. sys.argv[0] may lack ".exe", and the tools folder may not be on PATH."""
    import shutil  # noqa: PLC0415

    first = Path(sys.argv[0])
    candidates = [first, first.with_name(first.name + ".exe"), Path(sys.prefix) / "Scripts" / "mvt.exe",
                  Path(sys.prefix) / "bin" / "mvt"]
    for c in candidates:
        try:
            if c.is_file() and c.suffix.lower() in ("", ".exe"):
                return [str(c.resolve())]
        except OSError:
            continue
    found = shutil.which("mvt")
    if found:
        return [found]
    # the environment's Python stays where it is after a reinstall
    return [sys.executable, "-m", "mvocaltoolkit"]


def _cmd_quote(arg: str) -> str:
    return '"' + arg.replace('"', '""') + '"' if any(c in arg for c in ' &()^|<>"') else arg


def _cmd_line(parts: list[str]) -> str:
    return " ".join(_cmd_quote(p) for p in parts)


def update_log(home_root: Path) -> Path:
    return home_root / "logs" / "update.log"


def _ps(parts: list[str]) -> str:
    return " ".join("'" + p.replace("'", "''") + "'" for p in parts)


def _ps_args(parts: list[str]) -> str:
    return "@(" + ", ".join("'" + p.replace("'", "''") + "'" for p in parts) + ")" if parts else "@()"

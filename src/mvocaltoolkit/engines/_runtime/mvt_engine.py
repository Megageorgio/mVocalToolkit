"""Runtime for engine worker processes (stdlib only, runs inside each engine's own environment).

Protocol: JSON lines.
    request  (stdin):  {"id": 1, "method": "align", "params": {...}}
    response (stdout): {"id": 1, "result": ...}  or  {"id": 1, "error": {"type": "...", "message": "...", "trace": "..."}}
    event    (stdout): {"id": 1, "event": "progress", "data": {...}}
Anything printed by libraries goes to stderr (stdout is reserved for the protocol).

A worker module defines handlers with the @method decorator and calls run().
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from typing import Any, Callable

_methods: dict[str, Callable[..., Any]] = {}
_protocol_out = sys.stdout
_lock = threading.Lock()
_current_id: int | None = None


def method(name: str | None = None):
    def decorator(func: Callable[..., Any]):
        _methods[name or func.__name__] = func
        return func

    return decorator


def _send(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False, default=_default)
    with _lock:
        _protocol_out.write(line + "\n")
        _protocol_out.flush()


def _default(value: Any):
    # numpy / pathlib values
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    return str(value)


def progress(value: float | None = None, message: str | None = None, **data: Any) -> None:
    """Reports progress of the current request (value in 0..1)."""
    payload: dict[str, Any] = dict(data)
    if value is not None:
        payload["progress"] = value
    if message is not None:
        payload["message"] = message
    _send({"id": _current_id, "event": "progress", "data": payload})


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def device(requested: str | None = None) -> str:
    """Resolves "auto" to cuda/mps/cpu."""
    requested = (requested or os.environ.get("MVT_DEVICE") or "auto").lower()
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA is not available, falling back to CPU")
        return "cpu"
    return requested


def is_oom(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return "out of memory" in text or "outofmemory" in text


def free_memory() -> None:
    try:
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


@method("ping")
def _ping() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "methods": sorted(_methods)}
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def run() -> None:
    global _protocol_out, _current_id
    # Reserve the real stdout for the protocol, redirect everything else to stderr
    _protocol_out = sys.stdout
    sys.stdout = sys.stderr
    _send({"event": "ready", "data": {"methods": sorted(_methods)}})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            _send({"id": None, "error": {"type": "ProtocolError", "message": str(e)}})
            continue
        if request.get("method") == "shutdown":
            _send({"id": request.get("id"), "result": True})
            break
        _current_id = request.get("id")
        func = _methods.get(request.get("method", ""))
        if func is None:
            _send({"id": _current_id, "error": {"type": "UnknownMethod", "message": request.get("method")}})
            continue
        try:
            result = func(**(request.get("params") or {}))
            _send({"id": _current_id, "result": result})
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                break
            _send(
                {
                    "id": _current_id,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "oom": is_oom(e),
                        "trace": traceback.format_exc(limit=10),
                    },
                }
            )
            if is_oom(e):
                free_memory()
        finally:
            _current_id = None

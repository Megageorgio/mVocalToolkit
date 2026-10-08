"""Programs that use this toolkit. A toolkit started with --exit-when-unused stops by itself once no program has
been attached for a while and no job is running, so it never outlives the programs that need it (and never keeps
their folders busy). Several programs can share one toolkit: each attaches, sends a sign of life now and then and
detaches when it closes."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

# a program that hasn't been heard from for this long is considered gone (it may have crashed)
STALE_SECONDS = 90.0


@dataclass
class Client:
    id: str
    name: str
    pid: int | None
    attached: float = field(default_factory=time.monotonic)
    seen: float = field(default_factory=time.monotonic)


class Clients:
    def __init__(self) -> None:
        self._clients: dict[str, Client] = {}
        self.started = time.monotonic()
        # last moment something used the toolkit (a client or a job)
        self.last_used = time.monotonic()

    def attach(self, name: str, pid: int | None) -> Client:
        c = Client(secrets.token_hex(8), name or "program", pid)
        self._clients[c.id] = c
        self.last_used = time.monotonic()
        return c

    def ping(self, client_id: str) -> bool:
        c = self._clients.get(client_id)
        if c is None:
            return False
        c.seen = time.monotonic()
        self.last_used = c.seen
        return True

    def detach(self, client_id: str) -> bool:
        gone = self._clients.pop(client_id, None) is not None
        self.last_used = time.monotonic()
        return gone

    def active(self) -> list[Client]:
        now = time.monotonic()
        for cid in [cid for cid, c in self._clients.items() if now - c.seen > STALE_SECONDS]:
            self._clients.pop(cid, None)
        return list(self._clients.values())

    def describe(self) -> list[dict]:
        now = time.monotonic()
        return [{"id": c.id, "name": c.name, "pid": c.pid, "seconds": round(now - c.attached)} for c in self.active()]

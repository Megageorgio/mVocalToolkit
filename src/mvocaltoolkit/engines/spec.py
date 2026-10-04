"""Engine descriptions (engine.toml next to each engine's worker)."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

ENGINES_DIR = Path(__file__).parent


class SourceSpec(BaseModel):
    """Upstream source code downloaded as a GitHub archive at a pinned commit."""

    repo: str  # owner/name
    commit: str
    # files of the repo the worker needs on sys.path (relative); default: repo root
    path: str = "."


class EngineSpec(BaseModel):
    name: str
    title: str
    description: str = ""
    # Python version for the engine's environment, or "system" to run with the server's interpreter (dev/tests)
    python: str = "3.11"
    requirements: list[str] = Field(default_factory=list)
    # install torch with the toolkit's torch backend (cpu / cuda auto-detection)
    torch: bool = False
    worker: str = "worker.py"
    source: SourceSpec | None = None
    # what the engine can do: transcribe, align, segment, midi, tempo, text...
    capabilities: list[str] = Field(default_factory=list)
    # approximate download size, shown to users before installing
    size_hint: str = ""
    homepage: str = ""
    license: str = ""
    # directory with engine.toml and worker (filled by the loader)
    dir: Path = Path(".")

    @property
    def is_system(self) -> bool:
        return self.python == "system"

    def fingerprint(self) -> str:
        """Changes when anything that affects the environment changes (triggers reinstall)."""
        data = "|".join(
            [
                self.python,
                ",".join(self.requirements),
                str(self.torch),
                self.source.repo + "@" + self.source.commit if self.source else "",
            ]
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]


def load_engine_specs(*dirs: Path) -> dict[str, EngineSpec]:
    specs: dict[str, EngineSpec] = {}
    for base in dirs or (ENGINES_DIR,):
        if not base.exists():
            continue
        for toml_path in sorted(base.glob("*/engine.toml")):
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            spec = EngineSpec.model_validate({**data, "dir": toml_path.parent})
            specs[spec.name] = spec
    return specs

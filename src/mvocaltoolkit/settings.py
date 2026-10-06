"""Toolkit home directory and settings.

Everything the toolkit downloads or creates lives in one folder (the "home"):

    <home>/
        config.yaml         settings
        models/<id>/        installed models (one folder per model, with model.json)
        envs/<engine>/      isolated Python environments of the engines
        sources/<engine>/   upstream source code used by the engines
        cache/              downloads, intermediate results
        jobs/               job history
        uploads/            files uploaded through the API
        outputs/            results of jobs that were not asked to be written elsewhere
        catalogs/           user catalogs (*.json) and cached remote catalogs

The home folder is chosen with --home, the MVT_HOME environment variable, or defaults to ~/mVocalToolkit.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PORT = 8765


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    # Required for requests from other machines. Requests from localhost don't need it unless require_token_local.
    token: str = ""
    require_token_local: bool = False
    # "auto", "cpu", or a CUDA backend understood by `uv pip --torch-backend` (e.g. "cu128")
    torch_backend: str = "auto"
    # "auto", "cpu", "cuda", "cuda:1"...
    device: str = "auto"
    # Engines are stopped after this many seconds without requests (frees GPU memory). 0 = never.
    engine_idle_timeout: int = 600
    # Install engine environments automatically when a job needs them.
    auto_install_engines: bool = True
    # Download models automatically when a job needs them.
    auto_download_models: bool = True
    # Extra catalogs (URLs or file paths) merged with the built-in catalog.
    catalogs: list[str] = field(default_factory=list)
    # Optional token for the GitHub API (raises rate limits when resolving release assets).
    github_token: str = ""
    # Optional Hugging Face token (some models need it).
    hf_token: str = ""
    # `mvt serve` checks for a newer toolkit when it starts and updates itself (uv tool installs only).
    auto_update: bool = True
    # Max number of jobs running at the same time. Jobs that use the same engine are queued anyway.
    max_parallel_jobs: int = 2
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        values = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        settings = cls(**values)
        settings.extra.update(extra)
        return settings

    def to_dict(self) -> dict[str, Any]:
        data = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "extra"}  # type: ignore[attr-defined]
        data.update(self.extra)
        return data


class Home:
    """Paths inside the toolkit home folder."""

    def __init__(self, root: str | os.PathLike[str] | None = None):
        if root is None:
            root = os.environ.get("MVT_HOME") or Path.home() / "mVocalToolkit"
        self.root = Path(root).expanduser().resolve()

    @property
    def config_file(self) -> Path:
        return self.root / "config.yaml"

    def dir(self, name: str) -> Path:
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def models(self) -> Path:
        return self.dir("models")

    @property
    def envs(self) -> Path:
        return self.dir("envs")

    @property
    def sources(self) -> Path:
        return self.dir("sources")

    @property
    def cache(self) -> Path:
        return self.dir("cache")

    @property
    def downloads(self) -> Path:
        return self.dir("cache/downloads")

    @property
    def jobs(self) -> Path:
        return self.dir("jobs")

    @property
    def uploads(self) -> Path:
        return self.dir("uploads")

    @property
    def outputs(self) -> Path:
        return self.dir("outputs")

    @property
    def catalogs(self) -> Path:
        return self.dir("catalogs")

    @property
    def logs(self) -> Path:
        return self.dir("logs")

    def load_settings(self) -> Settings:
        if self.config_file.exists():
            data = yaml.safe_load(self.config_file.read_text(encoding="utf-8")) or {}
            settings = Settings.from_dict(data)
        else:
            settings = Settings()
        return settings

    def save_settings(self, settings: Settings) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(
            yaml.safe_dump(settings.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    def ensure_token(self, settings: Settings) -> Settings:
        if not settings.token:
            settings.token = secrets.token_urlsafe(24)
            self.save_settings(settings)
        return settings

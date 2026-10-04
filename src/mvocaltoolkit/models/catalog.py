"""Model catalogs.

A catalog is a JSON file listing models that can be downloaded on demand. The toolkit merges:
    - the built-in catalog (mvocaltoolkit/catalog/*.json),
    - user catalogs in <home>/catalogs/*.json,
    - remote catalogs listed in settings.catalogs (URLs, cached locally).

Model sources:
    {"type": "url", "url": "https://.../model.zip", "sha256": "..."}
    {"type": "github_release", "repo": "owner/name", "tag": "v1", "asset": "*.zip"}   asset: glob or list of globs
    {"type": "huggingface", "repo": "owner/name", "files": ["model.ckpt", "config.yaml"], "revision": "main"}
    {"type": "engine"}   the engine downloads the model itself (e.g. Whisper models)

An entry with "type": "pack" is an archive that contains several models (like the LabelMakr model pack).
All model folders found inside are registered as separate models named <prefix><folder name>.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from ..settings import Home, Settings

BUILTIN_DIR = Path(__file__).parent.parent / "catalog"


class ModelSource(BaseModel):
    type: Literal["url", "github_release", "huggingface", "engine"] = "url"
    url: str | None = None
    sha256: str | None = None
    repo: str | None = None
    tag: str | None = None
    asset: str | list[str] | None = None
    files: list[str] | None = None
    revision: str | None = None
    # extract archives (zip/tar) after download
    extract: bool = True
    # use only this sub-folder of the extracted archive (glob allowed)
    subpath: str | None = None


class CatalogEntry(BaseModel):
    id: str
    engine: str
    type: Literal["model", "pack"] = "model"
    name: str = ""
    version: str = ""
    languages: list[str] = Field(default_factory=list)
    description: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    size_hint: str = ""
    source: ModelSource = Field(default_factory=ModelSource)
    # parameters passed to the engine together with the model (e.g. whisper model name, language id)
    params: dict[str, Any] = Field(default_factory=dict)
    # default options of the operation (e.g. AP detector, rules for post-processing)
    defaults: dict[str, Any] = Field(default_factory=dict)
    # text frontend used before alignment (normalization, G2P conventions): en, ja, zh, ko, fr, ru...
    text_frontend: str | None = None
    # packs: prefix of the registered model ids
    prefix: str = ""
    tags: list[str] = Field(default_factory=list)
    catalog: str = ""


class Catalog:
    def __init__(self, home: Home, settings: Settings):
        self.home = home
        self.settings = settings
        self.entries: dict[str, CatalogEntry] = {}
        self.sources: list[str] = []

    def load(self, refresh_remote: bool = False) -> None:
        entries: dict[str, CatalogEntry] = {}
        sources: list[str] = []
        files = sorted(BUILTIN_DIR.glob("*.json")) + sorted(self.home.catalogs.glob("*.json"))
        for path in files:
            self._merge(entries, json.loads(path.read_text(encoding="utf-8")), str(path))
            sources.append(str(path))
        for ref in self.settings.catalogs:
            try:
                data = self._load_remote(ref, refresh_remote)
            except Exception as e:  # noqa: BLE001
                sources.append(f"{ref} (error: {e})")
                continue
            self._merge(entries, data, ref)
            sources.append(ref)
        self.entries = entries
        self.sources = sources

    def _merge(self, entries: dict[str, CatalogEntry], data: dict[str, Any], origin: str) -> None:
        name = data.get("name") or origin
        for raw in data.get("models", []):
            entry = CatalogEntry.model_validate({**raw, "catalog": name})
            entries[entry.id] = entry

    def _load_remote(self, ref: str, refresh: bool) -> dict[str, Any]:
        if not ref.startswith(("http://", "https://")):
            return json.loads(Path(ref).expanduser().read_text(encoding="utf-8"))
        cache = self.home.dir("catalogs/remote") / (str(abs(hash(ref))) + ".json")
        if cache.exists() and not refresh and time.time() - cache.stat().st_mtime < 24 * 3600:
            return json.loads(cache.read_text(encoding="utf-8"))
        try:
            response = httpx.get(ref, follow_redirects=True, timeout=30)
            response.raise_for_status()
            data = response.json()
            cache.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception:
            if cache.exists():
                return json.loads(cache.read_text(encoding="utf-8"))
            raise

    def get(self, model_id: str) -> CatalogEntry | None:
        return self.entries.get(model_id)

    def filter(self, engine: str | None = None, language: str | None = None) -> list[CatalogEntry]:
        result = []
        for entry in self.entries.values():
            if engine and entry.engine != engine:
                continue
            if language and entry.languages and not any(_lang_match(language, lang) for lang in entry.languages):
                continue
            result.append(entry)
        return result

    def add_user_catalog(self, ref: str) -> None:
        if ref not in self.settings.catalogs:
            self.settings.catalogs.append(ref)
            self.home.save_settings(self.settings)
        self.load(refresh_remote=True)


def _lang_match(wanted: str, available: str) -> bool:
    wanted = wanted.lower().replace("-", "_")
    available = available.lower().replace("-", "_")
    return available == wanted or available.split("_")[0] == wanted.split("_")[0] or available == "*"

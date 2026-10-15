"""Model catalogs.

A catalog is a JSON file listing models that can be downloaded on demand. The toolkit merges:
    - the main catalog: github.com/Megageorgio/mVocalToolkit-catalog (settings.catalog_url), read again at every
      start; when it can't be reached, the last copy read, else the one inside the toolkit (mvocaltoolkit/catalog),

    - user catalogs in <home>/catalogs/*.json,
    - remote catalogs listed in settings.catalogs (URLs, cached locally).

Model sources:
    {"type": "url", "url": "https://.../model.zip", "sha256": "..."}
    {"type": "github_release", "repo": "owner/name", "tag": "v1", "asset": "*.zip"}   asset: glob or list of globs
    {"type": "huggingface", "repo": "owner/name", "files": ["model.ckpt", "config.yaml"], "revision": "main"}
    {"type": "engine"}   the engine downloads the model itself (e.g. Whisper models)

An entry with "type": "pack" is an archive that contains several models (like the SOFA model pack v030).
All model folders found inside are registered as separate models named <prefix><folder name>.
A pack can list its models ("models": [{"id", "folder", "name", "languages", ...}]): they show up in the catalog
as separate models (so a GUI can offer them by language before anything is downloaded); using any of them
downloads the whole pack.

Every entry belongs to one or more tasks (what a GUI offers it for): align, transcribe, segment, midi, tempo,
separate, pitch. By default the task comes from the engine.

A catalog can also list public singing datasets ("datasets"), downloaded for training (see datasets.py). Every
dataset carries its license; one whose license doesn't allow training isn't offered at all.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from ..settings import Home, Settings

BUILTIN_DIR = Path(__file__).parent.parent / "catalog"
# the catalog format this toolkit reads ("format" in the file; a newer major format is not used)
FORMAT = 1


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


# task -> engines; the task of a model is derived from its engine unless the entry lists "tasks"
ENGINE_TASKS: dict[str, list[str]] = {
    "sofa": ["align"],
    "hubertfa": ["align"],
    "tifa": ["align"],
    "whisperx": ["transcribe"],
    "wfl_asr": ["segment"],
    "mcfl": ["segment"],
    "game": ["midi"],
    "tempo": ["tempo"],
    "separation": ["separate"],
    "pitch": ["pitch"],
    "vocoder": ["resynth"],
    "refiner": ["refine"],
    "diffsinger": ["train"],
    "singingvocoders": ["train"],
    "g2p": ["train"],
    "rvc": ["train"],
    "tifa_train": ["train"],
    "sofa_train": ["train"],
    "game_train": ["train"],
    "label_train": ["train"],
    "refiner_train": ["train"],
    "qwen_tts": ["train"],
    "voxcpm": ["train"],

}
# ids models had in earlier catalogs: installed folders are renamed on start
LEGACY_IDS: dict[str, str] = {
    "labelmakr-pack-v030": "sofa-pack-v030",
    "labelmakr-tgm-en-v100": "sofa-en-tgm-v1.0.0",
    "labelmakr-tgm_en_v100": "sofa-tgm_en_v100",
    "labelmakr-tgm_sofa_en": "sofa-pack-tgm_sofa_en",
    "labelmakr-millefeuille_fr": "sofa-pack-millefeuille_fr",
    "labelmakr-suco_zh": "sofa-pack-suco_zh",
    "labelmakr-colstone_jp": "sofa-pack-colstone_jp",
    "labelmakr-colstone_ko": "sofa-pack-colstone_ko"
}

TASKS = ["transcribe", "align", "segment", "midi", "tempo", "pitch", "separate", "resynth", "refine", "train"]


def engine_tasks(engine: str) -> list[str]:
    return list(ENGINE_TASKS.get(engine, []))


class PackMember(BaseModel):
    """A model inside a pack archive."""

    id: str
    folder: str = Field("", description="Folder name inside the archive (default: id without the pack prefix)")
    name: str = ""
    author: str = ""
    languages: list[str] = Field(default_factory=list)
    text_frontend: str | None = None
    description: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


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
    # what the model is used for (default: from the engine)
    tasks: list[str] = Field(default_factory=list)
    # packs: the models inside (optional)
    models: list[PackMember] = Field(default_factory=list)
    # a model that is part of a pack: id of the pack entry (using the model downloads the pack)
    pack: str | None = None
    catalog: str = ""

    def task_list(self) -> list[str]:
        return list(self.tasks) or engine_tasks(self.engine)

    def member_for_folder(self, folder: str) -> PackMember | None:
        for member in self.models:
            if (member.folder or member.id.removeprefix(self.prefix)) == folder:
                return member
        return None


class DatasetLicense(BaseModel):
    """What a dataset's license allows. A dataset whose license doesn't allow training is never offered."""

    name: str
    url: str = ""
    summary: str = ""
    training: bool = False
    publish_models: bool = False
    commercial: bool = False
    redistribution: bool = False
    # text to credit the dataset with (in a model's description, a voicebank's readme)
    attribution: str = ""


class DatasetSource(ModelSource):
    # sub-folder of the dataset folder the files go to (e.g. labels made by someone else for the same audio)
    into: str = ""
    # what this part is: "audio", "labels", "audio+labels"
    part: str = ""


class DatasetEntry(BaseModel):
    """A public singing dataset that can be downloaded for training (catalog key "datasets")."""

    id: str
    name: str
    author: str = ""
    languages: list[str] = Field(default_factory=list)
    hours: float | None = None
    singers: int | None = None
    description: str = ""
    homepage: str = ""
    citation: str = ""
    # what the dataset has besides audio: "phonemes", "words", "notes", "midi", "musicxml", "f0"...
    labels: list[str] = Field(default_factory=list)
    label_format: str = ""
    sample_rate: int | None = None
    size_hint: str = ""
    license: DatasetLicense
    sources: list[DatasetSource] = Field(default_factory=list)
    # the files can't be downloaded by the toolkit (a form, a login): the user downloads them from this page
    manual_download: str = ""
    tags: list[str] = Field(default_factory=list)
    catalog: str = ""


class Catalog:
    def __init__(self, home: Home, settings: Settings, engines: set[str] | None = None):
        self.home = home
        self.settings = settings
        # engines this toolkit has: models of other engines (a newer catalog) are left out
        self.engines = engines
        self.entries: dict[str, CatalogEntry] = {}
        self.datasets: dict[str, DatasetEntry] = {}
        self.sources: list[str] = []

    @property
    def main_url(self) -> str:
        return os.environ.get("MVT_CATALOG_URL", self.settings.catalog_url).strip()

    def _main(self, fetch: bool) -> tuple[dict[str, Any], str] | None:
        """The main catalog: fetched now ([fetch]), else the last copy read; None: use the one inside the toolkit."""
        url = self.main_url
        if not url:
            return None
        cache = self.home.dir("catalogs/remote") / "main.json"
        if fetch:
            try:
                if url.startswith(("http://", "https://")):
                    response = httpx.get(url, follow_redirects=True, timeout=8)
                    response.raise_for_status()
                    data = response.json()
                else:
                    data = json.loads(Path(url).expanduser().read_text(encoding="utf-8"))
                if _readable(data):
                    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                    return data, url
            except Exception:  # noqa: BLE001  (offline: the last copy)
                pass
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if _readable(data):
                    return data, f"{url} (copy of {time.strftime('%Y-%m-%d %H:%M', time.localtime(cache.stat().st_mtime))})"
            except (OSError, ValueError):
                pass
        return None

    def load(self, refresh_remote: bool = False, fetch_main: bool = False) -> None:
        """[fetch_main]: read the main catalog from its URL now (at start and on an update check)."""
        entries: dict[str, CatalogEntry] = {}
        datasets: dict[str, DatasetEntry] = {}
        sources: list[str] = []
        main = self._main(fetch_main or refresh_remote)
        if main is not None:
            for filename in ("training.json", "preparation.json"):
                assets = BUILTIN_DIR / filename
                if assets.is_file():
                    self._merge(entries, json.loads(assets.read_text(encoding="utf-8")), str(assets), datasets)
            self._merge(entries, main[0], main[1], datasets)
            sources.append(main[1])
            builtin: list[Path] = []
        else:
            builtin = sorted(BUILTIN_DIR.glob("*.json"))
        for path in builtin + sorted(self.home.catalogs.glob("*.json")):
            self._merge(entries, json.loads(path.read_text(encoding="utf-8")), str(path), datasets)
            sources.append(str(path))
        for ref in self.settings.catalogs:
            try:
                data = self._load_remote(ref, refresh_remote)
            except Exception as e:  # noqa: BLE001
                sources.append(f"{ref} (error: {e})")
                continue
            self._merge(entries, data, ref, datasets)
            sources.append(ref)
        self.entries = entries
        self.datasets = datasets
        self.sources = sources

    def _merge(self, entries: dict[str, CatalogEntry], data: dict[str, Any], origin: str,
               datasets: dict[str, DatasetEntry] | None = None) -> None:
        name = data.get("name") or origin
        for raw in data.get("datasets", []) if datasets is not None else []:
            try:
                dataset = DatasetEntry.model_validate({**raw, "catalog": name})
            except ValueError:  # no license, or a field this toolkit can't read: not offered
                continue
            if dataset.license.training:
                datasets[dataset.id] = dataset
        for raw in data.get("models", []):
            if self.engines is not None and raw.get("engine") not in self.engines:
                continue
            try:
                entry = CatalogEntry.model_validate({**raw, "catalog": name})
            except ValueError:  # a field this toolkit doesn't know how to read
                continue
            entries[entry.id] = entry
            if entry.type == "pack":
                for member in entry.models:
                    entries[member.id] = CatalogEntry(
                        id=member.id,
                        engine=entry.engine,
                        name=member.name or member.id,
                        version=entry.version,
                        languages=member.languages or entry.languages,
                        description=member.description,
                        author=member.author or entry.author,
                        license=entry.license,
                        homepage=entry.homepage,
                        size_hint=entry.size_hint,
                        source=entry.source,
                        params={**entry.params, **member.params},
                        defaults={**entry.defaults, **member.defaults},
                        text_frontend=member.text_frontend or entry.text_frontend,
                        tags=member.tags or entry.tags,
                        tasks=entry.tasks,
                        pack=entry.id,
                        catalog=name,
                    )

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

    def filter(self, engine: str | None = None, language: str | None = None, task: str | None = None,
               ) -> list[CatalogEntry]:
        result = []
        for entry in self.entries.values():
            if engine and entry.engine not in engine.split(","):
                continue
            if task and task not in entry.task_list():
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


def _readable(data: Any) -> bool:
    """A catalog this toolkit can read: a "models" list, a format it knows (none = 1)."""
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        return False
    try:
        return int(str(data.get("format", FORMAT)).split(".")[0]) <= FORMAT
    except ValueError:
        return False


def lang_match(wanted: str, available: str) -> bool:
    return _lang_match(wanted, available)


def _lang_match(wanted: str, available: str) -> bool:
    wanted = wanted.lower().replace("-", "_")
    available = available.lower().replace("-", "_")
    return available == wanted or available.split("_")[0] == wanted.split("_")[0] or available == "*"

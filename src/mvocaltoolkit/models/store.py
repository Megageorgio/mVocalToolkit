"""Installed models: download, extraction, registration, import of local models.

Each installed model lives in <home>/models/<id>/ with a model.json manifest:
    {"id", "engine", "name", "languages", "layout": {...detected files...}, "entry": {...catalog entry...}}
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx
from pydantic import BaseModel, Field

from ..settings import Home, Settings
from . import layout as layouts
from .catalog import Catalog, CatalogEntry, ModelSource

# (fraction or None, message, detail=None); detail of a download: file, file_index, files, done, total (bytes),
# speed (bytes/s), eta (s)
ProgressFunc = Callable[..., None]

ARCHIVE_SUFFIXES = (".zip", ".oudep", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".rar", ".7z")
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".onnx")


class InstalledModel(BaseModel):
    id: str
    engine: str
    name: str = ""
    version: str = ""
    languages: list[str] = Field(default_factory=list)
    path: str = ""
    layout: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    text_frontend: str | None = None
    tasks: list[str] = Field(default_factory=list)
    description: str = ""
    source: str = ""  # catalog id / pack id / "import"
    installed_at: float = Field(default_factory=time.time)

    def file(self, key: str) -> Path | None:
        value = self.layout.get(key)
        if not value or not isinstance(value, str):
            return None
        return Path(self.path) / value


class ModelNotFound(KeyError):
    pass


class ModelStore:
    def __init__(self, home: Home, settings: Settings, catalog: Catalog):
        self.home = home
        self.settings = settings
        self.catalog = catalog
        self._migrate_ids()

    def _migrate_ids(self) -> None:
        """Models installed under ids of earlier catalogs get their current ids (folder and manifest)."""
        from .catalog import LEGACY_IDS  # noqa: PLC0415

        for old, new in LEGACY_IDS.items():
            src = self.home.models / _safe(old)
            dst = self.home.models / _safe(new)
            if not (src / "model.json").exists() or dst.exists():
                continue
            try:
                src.rename(dst)
                manifest = dst / "model.json"
                data = json.loads(manifest.read_text(encoding="utf-8"))
                data["id"] = new
                manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                pass

    def get_installed(self, model_id: str) -> InstalledModel | None:
        from .catalog import LEGACY_IDS  # noqa: PLC0415

        model_id = LEGACY_IDS.get(model_id, model_id)
        return self._get_installed(model_id)

    # ---------- installed models ----------

    def installed(self) -> dict[str, InstalledModel]:
        result: dict[str, InstalledModel] = {}
        for manifest in sorted(self.home.models.glob("*/model.json")):
            try:
                model = InstalledModel.model_validate(json.loads(manifest.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
            model.path = str(manifest.parent)
            result[model.id] = model
        return result

    def _get_installed(self, model_id: str) -> InstalledModel | None:
        manifest = self.home.models / _safe(model_id) / "model.json"
        if not manifest.exists():
            return None
        model = InstalledModel.model_validate(json.loads(manifest.read_text(encoding="utf-8")))
        model.path = str(manifest.parent)
        return model

    def remove(self, model_id: str) -> bool:
        folder = self.home.models / _safe(model_id)
        if not folder.exists():
            return False
        shutil.rmtree(folder)
        return True

    def resolve_local(self, engine: str | Sequence[str], path: str) -> InstalledModel:
        """Uses a model folder (or file) given by path without installing it."""
        engines = [engine] if isinstance(engine, str) else list(engine)
        root = Path(path).expanduser()
        if root.is_file():
            root = root.parent
        for name in engines:
            detected = layouts.detect(name, root)
            if detected is not None and (detected or name not in layouts.DETECTORS):
                return InstalledModel(id=f"path:{root}", engine=name, name=root.name, path=str(root),
                                      layout=detected, languages=detected.get("languages", []), source="path")
        raise ModelNotFound(f"No {' / '.join(engines)} model found in {root}")

    async def require(self, engine: str | Sequence[str], model: str,
                      progress: ProgressFunc | None = None) -> InstalledModel:
        """Returns an installed model by id or path, downloading it from the catalog if needed.

        engine can be a list: any of these engines is accepted (e.g. sofa or hubertfa for alignment)."""
        engines = [engine] if isinstance(engine, str) else list(engine)
        if model.startswith("path:") or Path(model).expanduser().is_absolute():
            return self.resolve_local(engines, model.removeprefix("path:"))
        from .catalog import LEGACY_IDS  # noqa: PLC0415

        model = LEGACY_IDS.get(model, model)
        installed = self.get_installed(model)
        if installed is not None:
            if installed.engine not in engines:
                raise ModelNotFound(f"Model {model} is for engine {installed.engine}, not {' / '.join(engines)}")
            return installed
        entry = self.catalog.get(model)
        if entry is None:
            raise ModelNotFound(f"Model {model} is not installed and not found in the catalogs")
        if entry.engine not in engines:
            raise ModelNotFound(f"Model {model} is for engine {entry.engine}, not {' / '.join(engines)}")
        if not self.settings.auto_download_models:
            raise ModelNotFound(f"Model {model} is not installed (automatic download is disabled)")
        if entry.pack:
            pack = self.catalog.get(entry.pack)
            if pack is None:
                raise ModelNotFound(f"Pack {entry.pack} of model {model} is not in the catalogs")
            await self.download(pack, progress)
        else:
            await self.download(entry, progress)
        installed = self.get_installed(model)
        if installed is None:
            raise ModelNotFound(f"Model {model} was downloaded but not found")
        return installed

    # ---------- download ----------

    async def download(self, entry: CatalogEntry, progress: ProgressFunc | None = None) -> list[InstalledModel]:
        progress = _with_detail(progress) if progress else (lambda _v, _m, _d=None: None)
        if entry.source.type == "engine":
            # the engine downloads it on first use; register so that it shows as installed
            target = self.home.models / _safe(entry.id)
            target.mkdir(parents=True, exist_ok=True)
            return [self._register(entry, target, {}, source=entry.id)]

        # downloaded files (and .part files of interrupted downloads) are kept until the model is installed,
        # so an interrupted download resumes
        work = self.home.cache / "work" / _safe(entry.id)
        work.mkdir(parents=True, exist_ok=True)
        files = await self._fetch(entry.source, work, progress)
        progress(None, "Extracting", {})
        content = work / "content"
        shutil.rmtree(content, ignore_errors=True)
        content.mkdir()
        for file in files:
            if entry.source.extract and file.name.lower().endswith(ARCHIVE_SUFFIXES):
                _extract(file, content / _strip_archive_suffix(file.name))
            else:
                shutil.copy2(str(file), str(content / file.name))
        root = _flatten(content)
        if entry.source.subpath:
            matches = sorted(root.glob(entry.source.subpath))
            if not matches:
                raise FileNotFoundError(f"{entry.source.subpath} not found in the downloaded files")
            root = matches[0]

        if entry.type == "pack":
            installed = self._register_pack(entry, root)
        else:
            dirs = layouts.find_model_dirs(entry.engine, root)
            model_root = dirs[0] if dirs else root
            detected = layouts.detect(entry.engine, model_root)
            if detected is None:
                raise ValueError(f"Downloaded files don't look like a {entry.engine} model")
            target = self.home.models / _safe(entry.id)
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(model_root), str(target))
            installed = [self._register(entry, target, detected, source=entry.id)]
        shutil.rmtree(work, ignore_errors=True)
        progress(1.0, "Installed", {})
        return installed

    def _register_pack(self, entry: CatalogEntry, root: Path) -> list[InstalledModel]:
        installed = []
        for model_dir in layouts.find_model_dirs(entry.engine, root):
            detected = layouts.detect(entry.engine, model_dir)
            if detected is None:
                continue
            member = entry.member_for_folder(model_dir.name)
            model_id = member.id if member else f"{entry.prefix}{model_dir.name}"
            known = self.catalog.get(model_id) if member else None
            sub = known or CatalogEntry(
                id=model_id,
                engine=entry.engine,
                name=f"{model_dir.name} ({entry.name or entry.id})",
                version=entry.version,
                languages=_guess_languages(model_dir.name) or detected.get("languages") or entry.languages,
                author=entry.author,
                license=entry.license,
                defaults=entry.defaults,
                params=entry.params,
                tasks=entry.tasks,
                text_frontend=_guess_frontend(model_dir.name) or entry.text_frontend,
            )
            target = self.home.models / _safe(model_id)
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(model_dir), str(target))
            installed.append(self._register(sub, target, detected, source=entry.id))
        if not installed:
            raise ValueError(f"No {entry.engine} models found in pack {entry.id}")
        return installed

    def _register(self, entry: CatalogEntry, folder: Path, detected: dict[str, Any], source: str) -> InstalledModel:
        model = InstalledModel(
            id=entry.id,
            engine=entry.engine,
            name=entry.name or entry.id,
            version=entry.version,
            languages=entry.languages or list(detected.get("languages") or []),
            path=str(folder),
            layout=detected,
            params=entry.params,
            defaults=entry.defaults,
            text_frontend=entry.text_frontend,
            tasks=entry.task_list(),
            description=entry.description,
            source=source,
        )
        (folder / "model.json").write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return model

    def import_local(
        self,
        engine: str,
        path: str,
        model_id: str | None = None,
        name: str | None = None,
        languages: list[str] | None = None,
        text_frontend: str | None = None,
        copy: bool = True,
    ) -> list[InstalledModel]:
        """Registers a local model folder or archive (several models inside are registered separately)."""
        src = Path(path).expanduser()
        work = self.home.cache / "work" / ("import-" + hashlib.sha1(str(src).encode()).hexdigest()[:8])
        shutil.rmtree(work, ignore_errors=True)
        if src.is_file() and src.name.lower().endswith(ARCHIVE_SUFFIXES):
            _extract(src, work)
            root = _flatten(work)
        elif src.is_file() and src.name.lower().endswith(CHECKPOINT_SUFFIXES):
            # one checkpoint of a training run: it and the small files next to it (config, phonemes, languages)
            root = work / _safe(model_id or src.stem)
            root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, root / src.name)
            for f in src.parent.iterdir():
                if f.is_file() and f.suffix.lower() in (".yaml", ".yml", ".txt", ".json") and f.stat().st_size < 2_000_000:
                    shutil.copy2(f, root / f.name)
            copy = False
        elif src.is_dir():
            root = src
        else:
            raise FileNotFoundError(f"{src} is not a folder or archive")
        dirs = layouts.find_model_dirs(engine, root)
        if not dirs:
            raise ValueError(f"No {engine} models found in {src}")
        installed = []
        for model_dir in dirs:
            mid = model_id if (model_id and len(dirs) == 1) else _safe(model_dir.name if model_dir != root else src.stem)
            detected = layouts.detect(engine, model_dir) or {}
            entry = CatalogEntry(
                id=mid,
                engine=engine,
                name=name or model_dir.name,
                languages=languages or list(detected.get("languages") or []) or _guess_languages(model_dir.name),
                text_frontend=text_frontend or _guess_frontend(model_dir.name),
            )
            target = self.home.models / _safe(mid)
            shutil.rmtree(target, ignore_errors=True)
            if copy or root != src:
                shutil.copytree(model_dir, target)
            else:
                shutil.move(str(model_dir), str(target))
            installed.append(self._register(entry, target, layouts.detect(engine, target) or {}, source="import"))
        shutil.rmtree(work, ignore_errors=True)
        return installed

    # ---------- fetching ----------

    async def _fetch(self, source: ModelSource, work: Path, progress: ProgressFunc) -> list[Path]:
        urls: list[tuple[str, str, int | None]] = []  # url, filename, size
        if source.type == "url":
            if not source.url:
                raise ValueError("url source without url")
            urls.append((source.url, source.url.rstrip("/").split("/")[-1].split("?")[0] or "download", None))
        elif source.type == "github_release":
            urls += await self._github_assets(source)
        elif source.type == "huggingface":
            revision = source.revision or "main"
            for name in source.files or []:
                urls.append((f"https://huggingface.co/{source.repo}/resolve/{revision}/{name}", name, None))
        else:
            raise ValueError(f"Unsupported source type {source.type}")
        files = []
        total = len(urls)
        for index, (url, filename, size) in enumerate(urls):
            target = work / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and (size is None or target.stat().st_size == size):
                files.append(target)  # downloaded before (e.g. extraction was interrupted)
                continue

            started = {"t": time.monotonic(), "done": None}

            def file_progress(done: int, length: int | None, i=index, name=filename, st=started):
                fraction = (done / length) if length else None
                overall = ((i + (fraction or 0)) / total) if total else None
                if st["done"] is None:
                    st["done"] = done  # a resumed download doesn't count towards the speed
                elapsed = max(1e-3, time.monotonic() - st["t"])
                speed = (done - st["done"]) / elapsed / 2**20
                size_text = f"{done / 2**20:.1f} / {length / 2**20:.1f} MB" if length else f"{done / 2**20:.1f} MB"
                left = ""
                if length and speed > 0.05:
                    secs = int((length - done) / 2**20 / speed)
                    left = f", {secs // 60}:{secs % 60:02d} left"
                files_text = f" [{i + 1}/{total}]" if total > 1 else ""
                detail = {"file": name, "file_index": i + 1, "files": total, "done": done, "total": length,
                          "speed": speed * 2**20, "eta": int((length - done) / 2**20 / speed) if length and speed > 0.05 else None}
                progress(overall, f"Downloading {name}{files_text}: {size_text}, {speed:.1f} MB/s{left}", detail)

            await download_file(url, target, file_progress, sha256=source.sha256 if total == 1 else None,
                                headers=self._auth_headers(url))
            files.append(target)
        return files

    def _auth_headers(self, url: str) -> dict[str, str]:
        if "huggingface.co" in url and self.settings.hf_token:
            return {"Authorization": f"Bearer {self.settings.hf_token}"}
        return {}

    async def _github_assets(self, source: ModelSource) -> list[tuple[str, str, int | None]]:
        if not source.repo or not source.tag:
            raise ValueError("github_release source needs repo and tag")
        patterns = source.asset if isinstance(source.asset, list) else [source.asset or "*"]
        if not any(any(c in p for c in "*?[") for p in patterns):
            # Exact asset names: direct download links, no GitHub API (60 requests/hour without a token)
            base = f"https://github.com/{source.repo}/releases"
            base += "/latest/download" if source.tag == "latest" else f"/download/{source.tag}"
            return [(f"{base}/{name}", name, None) for name in dict.fromkeys(patterns)]
        data = await self._github_release(source.repo, source.tag)
        assets = data.get("assets", [])
        selected = []
        for pattern in patterns:
            matched = [a for a in assets if fnmatch.fnmatch(a["name"], pattern)]
            if not matched:
                names = ", ".join(a["name"] for a in assets) or "none"
                raise FileNotFoundError(
                    f"No asset matching '{pattern}' in {source.repo} release {source.tag}. Available: {names}"
                )
            for asset in matched:
                if asset not in selected:
                    selected.append(asset)
        return [(a["browser_download_url"], a["name"], a.get("size")) for a in selected]

    async def _github_release(self, repo: str, tag: str) -> dict[str, Any]:
        """Release metadata from the GitHub API, cached for an hour (a stale cache is used when the API fails)."""
        cache = self.home.dir("cache/github") / (_safe(f"{repo}@{tag}") + ".json")
        if cache.exists() and time.time() - cache.stat().st_mtime < 3600:
            return json.loads(cache.read_text(encoding="utf-8"))
        headers = {"Accept": "application/vnd.github+json"}
        token = self.settings.github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if tag == "latest":
            url = f"https://api.github.com/repos/{repo}/releases/latest"
        else:
            url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            if cache.exists():
                return json.loads(cache.read_text(encoding="utf-8"))
            status = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else None
            if status in (403, 429):
                raise RuntimeError(
                    f"GitHub API rate limit exceeded while looking up release {tag} of {repo}. Try again in an hour, "
                    "or set github_token in config.yaml (or the GITHUB_TOKEN environment variable) to a GitHub "
                    "personal access token; it needs no scopes."
                ) from e
            raise
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data


def _with_detail(progress: Callable[..., None]) -> Callable[..., None]:
    """[progress] as a function of (value, message, detail); older ones take (value, message) only."""
    import inspect  # noqa: PLC0415

    try:
        params = inspect.signature(progress).parameters.values()
        takes_three = any(p.kind == p.VAR_POSITIONAL for p in params) or len(
            [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 3
    except (TypeError, ValueError):
        takes_three = False
    return progress if takes_three else (lambda v, m, _d=None: progress(v, m))


async def download_file(
    url: str,
    target: Path,
    progress: Callable[[int, int | None], None] | None = None,
    sha256: str | None = None,
    headers: dict[str, str] | None = None,
) -> Path:
    """Downloads with resume support (keeps a .part file between attempts)."""
    part = target.with_name(target.name + ".part")
    request_headers = dict(headers or {})
    offset = part.stat().st_size if part.exists() else 0
    if offset:
        request_headers["Range"] = f"bytes={offset}-"
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60, read=300)) as client:
        async with client.stream("GET", url, headers=request_headers) as response:
            if response.status_code == 416:  # already complete
                pass
            else:
                response.raise_for_status()
                if response.status_code != 206:
                    offset = 0
                length = response.headers.get("content-length")
                total = int(length) + offset if length else None
                mode = "ab" if offset else "wb"
                done = offset
                last = 0.0
                with open(part, mode) as f:
                    async for chunk in response.aiter_bytes(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if progress and now - last > 0.3:
                            last = now
                            progress(done, total)
                if progress:
                    progress(done, total)
    if sha256:
        digest = hashlib.sha256()
        with open(part, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest().lower() != sha256.lower():
            part.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {target.name}")
    part.replace(target)
    return target


def _extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    # .oudep: an OpenUtau dependency package, a zip under another name
    if name.endswith((".zip", ".oudep")):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                # zips made on Windows/Mac sometimes have names in cp437 or with __MACOSX junk
                if info.filename.startswith("__MACOSX") or info.filename.endswith(".DS_Store"):
                    continue
                _check_member(target, info.filename)
                z.extract(info, target)
    elif name.endswith(".7z"):
        try:
            import py7zr  # noqa: PLC0415
        except ImportError:
            _extract_with_tool(archive, target)
            return
        with py7zr.SevenZipFile(archive) as z:
            for n in z.getnames():
                _check_member(target, n)
            z.extractall(target)
    elif name.endswith(".rar"):
        _extract_with_tool(archive, target)
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                _check_member(target, member.name)
            t.extractall(target)


def _extract_with_tool(archive: Path, target: Path) -> None:
    """RAR (and 7z without py7zr): a program that can read it is used (tar on Windows 11 reads RAR)."""
    import subprocess  # noqa: PLC0415

    candidates: list[list[str]] = []
    for exe in ("7z", "7za", "7zz", r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        found = shutil.which(exe) or (exe if Path(exe).is_file() else None)
        if found:
            candidates.append([found, "x", "-y", f"-o{target}", str(archive)])
    for exe in ("unrar", r"C:\Program Files\WinRAR\UnRAR.exe", r"C:\Program Files (x86)\WinRAR\UnRAR.exe"):
        found = shutil.which(exe) or (exe if Path(exe).is_file() else None)
        if found and archive.name.lower().endswith(".rar"):
            candidates.append([found, "x", "-o+", "-y", str(archive), str(target) + os.sep])
    for exe in ("bsdtar", "tar"):
        found = shutil.which(exe)
        if found:
            candidates.append([found, "-xf", str(archive), "-C", str(target)])
    errors = []
    for cmd in candidates:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=1800)
        except OSError as e:
            errors.append(f"{cmd[0]}: {e}")
            continue
        if r.returncode == 0 and any(target.iterdir()):
            for p in target.rglob("*"):
                _check_member(target, str(p.relative_to(target)))
            return
        errors.append(f"{Path(cmd[0]).name}: {(r.stderr or r.stdout).strip()[-200:]}")
    raise RuntimeError(
        f"Can't unpack {archive.name}: install 7-Zip (7-zip.org) or unpack it yourself and add the folder as your own model."
        + (" Tried: " + "; ".join(errors) if errors else "")
    )


def _check_member(target: Path, name: str) -> None:
    resolved = (target / name).resolve()
    if not str(resolved).startswith(str(target.resolve())):
        raise ValueError(f"Unsafe path in archive: {name}")


def _strip_archive_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _flatten(path: Path) -> Path:
    """Descends into single nested folders (zip/folder/folder/files -> folder with files)."""
    current = path
    while True:
        children = [p for p in current.iterdir() if not p.name.startswith(".")]
        if len(children) == 1 and children[0].is_dir():
            current = children[0]
        else:
            return current


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.@+" else "_" for c in name)


_LANG_HINTS = {
    "en": "en", "eng": "en", "english": "en",
    "ja": "ja", "jp": "ja", "jpn": "ja", "japanese": "ja",
    "zh": "zh", "cn": "zh", "mandarin": "zh", "chinese": "zh",
    "yue": "yue", "cantonese": "yue",
    "ko": "ko", "kr": "ko", "kor": "ko", "korean": "ko",
    "fr": "fr", "fra": "fr", "french": "fr",
    "ru": "ru", "rus": "ru", "russian": "ru",
    "es": "es", "spa": "es", "spanish": "es",
    "it": "it", "ita": "it", "italian": "it",
    "pt": "pt", "por": "pt", "portuguese": "pt",
    "de": "de", "ger": "de", "german": "de",
    "indonesian": "id",
    "pl": "pl", "polish": "pl",
    "ukrainian": "uk",
}


def _guess_languages(name: str) -> list[str]:
    tokens = [t for t in name.lower().replace("-", "_").replace(".", "_").split("_") if t]
    langs = []
    for token in tokens:
        lang = _LANG_HINTS.get(token)
        if lang and lang not in langs:
            langs.append(lang)
    return langs


def _guess_frontend(name: str) -> str | None:
    langs = _guess_languages(name)
    return langs[0] if langs else None


# ---------- listing for GUIs ----------


def model_listing(store: ModelStore, task: str | None = None, engine: str | None = None,
                  language: str | None = None) -> list[dict[str, Any]]:
    """Catalog entries (pack members included) + installed models, with an "installed" flag.

    Packs themselves are listed only when they don't list their models (otherwise the members are listed)."""
    from .catalog import engine_tasks, lang_match  # noqa: PLC0415

    installed = store.installed()
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in store.catalog.filter(engine, language, task):
        if entry.type == "pack" and entry.models:
            continue
        data = entry.model_dump(exclude={"source", "models"})
        data["tasks"] = entry.task_list()
        data["source_type"] = entry.source.type
        data["installed"] = entry.id in installed
        if entry.type == "pack":
            data["installed"] = any(m.source == entry.id for m in installed.values())
        result.append(data)
        seen.add(entry.id)
    engines = engine.split(",") if engine else None
    for model in installed.values():
        if model.id in seen or store.catalog.get(model.id) is not None:
            continue
        tasks = model.tasks or engine_tasks(model.engine)
        if engines and model.engine not in engines:
            continue
        if task and task not in tasks:
            continue
        if language and model.languages and not any(lang_match(language, lang) for lang in model.languages):
            continue
        data = model.model_dump()
        data.update({"installed": True, "type": "model", "tasks": tasks, "pack": None})
        result.append(data)
    return result

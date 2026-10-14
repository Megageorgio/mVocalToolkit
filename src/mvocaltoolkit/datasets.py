"""Public singing datasets from the catalogs, downloaded for training (other languages, more voices).

A dataset is only downloaded after its license was accepted (accept_license), and its folder keeps the license:
    <home>/datasets/<id>/LICENSE.txt   name, link, summary, what is allowed, the credit text
    <home>/datasets/<id>/dataset.json  the catalog entry it was downloaded from
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from .models.catalog import Catalog, DatasetEntry, lang_match
from .models.store import ARCHIVE_SUFFIXES, ModelStore, _extract, _safe, _with_detail


class LicenseNotAccepted(PermissionError):
    pass


class ManualDownload(RuntimeError):
    pass


def license_text(entry: DatasetEntry) -> str:
    lic = entry.license

    def yes(value: bool) -> str:
        return "yes" if value else "no"

    lines = [
        f"{entry.name}",
        f"License: {lic.name}" + (f" ({lic.url})" if lic.url else ""),
        "",
        lic.summary,
        "",
        f"Training models: {yes(lic.training)}",
        f"Publishing trained models: {yes(lic.publish_models)}",
        f"Commercial use: {yes(lic.commercial)}",
        f"Redistributing the data: {yes(lic.redistribution)}",
    ]
    if lic.attribution:
        lines += ["", f"Credit: {lic.attribution}"]
    if entry.homepage:
        lines += ["", f"Homepage: {entry.homepage}"]
    if entry.citation:
        lines += ["", "Citation:", entry.citation]
    return "\n".join(lines).strip() + "\n"


class DatasetStore:
    def __init__(self, models: ModelStore):
        self.models = models
        self.home = models.home

    @property
    def catalog(self) -> Catalog:
        return self.models.catalog

    @property
    def root(self) -> Path:
        return self.home.dir("datasets")

    def folder(self, dataset_id: str) -> Path:
        return self.root / _safe(dataset_id)

    def is_installed(self, dataset_id: str) -> bool:
        return (self.folder(dataset_id) / "dataset.json").is_file()

    def listing(self, language: str | None = None) -> list[dict[str, Any]]:
        result = []
        for entry in self.catalog.datasets.values():
            if language and not any(lang_match(language, lang) for lang in entry.languages):
                continue
            result.append({
                **entry.model_dump(),
                "installed": self.is_installed(entry.id),
                "path": str(self.folder(entry.id)) if self.is_installed(entry.id) else None,
            })
        return sorted(result, key=lambda d: (",".join(d["languages"]), d["name"].lower()))

    def remove(self, dataset_id: str) -> bool:
        folder = self.folder(dataset_id)
        if not folder.exists():
            return False
        shutil.rmtree(folder)
        return True

    async def download(self, entry: DatasetEntry, accept_license: bool,
                       progress: Callable[..., None] | None = None) -> Path:
        if not accept_license:
            raise LicenseNotAccepted(
                f"Read the license of {entry.name} and accept it to download:\n\n{license_text(entry)}")
        if entry.manual_download or not entry.sources:
            raise ManualDownload(
                f"{entry.name} can't be downloaded automatically (the authors ask to download it from their page): "
                f"{entry.manual_download or entry.homepage}")
        progress = _with_detail(progress) if progress else (lambda _v, _m, _d=None: None)
        target = self.folder(entry.id)
        work = self.home.cache / "work" / ("dataset-" + _safe(entry.id))
        work.mkdir(parents=True, exist_ok=True)
        staged = work / "content"
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir()
        count = len(entry.sources)
        for index, source in enumerate(entry.sources):
            def part_progress(value, message, detail=None, i=index):
                overall = (i + (value or 0)) / count if value is not None else None
                progress(overall, message, detail)

            files_dir = work / f"files{index}"
            files_dir.mkdir(exist_ok=True)
            # downloaded files stay in the work folder until the dataset is in place, so an interruption resumes
            files = await self.models._fetch(source, files_dir, part_progress)  # noqa: SLF001
            part_progress(None, "Extracting")
            dest = staged / source.into if source.into else staged
            dest.mkdir(parents=True, exist_ok=True)
            for file in files:
                if source.extract and file.name.lower().endswith(ARCHIVE_SUFFIXES):
                    unpacked = work / f"unpacked{index}"
                    shutil.rmtree(unpacked, ignore_errors=True)
                    _extract(file, unpacked)
                    root = _strip_wrapper(unpacked)
                    for item in root.iterdir():
                        shutil.move(str(item), str(dest / item.name))
                else:
                    shutil.copy2(str(file), str(dest / file.name))
        (staged / "LICENSE.txt").write_text(license_text(entry), encoding="utf-8")
        (staged / "dataset.json").write_text(entry.model_dump_json(indent=2), encoding="utf-8")
        shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(target))
        shutil.rmtree(work, ignore_errors=True)
        progress(1.0, "Downloaded", {})
        return target


def _strip_wrapper(path: Path) -> Path:
    """An archive with one top folder: that folder (only one level, so wavs/ and the like stay)."""
    children = [p for p in path.iterdir() if not p.name.startswith(".") and p.name != "__MACOSX"]
    return children[0] if len(children) == 1 and children[0].is_dir() else path

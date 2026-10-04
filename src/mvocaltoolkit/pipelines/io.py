"""Input resolution and output writing shared by all operations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import formats
from ..api_models import InputSpec, OutputSpec
from ..formats import ds_csv
from ..labels import Label
from ..settings import Home

_HTK_LINE = re.compile(r"^\s*\d+\s+\d+\s+\S+")


@dataclass
class Item:
    name: str
    audio: Path
    text: str | None = None
    words: list[str] | None = None
    phonemes: list[str] | None = None
    language: str | None = None
    # folder the outputs go to by default, and sub-folder relative to the input folder
    base_dir: Path = Path(".")
    rel_dir: Path = Path(".")
    # filled while processing
    tokens: list[str] | None = None
    data: dict = field(default_factory=dict)


def resolve_inputs(spec: InputSpec, home: Home) -> list[Item]:
    items: list[Item] = []
    for entry in spec.items:
        if entry.file_id:
            audio = _uploaded(home, entry.file_id)
        elif entry.path:
            audio = Path(entry.path).expanduser()
        else:
            raise ValueError("Each input item needs path or file_id")
        if not audio.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio}")
        item = Item(
            name=entry.name or audio.stem,
            audio=audio,
            text=entry.text,
            words=entry.words,
            phonemes=entry.phonemes,
            language=entry.language,
            base_dir=audio.parent,
        )
        if item.text is None and item.words is None and item.phonemes is None:
            item.text = read_sidecar_text(audio, spec.sidecar_text)
        items.append(item)
    if spec.folder:
        root = Path(spec.folder).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Folder not found: {root}")
        seen = {i.audio.resolve() for i in items}
        files: list[Path] = []
        for pattern in spec.patterns:
            files += list(root.rglob(pattern) if spec.recursive else root.glob(pattern))
        for audio in sorted(set(files)):
            if audio.resolve() in seen or any(part.startswith(".") for part in audio.relative_to(root).parts):
                continue
            seen.add(audio.resolve())
            items.append(
                Item(
                    name=audio.stem,
                    audio=audio,
                    text=read_sidecar_text(audio, spec.sidecar_text),
                    base_dir=root,
                    rel_dir=audio.parent.relative_to(root),
                )
            )
    if not items:
        raise ValueError("No input audio files")
    return items


def read_sidecar_text(audio: Path, extensions: list[str]) -> str | None:
    for ext in extensions:
        path = audio.with_suffix(ext)
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8-sig", errors="replace").strip()
        if not content:
            continue
        lines = [l for l in content.splitlines() if l.strip()]
        if lines and all(_HTK_LINE.match(l) for l in lines):
            continue  # a time-aligned label, not a transcription
        return " ".join(lines)
    return None


def _uploaded(home: Home, file_id: str) -> Path:
    folder = home.uploads / Path(file_id).name
    files = [p for p in folder.iterdir()] if folder.is_dir() else []
    if not files:
        raise FileNotFoundError(f"Uploaded file {file_id} not found")
    return files[0]


FORMAT_FOLDERS = {"htk": "htk", "textgrid": "textgrid", "json": "json", "audacity": "audacity"}


def output_folder(item: Item, out_dir: str | None, home: Home, job_id: str) -> Path:
    """Folder for non-label outputs (stems, f0, MIDI): out_dir/<sub-folder>, next to the audio, or
    <home>/outputs/<job> for uploaded files."""
    if out_dir:
        return Path(out_dir).expanduser() / item.rel_dir
    if item.audio.is_relative_to(home.uploads):
        return home.outputs / job_id
    return item.audio.parent


class OutputWriter:
    def __init__(self, spec: OutputSpec, home: Home, job_id: str):
        self.spec = spec
        self.formats = [formats.normalize_format(f) for f in spec.formats]
        self.home = home
        self.job_id = job_id
        self._csv_rows: dict[Path, list[dict[str, str]]] = {}

    def base(self, item: Item) -> Path:
        if self.spec.dir:
            base = Path(self.spec.dir).expanduser()
            if self.spec.layout != "flat":
                base = base / item.rel_dir
            return base
        if item.audio.is_relative_to(self.home.uploads):
            return self.home.outputs / self.job_id
        return item.audio.parent

    def path(self, item: Item, fmt: str) -> Path:
        base = self.base(item)
        ext = formats.EXTENSIONS[fmt]
        if self.spec.layout == "subfolders":
            return base / FORMAT_FOLDERS.get(fmt, fmt) / f"{item.name}{ext}"
        return base / f"{item.name}{ext}"

    def write(self, item: Item, label: Label) -> dict[str, str]:
        written: dict[str, str] = {}
        for fmt in self.formats:
            if fmt == "ds_csv":
                self._csv_rows.setdefault(self.base(item), []).append(ds_csv.row_from_label(item.name, label))
                continue
            path = self.path(item, fmt)
            if path.exists() and not self.spec.overwrite:
                continue
            formats.write(label, path, fmt)
            written[fmt] = str(path)
        return written

    def finalize(self) -> dict[str, str]:
        """Writes the collected transcriptions.csv files (one per output folder)."""
        written: dict[str, str] = {}
        for base, rows in self._csv_rows.items():
            path = base / self.spec.ds_csv_name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not self.spec.overwrite:
                continue
            path.write_text(ds_csv.dumps_rows(rows), encoding="utf-8")
            written[str(base)] = str(path)
        return written

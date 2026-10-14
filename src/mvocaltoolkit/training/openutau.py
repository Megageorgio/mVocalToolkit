"""An OpenUtau DiffSinger voicebank from the ONNX exports of runs, with a linter.

The layout OpenUtau's DiffSinger renderer reads (the one DiffTrainer and the community build):

    <bank>/
        character.txt           name=...
        character.yaml          singer_type: diffsinger, default_phonemizer, subbanks (speakers as colours)
        dsconfig.yaml           the acoustic model: acoustic, phonemes, languages, vocoder, speakers, use_*_embed...
        dsmain/                 acoustic.onnx, phonemes.json, languages.json, linguistic.onnx (from the variance run)
        embeds/                 <speaker>.emb of the acoustic model
        dsdur/                  dur.onnx, phonemes.json, languages.json, dsconfig.yaml, embeds/, dsdict*.yaml
        dspitch/                pitch.onnx, dsconfig.yaml, embeds/, dsdict*.yaml
        dsvariance/             variance.onnx, dsconfig.yaml, embeds/, dsdict*.yaml
        dsvocoder/              a custom vocoder (vocoder.onnx + vocoder.yaml); otherwise dsconfig names the
                                standard one, which OpenUtau downloads as a dependency
        readme.txt              what the bank was made of
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STANDARD_VOCODER = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
PHONEMIZER = "OpenUtau.Core.DiffSinger.DiffSingerPhonemizer"
# languages whose built-in OpenUtau phonemizers take language-tagged phonemes themselves
CANON_LANGUAGES = {"zh", "en", "de", "it", "ja", "yue", "ko", "pt", "ru", "es"}

# phoneme type by a first-letter table (as DiffTrainer); callers can give exact vowel lists per language
TYPE_BY_FIRST = {
    **{c: "vowel" for c in "aiueoAIUEONM"},
    **{c: "stop" for c in "bdgkpqt"},
    "c": "affricate", "j": "affricate",
    **{c: "fricative" for c in "fhsvz"},
    "l": "liquid", "r": "liquid",
    "m": "nasal", "n": "nasal",
    "w": "semivowel", "y": "semivowel",
}


@dataclass
class ModelExport:
    """One exported model folder (the output of the engine's export method)."""

    dir: Path
    name: str  # the file prefix: acoustic, variance (or whatever the export was named)

    @property
    def dsconfig(self) -> dict[str, Any]:
        path = self.dir / "dsconfig.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def file(self, suffix: str) -> Path | None:
        for candidate in (self.dir / f"{self.name}.{suffix}", self.dir / suffix):
            if candidate.is_file():
                return candidate
        return None

    def embeds(self) -> list[Path]:
        return sorted(p for p in self.dir.glob(f"{self.name}.*.emb"))

    def dictionaries(self) -> dict[str, Path]:
        out = {}
        for p in self.dir.glob("dictionary-*.txt"):
            out[p.stem.split("-", 1)[1]] = p
        return out


@dataclass
class VoicebankSpec:
    name: str
    out_dir: str
    acoustic: str  # export folder of the acoustic model
    variance: list[str] = field(default_factory=list)  # export folders with dur / pitch / variance models
    vocoder: str = ""  # folder with a custom vocoder (ONNX + vocoder.yaml), or empty for the standard one
    # phoneme -> type (vowel, stop, ...) overrides for the dsdict
    phoneme_types: dict[str, str] = field(default_factory=dict)
    vowels: dict[str, list[str]] = field(default_factory=dict)  # language -> vowels
    # shown in readme.txt
    notes: str = ""
    licenses: list[str] = field(default_factory=list)
    credits: list[str] = field(default_factory=list)
    overwrite: bool = False


@dataclass
class VoicebankResult:
    dir: str
    files: list[str]
    problems: list[dict[str, str]]
    speakers: list[str]
    languages: list[str]
    models: list[str]


def _detect_export(folder: Path) -> ModelExport:
    folder = Path(folder)
    cfg = folder / "dsconfig.yaml"
    if not cfg.is_file():
        raise FileNotFoundError(f"No dsconfig.yaml in {folder}: not an export folder")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    ref = data.get("acoustic") or data.get("linguistic") or data.get("phonemes") or ""
    name = str(ref).split(".")[0] if ref else "model"
    return ModelExport(folder, name)


def _phoneme_type(phoneme: str, language: str | None, spec: VoicebankSpec) -> str:
    short = phoneme.split("/", 1)[-1]
    if phoneme in spec.phoneme_types:
        return spec.phoneme_types[phoneme]
    if short in spec.phoneme_types:
        return spec.phoneme_types[short]
    if language and short in spec.vowels.get(language, []):
        return "vowel"
    return TYPE_BY_FIRST.get(short[:1], "stop")


def _yaml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def _write_dsdict(path: Path, symbols: list[tuple[str, str]], entries: list[tuple[str, list[str]]],
                  replacements: list[tuple[str, str]] | None = None) -> None:
    lines = ["symbols:", "- {symbol: SP, type: vowel}", "- {symbol: AP, type: vowel}"]
    seen = {"SP", "AP"}
    for symbol, kind in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        lines.append(f"- {{symbol: {_yaml_str(symbol)}, type: {kind}}}")
    if replacements:
        lines += ["", "replacements:"]
        for src, dst in replacements:
            lines.append(f"- {{from: {_yaml_str(src)}, to: {_yaml_str(dst)}}}")
    lines += ["", "entries:"]
    for grapheme, phones in entries:
        lines.append(f"- {{grapheme: {_yaml_str(grapheme)}, phonemes: [{', '.join(_yaml_str(p) for p in phones)}]}}")
    lines += ['- {grapheme: "SP", phonemes: [SP]}', '- {grapheme: "AP", phonemes: [AP]}', ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_dictionary(path: Path) -> list[tuple[str, list[str]]]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0].strip():
            out.append((parts[0].strip(), parts[1].split()))
    return out


def write_dsdicts(target: Path, dictionaries: dict[str, Path], phonemes: list[str], extra: list[str],
                  spec: VoicebankSpec, multilingual: bool) -> list[Path]:
    """dsdict.yaml (all languages) and dsdict-<lang>.yaml per language, as OpenUtau's phonemizers read them."""
    written = []
    globals_ = [p for p in extra if p not in ("AP", "SP")]
    if not multilingual or len(dictionaries) <= 1:
        lang = next(iter(dictionaries), None)
        symbols = [(p, _phoneme_type(p, lang, spec)) for p in globals_]
        entries = [(p, [p]) for p in globals_]
        for lang_code, path in dictionaries.items():
            for grapheme, phones in _read_dictionary(path):
                for ph in phones:
                    if ph not in {s for s, _ in symbols}:
                        symbols.append((ph, _phoneme_type(ph, lang_code, spec)))
                entries.append((grapheme, phones))
        out = target / "dsdict.yaml"
        _write_dsdict(out, symbols, entries)
        return [out]
    all_symbols: list[tuple[str, str]] = [(p, _phoneme_type(p, None, spec)) for p in globals_]
    all_entries: list[tuple[str, list[str]]] = [(p, [p]) for p in globals_]
    per_lang: dict[str, tuple[list[tuple[str, str]], list[tuple[str, list[str]]], list[tuple[str, str]]]] = {}
    for lang, path in dictionaries.items():
        symbols: list[tuple[str, str]] = []
        entries: list[tuple[str, list[str]]] = []
        replacements: list[tuple[str, str]] = []
        for grapheme, phones in _read_dictionary(path):
            full = [p if "/" in p or p in ("AP", "SP") else f"{lang}/{p}" for p in phones]
            for ph in full:
                if ph in ("AP", "SP"):
                    continue
                symbols.append((ph, _phoneme_type(ph, lang, spec)))
                short = ph.split("/", 1)[-1]
                replacements.append((short, ph))
            entries.append((grapheme, full))
        # phonemes of this language in the model's set that the dictionary doesn't use
        for ph in phonemes:
            if ph.startswith(f"{lang}/") and ph not in {s for s, _ in symbols}:
                symbols.append((ph, _phoneme_type(ph, lang, spec)))
        per_lang[lang] = (symbols, entries, list(dict.fromkeys(replacements)))
        all_symbols += symbols
        all_entries += entries
    for lang, (symbols, entries, replacements) in per_lang.items():
        out = target / f"dsdict-{lang}.yaml"
        _write_dsdict(out, all_symbols, entries, replacements if lang not in CANON_LANGUAGES else None)
        written.append(out)
    out = target / "dsdict.yaml"
    _write_dsdict(out, all_symbols, all_entries)
    written.append(out)
    return written


def assemble(spec: VoicebankSpec) -> VoicebankResult:
    problems: list[dict[str, str]] = []
    acoustic = _detect_export(Path(spec.acoustic))
    variances = [_detect_export(Path(v)) for v in spec.variance]
    bank = Path(spec.out_dir) / re.sub(r"[\\/:*?\"<>|]+", "_", spec.name.strip()).strip() or Path(spec.out_dir) / "voicebank"
    if bank.exists() and any(bank.iterdir()):
        if not spec.overwrite:
            raise FileExistsError(f"{bank} exists and is not empty")
        shutil.rmtree(bank)
    dsmain = bank / "dsmain"
    embeds = bank / "embeds"
    dsmain.mkdir(parents=True)
    embeds.mkdir()

    # --- the acoustic model
    aco_cfg = acoustic.dsconfig
    aco_onnx = acoustic.file("onnx") or acoustic.file("acoustic.onnx")
    if aco_onnx is None:
        raise FileNotFoundError(f"No acoustic ONNX in {acoustic.dir}")
    shutil.copy2(aco_onnx, dsmain / "acoustic.onnx")
    for key, target_name in (("phonemes", "phonemes.json"), ("languages", "languages.json")):
        src = acoustic.dir / str(aco_cfg.get(key) or "")
        if src.is_file():
            shutil.copy2(src, dsmain / target_name)
        else:
            problems.append({"file": str(src), "problem": f"the acoustic export has no {key} file"})
    speakers: list[str] = []
    for emb in acoustic.embeds():
        spk = emb.name[len(acoustic.name) + 1:-4]
        shutil.copy2(emb, embeds / f"{spk}.emb")
        speakers.append(spk)
    main_cfg = dict(aco_cfg)
    main_cfg["acoustic"] = "dsmain/acoustic.onnx"
    main_cfg["phonemes"] = "dsmain/phonemes.json"
    main_cfg["languages"] = "dsmain/languages.json"
    main_cfg["singer_type"] = "diffsinger"
    if speakers:
        main_cfg["speakers"] = [f"embeds/{s}" for s in speakers]
    else:
        main_cfg.pop("speakers", None)
    phonemes_list: list[str] = []
    try:
        phonemes_list = list(json.loads((dsmain / "phonemes.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    languages_map: dict[str, int] = {}
    try:
        languages_map = dict(json.loads((dsmain / "languages.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    multilingual = len(languages_map) > 1

    # --- the vocoder
    if spec.vocoder:
        voc_dir = Path(spec.vocoder)
        onnx = next(iter(voc_dir.glob("*.onnx")), None)
        if onnx is None:
            problems.append({"file": str(voc_dir), "problem": "no ONNX file in the vocoder folder"})
        else:
            dsvoc = bank / "dsvocoder"
            dsvoc.mkdir()
            shutil.copy2(onnx, dsvoc / onnx.name)
            voc_yaml = voc_dir / "vocoder.yaml"
            if voc_yaml.is_file():
                shutil.copy2(voc_yaml, dsvoc / "vocoder.yaml")
            else:
                problems.append({"file": str(voc_yaml), "problem": "the vocoder folder has no vocoder.yaml"})
            main_cfg["vocoder"] = onnx.stem
    else:
        main_cfg["vocoder"] = main_cfg.get("vocoder") or STANDARD_VOCODER
    (bank / "dsconfig.yaml").write_text(yaml.safe_dump(main_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # --- the variance models: dur, pitch, variance from one or several exports
    models = ["acoustic"]
    linguistic_done = False
    dsdur = bank / "dsdur"
    for var in variances:
        var_cfg = var.dsconfig
        ling = var.file("linguistic.onnx")
        if ling is not None and not linguistic_done:
            shutil.copy2(ling, dsmain / "linguistic.onnx")
            linguistic_done = True
        var_speakers = [e.name[len(var.name) + 1:-4] for e in var.embeds()]
        if var_speakers and sorted(var_speakers) != sorted(speakers) and speakers:
            problems.append({"file": str(var.dir), "problem": f"speakers differ from the acoustic model: {var_speakers} vs {speakers}"})
        # the phoneme sets must match
        try:
            var_phonemes = list(json.loads((var.dir / str(var_cfg.get("phonemes"))).read_text(encoding="utf-8")))
            if phonemes_list and var_phonemes != phonemes_list:
                problems.append({"file": str(var.dir), "problem": "the phoneme set differs from the acoustic model's (train both with the same dictionaries)"})
        except (OSError, ValueError, TypeError):
            pass
        base = {k: v for k, v in var_cfg.items() if k not in ("phonemes", "languages", "linguistic", "dur", "pitch", "variance", "speakers")}
        for key, folder_name, flag in (("dur", "dsdur", "predict_dur"), ("pitch", "dspitch", "predict_pitch"), ("variance", "dsvariance", None)):
            onnx = var.file(f"{key}.onnx")
            if onnx is None:
                continue
            folder = bank / folder_name
            if folder.exists():
                problems.append({"file": str(var.dir), "problem": f"{folder_name} comes from two exports; the first one is kept"})
                continue
            folder.mkdir()
            shutil.copy2(onnx, folder / f"{key}.onnx")
            sub = dict(base)
            if folder_name == "dsdur":
                for src_key, name in (("phonemes", "phonemes.json"), ("languages", "languages.json")):
                    src = var.dir / str(var_cfg.get(src_key) or "")
                    if src.is_file():
                        shutil.copy2(src, folder / name)
                sub["phonemes"] = "phonemes.json"
                sub["languages"] = "languages.json"
            else:
                sub["phonemes"] = "../dsdur/phonemes.json" if dsdur.exists() else "../dsmain/phonemes.json"
                sub["languages"] = "../dsdur/languages.json" if dsdur.exists() else "../dsmain/languages.json"
            sub["linguistic"] = "../dsmain/linguistic.onnx"
            sub[key] = f"{key}.onnx"
            if flag:
                sub[flag] = True
            if key == "pitch":
                sub.setdefault("use_expr", True)
            if var_speakers:
                sub_embeds = folder / "embeds"
                sub_embeds.mkdir()
                for emb in var.embeds():
                    shutil.copy2(emb, sub_embeds / f"{emb.name[len(var.name) + 1:-4]}.emb")
                sub["speakers"] = [f"embeds/{s}" for s in var_speakers]
            (folder / "dsconfig.yaml").write_text(yaml.safe_dump(sub, allow_unicode=True, sort_keys=False), encoding="utf-8")
            models.append(key)
    if variances and not linguistic_done:
        problems.append({"file": str(variances[0].dir), "problem": "no linguistic.onnx in the variance export"})

    # --- character files
    (bank / "character.txt").write_text(f"name={spec.name}\n", encoding="utf-8")
    character: dict[str, Any] = {
        "singer_type": "diffsinger",
        "default_phonemizer": PHONEMIZER,
        "text_file_encoding": "utf-8",
        "image": None, "portrait": None, "portrait_opacity": 0.45,
    }
    if speakers:
        character["subbanks"] = [{"color": f"{i:02}: {s}", "suffix": s} for i, s in enumerate(speakers, start=1)]
    (bank / "character.yaml").write_text(yaml.safe_dump(character, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # --- dictionaries (dsdict) at the root and next to every model that phonemizes (dsdur, dspitch, dsvariance)
    dictionaries = acoustic.dictionaries()
    if dictionaries:
        used: set[str] = set()
        for lang, path in dictionaries.items():
            for _, phones in _read_dictionary(path):
                used.update(p if "/" in p or p in ("AP", "SP") else f"{lang}/{p}" for p in phones)
                used.update(phones)
        extra = [p for p in phonemes_list if p not in ("AP", "SP") and p not in used]
        made = write_dsdicts(bank, dictionaries, phonemes_list, extra, spec, multilingual)
        for sub_name in ("dsdur", "dspitch", "dsvariance"):
            sub = bank / sub_name
            if sub.exists():
                for f in made:
                    shutil.copy2(f, sub / f.name)
    else:
        problems.append({"file": str(acoustic.dir), "problem": "no dictionary-<lang>.txt next to the export: no dsdict.yaml made"})

    # --- readme
    lines = [f"{spec.name}", "", "A DiffSinger voicebank for OpenUtau, made with SingForge / mVocalToolkit.", ""]
    lines.append("Models: " + ", ".join(models))
    if languages_map:
        lines.append("Languages: " + ", ".join(sorted(languages_map)))
    if speakers:
        lines.append("Speakers: " + ", ".join(speakers))
    lines.append("Vocoder: " + str(main_cfg.get("vocoder")))
    if spec.licenses:
        lines += ["", "Licenses:"] + [f"- {x}" for x in spec.licenses]
    if spec.credits:
        lines += ["", "Credits:"] + [f"- {x}" for x in spec.credits]
    if spec.notes:
        lines += ["", spec.notes]
    (bank / "readme.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    problems += lint(bank)
    files = sorted(str(p.relative_to(bank)).replace("\\", "/") for p in bank.rglob("*") if p.is_file())
    return VoicebankResult(dir=str(bank), files=files, problems=problems, speakers=speakers,
                           languages=sorted(languages_map), models=models)


def lint(bank: Path) -> list[dict[str, str]]:
    """What OpenUtau would stumble over: missing files, dangling references, mismatched settings."""
    problems: list[dict[str, str]] = []

    def need(path: Path, what: str) -> bool:
        if not path.is_file():
            problems.append({"file": str(path), "problem": f"missing: {what}"})
            return False
        return True

    need(bank / "character.txt", "character.txt (OpenUtau lists singers by it)")
    if need(bank / "character.yaml", "character.yaml"):
        try:
            ch = yaml.safe_load((bank / "character.yaml").read_text(encoding="utf-8")) or {}
            if ch.get("singer_type") != "diffsinger":
                problems.append({"file": "character.yaml", "problem": "singer_type must be diffsinger"})
            for sb in ch.get("subbanks") or []:
                suffix = sb.get("suffix")
                if suffix and not (bank / "embeds" / f"{suffix}.emb").is_file():
                    problems.append({"file": "character.yaml", "problem": f"subbank {suffix} has no embeds/{suffix}.emb"})
        except Exception as e:  # noqa: BLE001
            problems.append({"file": "character.yaml", "problem": f"not readable: {e}"})
    configs = [bank / "dsconfig.yaml"] + [bank / d / "dsconfig.yaml" for d in ("dsdur", "dspitch", "dsvariance") if (bank / d).exists()]
    use_lang: set[bool] = set()
    for cfg_path in configs:
        if not need(cfg_path, "dsconfig.yaml"):
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa: BLE001
            problems.append({"file": str(cfg_path), "problem": f"not readable: {e}"})
            continue
        folder = cfg_path.parent
        for key in ("acoustic", "phonemes", "languages", "linguistic", "dur", "pitch", "variance"):
            value = cfg.get(key)
            if value and not (folder / str(value)).is_file():
                problems.append({"file": str(cfg_path), "problem": f"{key}: {value} not found"})
        for spk in cfg.get("speakers") or []:
            if not (folder / f"{spk}.emb").is_file():
                problems.append({"file": str(cfg_path), "problem": f"speaker embedding {spk}.emb not found"})
        if "use_lang_id" in cfg:
            use_lang.add(bool(cfg["use_lang_id"]))
        if cfg_path.parent == bank and not cfg.get("vocoder"):
            problems.append({"file": str(cfg_path), "problem": "no vocoder named"})
    if len(use_lang) > 1:
        problems.append({"file": "dsconfig.yaml", "problem": "use_lang_id differs between the models: OpenUtau fails with a 'languages' input error"})
    for name in ("dsdict.yaml",):
        if not (bank / name).is_file() and not list(bank.glob("dsdict-*.yaml")):
            problems.append({"file": name, "problem": "no dsdict.yaml: the phonemizer has no dictionary"})
    for p in bank.rglob("*.yaml"):
        try:
            p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append({"file": str(p), "problem": "not UTF-8"})
    return problems

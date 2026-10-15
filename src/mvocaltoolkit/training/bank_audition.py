"""Bounded, sequential singer/language auditions using actual bank vocabularies."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re

from pydantic import BaseModel, Field
import yaml

from ..jobs import JobCancelled
from ..engines._runtime.mvt_mel import compatibility_errors
from .render_resources import validate_render_resources
from .test_phrases import PhraseOptions, PhraseRequest, plan_phrase, write_phrase


class Selection(BaseModel):
    model_config = {"extra": "forbid"}
    speaker: str = ""
    language: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")


class AuditionRequest(BaseModel):
    model_config = {"extra": "forbid"}
    bank: str = Field(min_length=1)
    vocoder: str = ""
    selections: list[Selection] = Field(min_length=1, max_length=16)
    phrase: PhraseOptions = Field(default_factory=PhraseOptions)
    steps: int = Field(20, strict=True, ge=1, le=1000)
    cpu_threads: int = Field(2, strict=True, ge=1, le=32)
    cache_models: bool = Field(True, strict=True)


def _local(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Bank metadata paths must be nonempty strings")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing bank file or path outside bank: {name}")
    return path


def _vocabulary(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        ids = list(data.values())
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid vocabulary IDs: {path}")
        data = sorted(data, key=data.get)
    if not isinstance(data, list) or not data or any(not isinstance(s, str) or not s or any(c.isspace() for c in s) for s in data) or len(set(data)) != len(data):
        raise ValueError(f"Invalid vocabulary: {path}")
    return data


def bank_inventory(bank: str) -> dict:
    if not bank.strip(): raise ValueError("Select an exported bank")
    root = Path(bank).expanduser().resolve()
    cfg = yaml.safe_load(_local(root, "dsconfig.yaml").read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict): raise ValueError("dsconfig.yaml must contain an object")
    phones = _vocabulary(_local(root, cfg.get("phonemes", "phonemes.json")))
    refs = cfg.get("speakers") or []
    if not isinstance(refs, list) or any(not isinstance(s, str) or not s for s in refs):
        raise ValueError("Bank speakers must be a list of embedding references")
    speakers = []
    for ref in refs:
        _local(root, ref + ".emb")
        speakers.append(Path(ref).name)
    if len(set(speakers)) != len(speakers): raise ValueError("Speaker names must be distinct for audition selection")
    languages = []
    if cfg.get("use_lang_id") or cfg.get("languages"):
        languages = _vocabulary(_local(root, cfg.get("languages", "languages.json")))
        languages = [s for s in languages if s not in {"SP", "AP", "default", ""} and not s.startswith("<")]
    if not languages:
        languages = sorted({p.split("/", 1)[0] for p in phones if "/" in p and not p.startswith("<")})
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", s) for s in languages):
        raise ValueError("Bank language names cannot be used as audition language codes")
    return dict(bank=str(root), speakers=speakers or [""], languages=languages, requires_language=not languages)


def plan_audition(req: AuditionRequest) -> dict:
    bank = bank_inventory(req.bank)
    root = Path(bank["bank"])
    cfg = yaml.safe_load(_local(root, "dsconfig.yaml").read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict): raise ValueError("dsconfig.yaml must contain an object")
    _local(root, cfg.get("acoustic"))
    vocoder = Path(req.vocoder).expanduser().resolve() if req.vocoder else root / "dsvocoder"
    vcfg = yaml.safe_load(_local(vocoder, "vocoder.yaml").read_text(encoding="utf-8-sig"))
    if not isinstance(vcfg, dict): raise ValueError("vocoder.yaml must contain an object")
    _local(vocoder, vcfg.get("model", "vocoder.onnx"))
    issues = compatibility_errors(cfg, vcfg)
    if issues: raise ValueError("Incompatible vocoder: " + "; ".join(issues))
    seen, rows = set(), []
    for selection in req.selections:
        pair = (selection.speaker, selection.language)
        if pair in seen: raise ValueError("Duplicate singer/language selection")
        seen.add(pair)
        if selection.speaker not in bank["speakers"]: raise ValueError(f"Unknown bank singer: {selection.speaker}")
        if bank["languages"] and selection.language not in bank["languages"]: raise ValueError(f"Unknown bank language: {selection.language}")
        phrase = plan_phrase(PhraseRequest(**req.phrase.model_dump(), model_dir=bank["bank"], language=selection.language))
        rows.append(dict(**selection.model_dump(), phrase=phrase))
    validate_render_resources(req.model_dump())
    return dict(bank=bank, rows=rows, count=len(rows), audio_seconds=sum(row["phrase"]["seconds"] for row in rows))


async def run_audition(req: AuditionRequest, plan: dict, out: Path, job, call) -> dict:
    """Caller owns the shared inference slot. Each render owns its graph cache."""
    job.check_cancelled()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", requested=len(plan["rows"]), completed=0, rows=[], bank=plan["bank"]["bank"],
                  settings=req.model_dump(), files=[], report=str(out / "audition-report.json"))
    def save():
        temp = out / "audition-report.tmp"
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        temp.replace(out / "audition-report.json")
        job.info.result = report
    save()
    try:
        for index, row in enumerate(plan["rows"]):
            job.check_cancelled()
            folder = out / f"{index + 1:02d}"
            phrase = write_phrase(row["phrase"], folder / "phrase")
            job.progress(index / len(plan["rows"]), stage="audition", message=f"{index + 1}/{len(plan['rows'])}: {row['speaker'] or 'default'} / {row['language']}")
            params = dict(bank=plan["bank"]["bank"], ds=phrase["path"], out_dir=str(folder / "audio"),
                          language=row["language"], speaker=row["speaker"] or None, steps=req.steps,
                          cpu_threads=req.cpu_threads, cache_models=req.cache_models)
            if req.vocoder: params["vocoder"] = str(Path(req.vocoder).expanduser().resolve())
            result = await call(params)
            report["rows"].append(dict(speaker=row["speaker"], language=row["language"], phrase=phrase["path"],
                                       syllables=phrase["syllables"], result=result))
            report["completed"] += 1
            report["files"].extend(result.get("files", []))
            save()
        job.check_cancelled()
        report["status"] = "done"
    except (JobCancelled, asyncio.CancelledError):
        report["status"] = "cancelled"
        raise
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()
    return report

"""Pronunciations of words an aligner's dictionary lacks: the user's own words and G2P guesses.

Own words are kept per model in <home>/dictionaries/<model id>.txt (the dictionary format: word, tab, phonemes) and
are used by every request with that model. A model without a G2P model of its own (HubertFA, SOFA models without
g2p/) gets guesses from the G2P of another installed SOFA model of the same language, when every guessed phoneme is
one the model knows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dictionary import Dictionary

if TYPE_CHECKING:
    from ..models.store import InstalledModel
    from ..settings import Home
    from ..toolkit import Toolkit


def _safe(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name).strip("._") or "model"


def user_words_path(home: Home, model_id: str) -> Path:
    return home.dir("dictionaries") / f"{_safe(model_id)}.txt"


def load_user_words(home: Home, model_id: str) -> dict[str, list[str]]:
    path = user_words_path(home, model_id)
    if not path.exists():
        return {}
    words: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        word, _, phones = line.partition("\t") if "\t" in line else line.partition(" ")
        word, phonemes = word.strip(), phones.split()
        if word and phonemes:
            words[word] = phonemes
    return words


def save_user_words(home: Home, model_id: str, words: dict[str, list[str]]) -> dict[str, list[str]]:
    clean = {w.strip(): [p for p in ph if p] for w, ph in words.items() if w.strip() and any(ph)}
    path = user_words_path(home, model_id)
    if clean:
        lines = [f"{w}\t{' '.join(ph)}" for w, ph in sorted(clean.items())]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    elif path.exists():
        path.unlink()
    return clean


def plain_tokens(text: str) -> list[str]:
    """Words of a plain text as a model's own G2P (TIFA) looks them up: lower case, without punctuation."""
    return [t.strip("-'") for t in re.findall(r"[\w'-]+", text.lower()) if t.strip("-'")]


def _tifa_dictionary(model: InstalledModel, language: str | None) -> Path | None:
    """The dictionary of a TIFA model's G2P (config.yaml, inference.g2p.converters) for the language."""
    import yaml  # noqa: PLC0415

    try:
        config = yaml.safe_load((Path(model.path) / model.layout.get("config", "config.yaml")).read_text(encoding="utf-8"))
        converters = config["inference"]["g2p"]["converters"]
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return None
    found = [(c.get("language"), (c.get("kwargs") or {}).get("dict_path")) for c in converters
             if isinstance(c, dict) and c.get("id") == "dictionary"]
    found = [(lang, p) for lang, p in found if isinstance(p, str)]
    pick = next((p for lang, p in found if language and lang and _base(lang) == _base(language)), None)
    if pick is None and len(found) == 1:
        pick = found[0][1]
    if pick is None:
        return None
    # "@" is the model folder
    return Path(model.path) / pick[1:].lstrip("/\\") if pick.startswith("@") else Path(pick)


def dictionary_path(model: InstalledModel, language: str | None) -> Path | None:
    if model.engine == "tifa":
        return _tifa_dictionary(model, language)
    if model.engine == "hubertfa":
        dictionaries: dict[str, str] = model.layout.get("dictionaries") or {}
        if not dictionaries:
            return None
        name = dictionaries.get(language or "")
        if name is None and language:
            base = language.lower().split("_")[0].split("-")[0]
            name = next((v for k, v in dictionaries.items() if k.lower().split("_")[0] == base), None)
        if name is None and len(dictionaries) == 1:
            name = next(iter(dictionaries.values()))
        return Path(model.path) / name if name else None
    return model.file("dictionary")


def has_own_g2p(model: InstalledModel) -> bool:
    return model.engine == "sofa" and bool(model.layout.get("g2p"))


def _base(lang: str) -> str:
    return lang.lower().replace("-", "_").split("_")[0]


def g2p_donors(tk: Toolkit, model: InstalledModel, language: str | None) -> list[InstalledModel]:
    """Installed SOFA models with a G2P model for the language (the model itself first)."""
    langs = {_base(language)} if language else {_base(lang) for lang in model.languages if lang != "*"}
    donors = [m for m in tk.models.installed().values()
              if has_own_g2p(m) and m.id != model.id and any(_base(lang) in langs for lang in m.languages)]
    return ([model] if has_own_g2p(model) else []) + donors


async def guess(tk: Toolkit, model: InstalledModel, language: str | None, words: list[str],
                phoneme_set: set[str] | None = None) -> dict[str, list[str]]:
    """G2P guesses for [words]: from the model's own G2P, else from donors whose phonemes the model knows."""
    left = [w for w in dict.fromkeys(words) if w and w not in ("SP", "AP")]
    found: dict[str, list[str]] = {}
    for donor in g2p_donors(tk, model, language):
        if not left:
            break
        try:
            got: dict[str, Any] = await tk.engines.call(
                "sofa", "g2p", {"model": {"path": donor.path, "layout": donor.layout}, "words": left}
            )
        except Exception:  # noqa: BLE001
            continue
        for word in list(left):
            phonemes = got.get(word)
            if not phonemes:
                continue
            if donor.id != model.id and phoneme_set and not set(phonemes) <= phoneme_set:
                continue
            found[word] = list(phonemes)
            left.remove(word)
    return found


def model_phonemes(model: InstalledModel, language: str | None) -> set[str] | None:
    path = dictionary_path(model, language)
    if path is None or not path.exists():
        return None
    return Dictionary.load(path).phonemes()


async def fill_unknown(tk: Toolkit, model: InstalledModel, language: str | None, tokens: list[list[str]],
                       extra: dict[str, list[str]]) -> dict[str, list[str]]:
    """Own words plus G2P guesses for the tokens the model's dictionary lacks; returns the words to add."""
    words = {**load_user_words(tk.home, model.id), **extra}
    path = dictionary_path(model, language)
    if path is None or not path.exists():
        return words
    dictionary = Dictionary.load(path).with_extra(words)
    missing = dictionary.missing([t for toks in tokens for t in toks])
    # a model with its own G2P guesses in the engine; the others borrow one
    if missing and not has_own_g2p(model):
        words.update(await guess(tk, model, language, missing, dictionary.phonemes()))
    return words

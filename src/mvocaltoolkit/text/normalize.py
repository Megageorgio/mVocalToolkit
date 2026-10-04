"""Text normalization before G2P / alignment.

Pure-Python frontends live here. Frontends that need heavy libraries (Japanese, Chinese, Korean)
run in the "textfront" engine (see engines/textfront).

A frontend turns raw lyrics into a list of tokens that the aligner's dictionary understands:
words for dictionary-based languages (en, fr, ru...), syllables or phonemes for ja/zh/ko.
"""

from __future__ import annotations

import re
import unicodedata

# frontends that are implemented in the textfront engine
ENGINE_FRONTENDS = {"ja", "zh", "ko", "yue"}

_PUNCT = re.compile(r"[^\w\s'’\-]", re.UNICODE)
_SPACES = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")

FR_CONTRACTIONS = ["m'", "n'", "l'", "j'", "c'", "ç'", "s'", "t'", "d'", "qu'"]


def base_cleanup(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    # bracketed annotations: [chorus], (x2)
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    text = _PUNCT.sub(" ", text)
    text = text.replace("_", " ")
    return _SPACES.sub(" ", text).strip()


def normalize_en(text: str) -> list[str]:
    text = base_cleanup(text).lower()
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text).replace("-", " ")
    # drop quotes that aren't inside words
    tokens = [t.strip("'") for t in text.split()]
    return [t for t in tokens if t]


def normalize_fr(text: str) -> list[str]:
    text = base_cleanup(text).lower().replace("-", " ")
    for contraction in FR_CONTRACTIONS:
        text = text.replace(contraction, contraction + " ")
    return [t for t in text.split() if t and t != "'"]


def normalize_ru(text: str) -> list[str]:
    text = base_cleanup(text).lower().replace("-", " ").replace("'", "")
    return [t for t in text.split() if t]


def normalize_generic(text: str) -> list[str]:
    text = base_cleanup(text).lower().replace("-", " ")
    return [t.strip("'") for t in text.split() if t.strip("'")]


PYTHON_FRONTENDS = {
    "en": normalize_en,
    "fr": normalize_fr,
    "ru": normalize_ru,
}


def frontend_name(language: str | None) -> str:
    if not language:
        return "generic"
    return language.lower().replace("-", "_").split("_")[0]


def normalize(text: str, language: str | None) -> list[str]:
    """Normalizes with a pure-Python frontend (raises for frontends that need the engine)."""
    name = frontend_name(language)
    if name in ENGINE_FRONTENDS:
        raise ValueError(f"Frontend {name} runs in the textfront engine")
    return PYTHON_FRONTENDS.get(name, normalize_generic)(text)


def has_digits(text: str) -> bool:
    return bool(_DIGITS.search(text))

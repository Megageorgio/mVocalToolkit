"""Text frontends for languages that need heavy libraries.

Output conventions follow LabelMakr so that the usual SOFA models/dictionaries work:
    ja: OpenJTalk phonemes, devoiced vowels lower-cased (A -> a), N kept, pau -> SP
    zh: pinyin syllables without tones
    ko: pronunciation-normalized hangul (g2pk), split into words
"""

from __future__ import annotations

import re

import mvt_engine as rt

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)
_g2pk = None


def _fix(text: str) -> str:
    try:
        from ftfy import fix_text  # noqa: PLC0415

        return fix_text(text)
    except ImportError:
        return text


def _clean(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）", " ", _fix(text))
    return re.sub(r"\s+", " ", text).strip()


def ja(text: str) -> list[str]:
    import pyopenjtalk  # noqa: PLC0415

    text = _clean(text).replace(" ", "、")
    phones = pyopenjtalk.g2p(text).split()
    result = []
    for ph in phones:
        if ph in ("pau", "sil"):
            if result and result[-1] != "SP":
                result.append("SP")
            continue
        if ph in ("A", "I", "U", "E", "O"):
            ph = ph.lower()
        result.append(ph)
    while result and result[-1] == "SP":
        result.pop()
    return result


def zh(text: str) -> list[str]:
    from pypinyin import lazy_pinyin  # noqa: PLC0415

    text = _PUNCT.sub(" ", _clean(text))
    tokens: list[str] = []
    # hanzi -> pinyin; latin text (already pinyin, or English words) is kept as words
    for chunk in re.findall(r"[㐀-鿿豈-﫿]+|[^\s㐀-鿿豈-﫿]+", text):
        if re.match(r"[㐀-鿿豈-﫿]", chunk):
            tokens += [s.strip().lower() for s in lazy_pinyin(chunk, errors="ignore")]
        else:
            tokens.append(chunk.lower())
    return [t for t in tokens if t]


CMUDICT_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/cmudict.zip"


def _ensure_cmudict() -> None:
    """g2pk2 needs NLTK's cmudict (English words inside Korean text). NLTK's own downloader refuses to work
    through proxies, so the corpus is fetched directly into the toolkit cache."""
    import os  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import nltk  # noqa: PLC0415

    root = Path(os.environ.get("MVT_CACHE", ".")) / "nltk"
    if str(root) not in nltk.data.path:
        nltk.data.path.insert(0, str(root))
    try:
        nltk.data.find("corpora/cmudict.zip")
        return
    except LookupError:
        pass
    target = root / "corpora" / "cmudict.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    rt.progress(message="Downloading cmudict for Korean G2P")
    with urllib.request.urlopen(CMUDICT_URL, timeout=120) as response:
        data = response.read()
    tmp = target.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(target)


def ko(text: str) -> list[str]:
    global _g2pk
    if _g2pk is None:
        _ensure_cmudict()
        from g2pk2 import G2p  # noqa: PLC0415

        _g2pk = G2p()
    text = _PUNCT.sub(" ", _clean(text))
    return [w for w in _g2pk(text).split() if w]


FRONTENDS = {"ja": ja, "zh": zh, "ko": ko}


@rt.method()
def normalize(language: str, texts: list[str]) -> list[list[str]]:
    func = FRONTENDS.get(language)
    if func is None:
        raise ValueError(f"No text frontend for {language}")
    return [func(text) if text and text.strip() else [] for text in texts]


if __name__ == "__main__":
    rt.run()

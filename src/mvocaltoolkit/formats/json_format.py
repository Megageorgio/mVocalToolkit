from __future__ import annotations

from ..labels import Label


def dumps(label: Label, tier: str = "phones") -> str:
    return label.model_dump_json(indent=2, exclude_none=True)


def loads(text: str, tier: str = "phones") -> Label:
    return Label.model_validate_json(text)

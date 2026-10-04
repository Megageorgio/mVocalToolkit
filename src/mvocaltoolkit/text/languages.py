"""Language codes and names shown by GUIs (English name, native name)."""

from __future__ import annotations

LANGUAGE_NAMES: dict[str, tuple[str, str]] = {
    "en": ("English", "English"),
    "ru": ("Russian", "Русский"),
    "uk": ("Ukrainian", "Українська"),
    "be": ("Belarusian", "Беларуская"),
    "ja": ("Japanese", "日本語"),
    "zh": ("Chinese", "中文"),
    "yue": ("Cantonese", "粵語"),
    "ko": ("Korean", "한국어"),
    "fr": ("French", "Français"),
    "de": ("German", "Deutsch"),
    "es": ("Spanish", "Español"),
    "pt": ("Portuguese", "Português"),
    "it": ("Italian", "Italiano"),
    "pl": ("Polish", "Polski"),
    "cs": ("Czech", "Čeština"),
    "nl": ("Dutch", "Nederlands"),
    "tr": ("Turkish", "Türkçe"),
    "vi": ("Vietnamese", "Tiếng Việt"),
    "th": ("Thai", "ไทย"),
    "id": ("Indonesian", "Bahasa Indonesia"),
    "tl": ("Tagalog", "Tagalog"),
    "ar": ("Arabic", "العربية"),
    "fi": ("Finnish", "Suomi"),
    "sv": ("Swedish", "Svenska"),
    "*": ("Any language", "Any language"),
}


def language_info(code: str) -> dict[str, str]:
    base = code.lower().replace("-", "_")
    name, native = LANGUAGE_NAMES.get(base) or LANGUAGE_NAMES.get(base.split("_")[0]) or (code, code)
    return {"code": code, "name": name, "native_name": native}

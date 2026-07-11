"""Language dispatcher for growth-layer text constants.

言語設定に基づいて適切なテキスト定数モジュールを返すディスパッチャ.

使用方法:
    from growth import texts
    texts.set_language("en")  # ゲーム開始時に1回呼ぶ
    t = texts.get()
    block = t.GROUNDING_ATTENTION

set_language() はモジュールレベルの状態を更新するため, プロセス内で一度だけ呼べばよい.
デフォルトは "ja" (日本語).
"""

from __future__ import annotations

from typing import Any

_lang: str = "ja"
_SUPPORTED = ("ja", "en")


def set_language(lang: str) -> None:
    """Set the active language for all growth-layer text constants.

    growth層のテキスト定数が参照する言語を設定する.

    Args:
        lang (str): Language code — "ja" or "en" / 言語コード ("ja" または "en")
    """
    global _lang  # noqa: PLW0603
    if lang not in _SUPPORTED:
        msg = f"Unsupported language: {lang!r}. Choose one of {_SUPPORTED}."
        raise ValueError(msg)
    _lang = lang


def get_language() -> str:
    """Return the currently active language code.

    現在有効な言語コードを返す.

    Returns:
        str: Language code / 言語コード
    """
    return _lang


def get() -> Any:  # noqa: ANN401
    """Return the text constants module for the active language.

    有効な言語のテキスト定数モジュールを返す.

    Returns:
        Module with text constants (texts_ja or texts_en) /
            テキスト定数モジュール (texts_ja または texts_en)
    """
    if _lang == "en":
        from growth import texts_en  # noqa: PLC0415

        return texts_en
    from growth import texts_ja  # noqa: PLC0415

    return texts_ja

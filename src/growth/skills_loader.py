"""Module that loads strategy skills (markdown) for the growth layer.

growth層の戦略skill(markdown)をロードするモジュール.

skills/common.md は全役職共通の規範. skills/{n}players/{role}.md は役職・ゲームサイズ別の
戦略skill(第2スライスで追加). ファイル内容はプロセス内でキャッシュする.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from growth import texts

_SKILLS_DIR = Path(__file__).parent / "skills"


@cache
def _read_skill(relative_path: str) -> str:
    """Read a skill markdown file, returning empty string if absent.

    skillのmarkdownファイルを読み込む. 存在しなければ空文字列を返す.

    Args:
        relative_path (str): Path relative to the skills directory / skillsディレクトリからの相対パス

    Returns:
        str: File content, or empty string if the file does not exist /
            ファイル内容. 存在しない場合は空文字列
    """
    path = _SKILLS_DIR / relative_path
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _skill_path(relative_path: str) -> str:
    """Resolve skill path with language prefix.

    言語プレフィックスを付けた skillパスを解決する.
    skills/{lang}/{relative_path} が存在すればそちらを優先し,
    存在しなければ skills/{relative_path} にフォールバックする.

    Args:
        relative_path (str): Path relative to skills directory / skillsディレクトリからの相対パス

    Returns:
        str: Resolved relative path for _read_skill / _read_skill に渡す相対パス
    """
    lang = texts.get_language()
    lang_path = f"{lang}/{relative_path}"
    if (_SKILLS_DIR / lang_path).is_file():
        return lang_path
    return relative_path


def load_common_norms() -> str:
    """Load the common norms shared by all roles.

    全役職共通の規範を読み込む. 言語別ファイルが存在する場合はそちらを優先する.

    Returns:
        str: Content of skills/{lang}/common.md or skills/common.md /
            言語別またはデフォルトの common.md の内容
    """
    return _read_skill(_skill_path("common.md"))


def load_role_skill(
    role_value: str,
    player_count: int | None,
    track: str = "turn",
) -> str:
    """Load the role- and game-size-specific strategy skill.

    役職とゲームサイズに応じた戦略skillを読み込む(第2スライスで内容を追加).
    言語別ファイルが存在する場合はそちらを優先する.
    track が "freeform" の場合は {track}/{n}players/{role}.md を優先し、
    無ければ {n}players/{role}.md にフォールバックする.

    Args:
        role_value (str): Role value such as "SEER" / 役職の値 (例: "SEER")
        player_count (int | None): Number of players in the game / ゲームの参加人数
        track (str): Game track — "turn" or "freeform" / トラック

    Returns:
        str: Strategy skill content, or empty string if not yet defined /
            戦略skillの内容. 未定義の場合は空文字列
    """
    if player_count is None:
        return ""
    role_file = f"{player_count}players/{role_value.lower()}.md"
    # freeform専用ファイルがあれば優先、なければ通常のファイルを使う
    if track == "freeform":
        freeform_path = f"freeform/{role_file}"
        resolved = _skill_path(freeform_path)
        if (_SKILLS_DIR / resolved).is_file():
            return _read_skill(resolved)
    return _read_skill(_skill_path(role_file))

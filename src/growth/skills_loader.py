"""Module that loads strategy skills (markdown) for the growth layer.

growth層の戦略skill(markdown)をロードするモジュール.

skills/common.md は全役職共通の規範. skills/{n}players/{role}.md は役職・ゲームサイズ別の
戦略skill(第2スライスで追加). ファイル内容はプロセス内でキャッシュする.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

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


def load_common_norms() -> str:
    """Load the common norms shared by all roles.

    全役職共通の規範を読み込む.

    Returns:
        str: Content of skills/common.md / skills/common.md の内容
    """
    return _read_skill("common.md")


def load_role_skill(role_value: str, player_count: int | None) -> str:
    """Load the role- and game-size-specific strategy skill.

    役職とゲームサイズに応じた戦略skillを読み込む(第2スライスで内容を追加).

    Args:
        role_value (str): Role value such as "SEER" / 役職の値 (例: "SEER")
        player_count (int | None): Number of players in the game / ゲームの参加人数

    Returns:
        str: Strategy skill content, or empty string if not yet defined /
            戦略skillの内容. 未定義の場合は空文字列
    """
    if player_count is None:
        return ""
    return _read_skill(f"{player_count}players/{role_value.lower()}.md")

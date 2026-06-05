"""Module for the post-game self-review (truth-grounded improvement loop).

終局後の自己レビュー(真値接地の自己改善ループ)を担うモジュール.

ゲーム終了時に全員の正体が開示されるのを真値として, 自分のテル読み・判断・ペルソナ保持を
批評し, 次戦への教訓を抽出して役職別ファイルへ追記する. 教訓は次ゲームの初期化時に読み込まれ,
振る舞いに還流する. LLM呼び出しはバックグラウンドスレッドから行われる想定で, タイムアウトで保護する.
"""

from __future__ import annotations

import fcntl
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

_LESSONS_DIR = Path(__file__).parent / "lessons"
# 初期化時に読み込む教訓の最大行数
_MAX_LESSONS_LINES = 8
# 1回のレビューで採用する教訓の最大数
_MAX_LESSONS_PER_REVIEW = 3
# レビュー呼び出しのタイムアウト秒数
_REVIEW_TIMEOUT_S = 90.0
# 人狼陣営の役職集合
_WOLF_SIDE = {"WEREWOLF", "POSSESSED"}


def _value(obj: object) -> str:
    """Return the string value of an enum-like object (or its str form).

    列挙体ふうのオブジェクトの文字列値を返す.

    Args:
        obj (object): Role or Status value / 役職またはステータスの値

    Returns:
        str: String value such as "SEER"/"ALIVE" / "SEER"/"ALIVE" などの文字列値
    """
    return str(getattr(obj, "value", obj))


def _compute_outcome(role_value: str, role_map: dict[str, Any], status_map: dict[str, Any]) -> str:
    """Determine whether the agent's side won, from the revealed roles.

    開示された正体から, 自分の陣営が勝ったかを判定する.

    Args:
        role_value (str): The agent's own role value / 自分の役職の値
        role_map (dict[str, Any]): Revealed roles by name / 名前ごとの正体
        status_map (dict[str, Any]): Final alive/dead status by name / 名前ごとの最終生死

    Returns:
        str: "勝利" or "敗北" / 勝敗
    """
    alive_wolves = sum(
        1
        for name, role in role_map.items()
        if _value(role) == "WEREWOLF" and _value(status_map.get(name)) == "ALIVE"
    )
    village_won = alive_wolves == 0
    my_wolf_side = role_value in _WOLF_SIDE
    won = village_won != my_wolf_side
    return "勝利" if won else "敗北"


def build_review_prompt(
    role_value: str,
    agent_name: str,
    role_map: dict[str, Any],
    status_map: dict[str, Any],
    talk_history: list[Any],
) -> str:
    """Build the post-game review prompt grounded in the revealed roles.

    開示された正体に接地した終局レビューのプロンプトを構築する.

    Args:
        role_value (str): The agent's own role value / 自分の役職の値
        agent_name (str): The agent's in-game name / ゲーム内のエージェント名
        role_map (dict[str, Any]): Revealed roles by name / 名前ごとの正体
        status_map (dict[str, Any]): Final alive/dead status by name / 名前ごとの最終生死
        talk_history (list[Any]): Talk history of the game / ゲームのトーク履歴

    Returns:
        str: The review prompt / レビューのプロンプト
    """
    roles_block = "\n".join(f"- {name}: {_value(role)}" for name, role in role_map.items())
    alive = [name for name, status in status_map.items() if _value(status) == "ALIVE"]
    transcript = "\n".join(
        f"{talk.agent}: {talk.text}" for talk in talk_history if talk.text and not talk.skip
    )
    outcome = _compute_outcome(role_value, role_map, status_map)
    return (
        f"あなたは人狼ゲームで「{role_value}」({agent_name})として対局し、ゲームは終了しました。"
        "全員の正体が判明しています。\n\n"
        "## 各プレイヤーの正体(真値)\n"
        f"{roles_block}\n\n"
        "## 最終結果\n"
        f"- 生存者: {', '.join(alive) if alive else 'なし'}\n"
        f"- あなたの陣営: {outcome}\n\n"
        "## 対局中の全発言\n"
        f"{transcript}\n\n"
        "## 振り返り\n"
        f"判明した正体に照らして、あなた({agent_name})自身の振る舞いを批評してください。特に次の観点です。\n"
        "- 他者の感情・態度の「テル」から行った役職の読みは、正体に照らして当たっていたか外れていたか。\n"
        "- 感情に引きずられて誤った判断や発言をした箇所はあったか。\n"
        "- 自分の性格(プロフィール)を保てたか。\n"
        "次の対局で改善するための教訓を、1〜2個、簡潔な箇条書き(各1行)で挙げてください。"
        "教訓の箇条書きのみを出力し、前置き・説明・見出しは書かないこと。"
    )


def _request_review(llm_model: BaseChatModel, prompt: str) -> str | None:
    """Invoke the LLM for the review, bounded by a timeout.

    レビューのためにLLMを呼び出す. タイムアウトで保護する.

    Args:
        llm_model (BaseChatModel): The chat model to invoke / 呼び出すチャットモデル
        prompt (str): The review prompt / レビューのプロンプト

    Returns:
        str | None: The review text, or None on error/timeout /
            レビュー本文. エラーまたはタイムアウト時は None
    """
    result: dict[str, str] = {}

    def _call() -> None:
        try:
            result["text"] = (llm_model | StrOutputParser()).invoke([HumanMessage(content=prompt)])
        except Exception:
            logger.exception("post_game_review のLLM呼び出しに失敗しました")

    inner = threading.Thread(target=_call, daemon=True)
    inner.start()
    inner.join(_REVIEW_TIMEOUT_S)
    return result.get("text")


def _parse_lessons(text: str) -> list[str]:
    """Extract bullet lessons from the review text.

    レビュー本文から箇条書きの教訓を抽出する.

    Args:
        text (str): The raw review text / レビューの生テキスト

    Returns:
        list[str]: Lesson lines (markers stripped) / 教訓の行(記号を除去)
    """
    lessons: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-・*0123456789. ").strip()
        if line:
            lessons.append(line)
    return lessons[:_MAX_LESSONS_PER_REVIEW]


def _append_lessons(role_value: str, player_count: int, lessons: list[str], outcome: str) -> None:
    """Append lessons to the role-specific file (cross-process safe).

    教訓を役職別ファイルへ追記する(プロセス間で安全).

    Args:
        role_value (str): The agent's role value / 自分の役職の値
        player_count (int): Number of players in the game / ゲームの参加人数
        lessons (list[str]): Lesson lines to append / 追記する教訓の行
        outcome (str): Game outcome label / 勝敗ラベル
    """
    if not lessons:
        return
    path = _LESSONS_DIR / f"{player_count}players" / f"{role_value.lower()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    block = "".join(f"- ({stamp}/{outcome}) {lesson}\n" for lesson in lessons)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(block)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def run_post_game_review(
    llm_model: BaseChatModel,
    role_value: str,
    agent_name: str,
    info: Info,
    talk_history: list[Any],
) -> None:
    """Run the post-game review end to end and persist the lessons.

    終局レビューを通しで実行し, 教訓を永続化する. バックグラウンドスレッドから呼ぶ想定.

    Args:
        llm_model (BaseChatModel): The chat model to invoke / 呼び出すチャットモデル
        role_value (str): The agent's role value / 自分の役職の値
        agent_name (str): The agent's in-game name / ゲーム内のエージェント名
        info (Info): Final game info with revealed roles / 正体開示済みの最終ゲーム情報
        talk_history (list[Any]): Talk history of the game / ゲームのトーク履歴
    """
    role_map = info.role_map
    status_map = info.status_map
    if not role_map:
        return
    try:
        prompt = build_review_prompt(role_value, agent_name, role_map, status_map, talk_history)
        text = _request_review(llm_model, prompt)
        if not text:
            return
        outcome = _compute_outcome(role_value, role_map, status_map)
        _append_lessons(role_value, len(role_map), _parse_lessons(text), outcome)
        logger.info("post_game_review を保存しました: role=%s", role_value)
    except Exception:
        logger.exception("post_game_review の実行に失敗しました")


def load_lessons(role_value: str, player_count: int | None) -> str:
    """Load recent lessons learned for the role to inject at game start.

    ゲーム開始時に注入する, その役職の直近の教訓を読み込む.

    Args:
        role_value (str): The agent's role value / 自分の役職の値
        player_count (int | None): Number of players in the game / ゲームの参加人数

    Returns:
        str: Lessons block, or empty string if none / 教訓ブロック. 無ければ空文字列
    """
    if player_count is None:
        return ""
    path = _LESSONS_DIR / f"{player_count}players" / f"{role_value.lower()}.md"
    if not path.is_file():
        return ""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return ""
    recent = lines[-_MAX_LESSONS_LINES:]
    return "## 過去の対局から学んだ教訓(自己改善。次に活かすこと)\n" + "\n".join(recent)

"""Module for the post-game self-review (truth-grounded improvement loop).

終局後の自己レビュー(真値接地の自己改善ループ)を担うモジュール.

ゲーム終了時に全員の正体が開示されるのを真値として, 自分のテル読み・判断・ペルソナ保持を
批評し, 次戦への教訓を抽出して役職別ファイルへ追記する. 教訓は次ゲームの初期化時に読み込まれ,
振る舞いに還流する. LLM呼び出しはバックグラウンドスレッドから行われる想定で, タイムアウトで保護する.
"""

from __future__ import annotations

import fcntl
import json
import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info
    from langchain_core.language_models.chat_models import BaseChatModel

from growth import texts

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
    t = texts.get()
    return t.REVIEW_OUTCOME_WIN if won else t.REVIEW_OUTCOME_LOSE


def build_review_prompt(
    role_value: str,
    agent_name: str,
    role_map: dict[str, Any],
    status_map: dict[str, Any],
    talk_history: list[Any],
    track: str = "turn",
) -> str:
    """Build the post-game review prompt grounded in the revealed roles.

    開示された正体に接地した終局レビューのプロンプトを構築する.

    既存の教訓(その役職/人数/トラックで既に蓄積されたもの)をプロンプトに含め,
    新しい教訓が既存の教訓と論理的に矛盾しないよう自己チェックさせる(ADR-013)。

    Args:
        role_value (str): The agent's own role value / 自分の役職の値
        agent_name (str): The agent's in-game name / ゲーム内のエージェント名
        role_map (dict[str, Any]): Revealed roles by name / 名前ごとの正体
        status_map (dict[str, Any]): Final alive/dead status by name / 名前ごとの最終生死
        talk_history (list[Any]): Talk history of the game / ゲームのトーク履歴
        track (str): Game track — "turn" or "freeform" / トラック

    Returns:
        str: The review prompt / レビューのプロンプト
    """
    t = texts.get()
    roles_block = "\n".join(f"- {name}: {_value(role)}" for name, role in role_map.items())
    alive = [name for name, status in status_map.items() if _value(status) == "ALIVE"]
    alive_str = ", ".join(alive) if alive else t.GROUNDING_ALIVE_NONE
    transcript = "\n".join(
        f"{talk.agent}: {talk.text}" for talk in talk_history if talk.text and not talk.skip
    )
    outcome = _compute_outcome(role_value, role_map, status_map)
    # 役職別の振り返り観点を選択する
    focus_map = {
        "SEER": t.REVIEW_FOCUS_SEER,
        "MEDIUM": t.REVIEW_FOCUS_MEDIUM,
        "BODYGUARD": t.REVIEW_FOCUS_BODYGUARD,
        "WEREWOLF": t.REVIEW_FOCUS_WEREWOLF,
        "POSSESSED": t.REVIEW_FOCUS_POSSESSED,
        "VILLAGER": t.REVIEW_FOCUS_VILLAGER,
    }
    role_focus = focus_map.get(role_value, t.REVIEW_FOCUS_DEFAULT)
    # 既存の教訓を読み込み, 新しい教訓が矛盾しないよう自己チェックさせる(ADR-013)。
    # load_lessons() の見出し行はTALK/WHISPER時の注入向けの文言なので, ここでは
    # レビュー専用の見出し・指示をREVIEW_PROMPT_TEMPLATE側で与え, 見出し行は使わない。
    player_count = len(role_map)
    raw_lessons = load_lessons(role_value, player_count, track=track)
    if raw_lessons:
        existing_lessons = raw_lessons.split("\n", 1)[1] if "\n" in raw_lessons else raw_lessons
    else:
        existing_lessons = t.REVIEW_NO_EXISTING_LESSONS
    return t.REVIEW_PROMPT_TEMPLATE.format(
        role_value=role_value,
        agent_name=agent_name,
        roles_block=roles_block,
        alive=alive_str,
        outcome=outcome,
        transcript=transcript,
        role_focus=role_focus,
        existing_lessons=existing_lessons,
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


_SENSITIVITY_PATTERN = re.compile(r"SENSITIVITY:\s*(HIGH|LOW|OK)", re.IGNORECASE)
_SENSITIVITY_STEP = 0.05
_CALIBRATION_FILE = "_calibration.json"


_HEADING_PATTERN = re.compile(r"^#{1,4}\s|^\*\*.*\*\*$|\*\*\s*$")
_MIN_LESSON_LENGTH = 15


def _parse_lessons(text: str) -> list[str]:
    """Extract bullet lessons from the review text.

    レビュー本文から箇条書きの教訓を抽出する. SENSITIVITY: 行, 見出し行,
    短すぎる行は教訓に含めない.

    Args:
        text (str): The raw review text / レビューの生テキスト

    Returns:
        list[str]: Lesson lines (markers stripped) / 教訓の行(記号を除去)
    """
    lessons: list[str] = []
    for raw in text.splitlines():
        if _SENSITIVITY_PATTERN.search(raw):
            continue
        line = raw.strip().lstrip("-・*0123456789. ").strip()
        if not line or len(line) < _MIN_LESSON_LENGTH:
            continue
        if _HEADING_PATTERN.search(line):
            continue
        lessons.append(line)
    return lessons[:_MAX_LESSONS_PER_REVIEW]


def _parse_sensitivity_judgment(text: str) -> float:
    """Extract the sensitivity calibration signal from the review output.

    レビュー出力から感受性較正の信号を抽出する.

    HIGH → 感受性が高すぎた → 下げる(負の調整)
    LOW  → 感受性が低すぎた → 上げる(正の調整)
    OK   → 変更なし(ゼロ)

    Args:
        text (str): The raw review text / レビューの生テキスト

    Returns:
        float: Adjustment delta / 調整量(0 は変更なし)
    """
    match = _SENSITIVITY_PATTERN.search(text)
    if not match:
        return 0.0
    judgment = match.group(1).upper()
    if judgment == "HIGH":
        return -_SENSITIVITY_STEP
    if judgment == "LOW":
        return _SENSITIVITY_STEP
    return 0.0


def _lessons_base(track: str = "turn") -> Path:
    """Return the base lessons directory, with track-specific subdir if freeform.

    教訓のベースディレクトリを返す. freeformの場合はfreeform専用ディレクトリを優先.
    """
    lang = texts.get_language()
    lang_dir = _LESSONS_DIR / lang
    base = lang_dir if lang_dir.is_dir() else _LESSONS_DIR
    if track == "freeform":
        freeform_dir = base / "freeform"
        if freeform_dir.is_dir():
            return freeform_dir
    return base


def _calibration_path(role_value: str, player_count: int, track: str = "turn") -> Path:
    """Return the path to the calibration JSON for the given role.

    指定された役職の較正JSONのパスを返す.
    """
    base = _lessons_base(track)
    return base / f"{player_count}players" / f"{role_value.lower()}{_CALIBRATION_FILE}"


def _update_calibration(role_value: str, player_count: int, delta: float, track: str = "turn") -> None:
    """Persist a sensitivity adjustment as a running average.

    感受性調整値を移動平均として永続化する.

    Args:
        role_value (str): The agent's role value / 自分の役職の値
        player_count (int): Number of players / 参加人数
        delta (float): Adjustment delta from this game / 今回の調整量
        track (str): Game track — "turn" or "freeform" / トラック
    """
    if delta == 0.0:
        return
    path = _calibration_path(role_value, player_count, track)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, float] = {"sensitivity_adj": 0.0, "samples": 0}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            pass
    n = data.get("samples", 0)
    old_adj = data.get("sensitivity_adj", 0.0)
    new_n = n + 1
    new_adj = (old_adj * n + delta) / new_n
    with path.open("w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump({"sensitivity_adj": round(new_adj, 4), "samples": new_n}, f)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    logger.info("感受性較正を更新: role=%s, delta=%.3f, cumulative=%.4f (n=%d)",
                role_value, delta, new_adj, new_n)


def load_calibration(role_value: str, player_count: int | None, track: str = "turn") -> float:
    """Load the cumulative sensitivity adjustment for the role.

    役職の累積感受性調整値を読み込む.

    Args:
        role_value (str): The agent's role value / 自分の役職の値
        player_count (int | None): Number of players / 参加人数
        track (str): Game track — "turn" or "freeform" / トラック

    Returns:
        float: Cumulative adjustment (0.0 if none) / 累積調整値
    """
    if player_count is None:
        return 0.0
    path = _calibration_path(role_value, player_count, track)
    if not path.is_file():
        return 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("sensitivity_adj", 0.0))
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0


def _append_lessons(role_value: str, player_count: int, lessons: list[str], outcome: str, track: str = "turn") -> None:
    """Append lessons to the role-specific file (cross-process safe).

    教訓を役職別ファイルへ追記する(プロセス間で安全).

    Args:
        role_value (str): The agent's role value / 自分の役職の値
        player_count (int): Number of players in the game / ゲームの参加人数
        lessons (list[str]): Lesson lines to append / 追記する教訓の行
        outcome (str): Game outcome label / 勝敗ラベル
        track (str): Game track — "turn" or "freeform" / トラック
    """
    if not lessons:
        return
    base = _lessons_base(track)
    path = base / f"{player_count}players" / f"{role_value.lower()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    block = "".join(f"- ({stamp}/{outcome}) {lesson}\n" for lesson in lessons)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(block)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# 終局レビューを実行する最小発話数. これ未満の希薄な対局は材料不足で教訓を捏造しやすい.
_MIN_TRANSCRIPT_TALKS = 6


def run_post_game_review(
    llm_model: BaseChatModel,
    role_value: str,
    agent_name: str,
    info: Info,
    talk_history: list[Any],
    track: str = "turn",
) -> None:
    """Run the post-game review end to end and persist the lessons.

    終局レビューを通しで実行し, 教訓を永続化する. バックグラウンドスレッドから呼ぶ想定.

    Args:
        llm_model (BaseChatModel): The chat model to invoke / 呼び出すチャットモデル
        role_value (str): The agent's role value / 自分の役職の値
        agent_name (str): The agent's in-game name / ゲーム内のエージェント名
        info (Info): Final game info with revealed roles / 正体開示済みの最終ゲーム情報
        talk_history (list[Any]): Talk history of the game / ゲームのトーク履歴
        track (str): Game track — "turn" or "freeform" / トラック
    """
    role_map = info.role_map
    status_map = info.status_map
    if not role_map:
        return
    # トランスクリプトが希薄(実発話がごく少数)な対局は, 材料不足で教訓を捏造しやすいので
    # レビューをスキップし, 偽の教訓が将来ゲームへ蓄積されるのを防ぐ.
    substantive = sum(
        1 for talk in talk_history
        if getattr(talk, "text", None) and not getattr(talk, "skip", False)
    )
    if substantive < _MIN_TRANSCRIPT_TALKS:
        logger.info("post_game_review をスキップ: 発話が希薄 (%d件)", substantive)
        return
    try:
        prompt = build_review_prompt(role_value, agent_name, role_map, status_map, talk_history, track)
        text = _request_review(llm_model, prompt)
        if not text:
            return
        outcome = _compute_outcome(role_value, role_map, status_map)
        player_count = len(role_map)
        _append_lessons(role_value, player_count, _parse_lessons(text), outcome, track)
        delta = _parse_sensitivity_judgment(text)
        _update_calibration(role_value, player_count, delta, track)
        logger.info("post_game_review を保存しました: role=%s track=%s", role_value, track)
    except Exception:
        logger.exception("post_game_review の実行に失敗しました")


_OUTCOME_TAG_PATTERN = re.compile(r"^\s*-\s*\([^/]+/([^)]+)\)\s*")
_DEDUP_MIN_OVERLAP = 0.6


def _extract_outcome(line: str) -> str | None:
    """行から勝敗タグを抽出する. 該当なしなら None."""
    m = _OUTCOME_TAG_PATTERN.match(line)
    return m.group(1) if m else None


def _lesson_body(line: str) -> str:
    """行からタグを除いた教訓本文を返す."""
    m = _OUTCOME_TAG_PATTERN.match(line)
    return line[m.end():].strip() if m else line.strip()


def _is_near_duplicate(candidate: str, selected: list[str]) -> bool:
    """既選出の教訓と大きく重複するかを簡易判定する.

    英語は語単位、日本語は文字単位で Jaccard 係数を計算する。
    英語で文字単位だとアルファベット集合が似通いすぎて全て重複判定になるため。
    """
    c_words = candidate.split()
    use_words = len(c_words) > 3
    c_set = set(c_words) if use_words else set(candidate)
    for s in selected:
        if use_words:
            s_set = set(s.split())
        else:
            s_set = set(s)
        overlap = len(c_set & s_set) / max(len(c_set | s_set), 1)
        if overlap > _DEDUP_MIN_OVERLAP:
            return True
    return False


_LESSON_TAG_PATTERN = re.compile(r"^\[SELF\]\s*|^\[STEER\]\s*|^\[PERSONA\]\s*")
_LESSON_TAGS = ("SELF", "STEER", "PERSONA")

# 状況タグ: 教訓の2つ目の括弧 [TAG] で場面を示す
# 例: - (2026-07-07/Victory) [SELF][CO] Day 0の先制COが正解だった → ...
# パターンは (date/result) [ROLE_TAG][SITUATION_TAG] の形を想定
_SITUATION_TAG_PATTERN = re.compile(r"^-?\s*\([^)]+\)\s*\[[A-Z_]+\]\s*\[([A-Z_]+)\]")
_SITUATION_TAGS = (
    "CO",          # 占い師CO・対抗COの場面
    "COUNTER",     # 対抗COへの対処
    "VOTE",        # 投票の判断
    "WHISPER",     # 囁きフェーズ・人狼同士の連携
    "ATTACK",      # 襲撃先の選定
    "GUARD",       # 護衛先の選定
    "DIVINE",      # 占い先の選定
    "MEDIUM",      # 霊媒結果の公表・活用
    "ENDGAME",     # 終盤(残り少人数・パリティ接近)
    "DEFENSE",     # 自分が疑われた時の防御
    "GENERAL",     # 汎用(特定の場面に限定しない)
)


def _strip_tag(text: str) -> str:
    """行頭の [SELF]/[STEER]/[PERSONA] タグを取り除いた本文を返す.

    入力は _lesson_body() 通過後(勝敗タグ除去済み)の文字列を想定.
    """
    return _LESSON_TAG_PATTERN.sub("", text).strip()


def _extract_tag(text: str) -> str | None:
    """行頭のタグ名(SELF/STEER/PERSONA)を返す. 無ければ None.

    入力は _lesson_body() 通過後(勝敗タグ除去済み)の文字列を想定.
    """
    m = _LESSON_TAG_PATTERN.match(text)
    if not m:
        return None
    matched = m.group(0).strip()
    for tag in _LESSON_TAGS:
        if matched.startswith(f"[{tag}]"):
            return tag
    return None


def _extract_situation(line: str) -> str:
    """Extract the situation tag from a lesson line.

    教訓行から状況タグを抽出する.

    Args:
        line (str): A lesson line / 教訓行

    Returns:
        str: Situation tag (e.g. "CO"), or "GENERAL" if none / 状況タグ
    """
    m = _SITUATION_TAG_PATTERN.match(line)
    return m.group(1) if m else "GENERAL"


def load_lessons(
    role_value: str,
    player_count: int | None,
    tags: tuple[str, ...] | None = None,
    situations: tuple[str, ...] | None = None,
    track: str = "turn",
) -> str:
    """Load recent lessons learned for the role, optionally filtered and prioritized.

    ゲーム開始時または行動時に注入する, その役職の教訓を読み込む.
    敗北時の教訓と勝利時の教訓を半分ずつ選択する(LessonLの正/負バランス方式).
    見出し行・重複行を除外する.

    tags を指定した場合はそのタグの教訓のみを返す(例: ("STEER", "PERSONA")).
    situations を指定した場合は, その状況タグを持つ教訓を優先選択する.
    該当する状況の教訓が足りない場合は GENERAL タグの教訓で補充する.

    Args:
        role_value (str): The agent's role value / 自分の役職の値
        player_count (int | None): Number of players in the game / ゲームの参加人数
        tags (tuple[str, ...] | None): Filter by lesson tags (SELF/STEER/PERSONA) /
            タグで絞り込む(None は全て)
        situations (tuple[str, ...] | None): Situation tags to prioritize /
            優先する状況タグ(例: ("CO", "COUNTER"))

    Returns:
        str: Lessons block, or empty string if none / 教訓ブロック. 無ければ空文字列
    """
    if player_count is None:
        return ""
    base = _lessons_base(track)
    path = base / f"{player_count}players" / f"{role_value.lower()}.md"
    if not path.is_file():
        return ""
    t = texts.get()
    lose_label = t.REVIEW_OUTCOME_LOSE
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return ""
    # タグ指定がある場合はそのタグの行のみ残す
    if tags:
        tag_set = {tag.upper() for tag in tags}
        lines = [l for l in lines if _extract_tag(_lesson_body(l)) in tag_set]
        if not lines:
            return ""
    valid: list[str] = []
    for line in lines:
        body = _strip_tag(_lesson_body(line))
        if len(body) < _MIN_LESSON_LENGTH or _HEADING_PATTERN.search(body):
            continue
        valid.append(line)
    losses = [l for l in valid if _extract_outcome(l) == lose_label]
    wins = [l for l in valid if l not in losses]

    # 状況タグの優先選択: 指定された状況に合致する教訓を優先し、
    # 不足分は GENERAL で補充する
    sit_set = {s.upper() for s in situations} if situations else None

    def _select_lessons(pool: list[str], budget: int, selected_bodies: list[str]) -> list[str]:
        """Select up to budget lessons from pool, prioritizing situation matches."""
        result: list[str] = []
        if sit_set:
            # 1st pass: situation-matched lessons (newest first)
            for line in reversed(pool):
                if len(result) >= budget:
                    break
                sit = _extract_situation(line)
                if sit in sit_set:
                    body = _strip_tag(_lesson_body(line))
                    if _is_near_duplicate(body, selected_bodies + [strip_tag_for_dedup(r) for r in result]):
                        continue
                    result.append(line)
                    selected_bodies.append(body)
            # 2nd pass: GENERAL lessons (newest first)
            for line in reversed(pool):
                if len(result) >= budget:
                    break
                sit = _extract_situation(line)
                if sit == "GENERAL":
                    body = _strip_tag(_lesson_body(line))
                    if _is_near_duplicate(body, selected_bodies + [strip_tag_for_dedup(r) for r in result]):
                        continue
                    result.append(line)
                    selected_bodies.append(body)
            # 3rd pass: any remaining (newest first)
            for line in reversed(pool):
                if len(result) >= budget:
                    break
                if line in result:
                    continue
                body = _strip_tag(_lesson_body(line))
                if _is_near_duplicate(body, selected_bodies + [strip_tag_for_dedup(r) for r in result]):
                    continue
                result.append(line)
                selected_bodies.append(body)
        else:
            # 従来通り: 最新順に選ぶ
            for line in reversed(pool):
                if len(result) >= budget:
                    break
                body = _strip_tag(_lesson_body(line))
                if _is_near_duplicate(body, selected_bodies):
                    continue
                result.append(line)
                selected_bodies.append(body)
        return result

    # LessonL方式: 正(勝利)と負(敗北)の教訓を半分ずつ選択する.
    half = _MAX_LESSONS_LINES // 2
    selected_bodies: list[str] = []
    selected_lines: list[str] = []

    # 敗北の教訓を半分選ぶ
    selected_lines.extend(_select_lessons(losses, half, selected_bodies))
    # 勝利の教訓を残り枠選ぶ
    remaining = _MAX_LESSONS_LINES - len(selected_lines)
    if remaining > 0:
        selected_lines.extend(_select_lessons(wins, remaining, selected_bodies))

    selected_lines.reverse()
    if not selected_lines:
        return ""
    return t.REVIEW_LESSONS_HEADER + "\n" + "\n".join(selected_lines)


def strip_tag_for_dedup(line: str) -> str:
    """Strip both tag and situation tag from a lesson line for dedup comparison."""
    body = _lesson_body(line)
    body = _strip_tag(body)
    # Also strip situation tag if present
    body = re.sub(r"^\[[A-Z_]+\]\s*", "", body)
    return body

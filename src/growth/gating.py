"""Module that gates decisions by bounded rationality driven by affect.

感情に駆動される限定合理性で意思決定をゲーティングするモジュール.

感情が強いとき, 意思決定の過程そのものが変わるとみなす(QRE-lambda/level-k の着想).
拙速で反射的になる, あるいは強気に決め打ちする様子をプロンプトで促し, さらに強い動揺の
下では, 熟慮した投票先などを直前に自分を圧迫した相手へ反射的に上書きする(構造的な摂動).
摂動の確率は覚醒と感受性で個体化される. 追加の LLM 呼び出しは行わず, 接続層にも影響しない.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Talk

from growth import texts

# 直近の圧力源を拾うために遡るトーク数
_RECENT_TALK_WINDOW = 12

_GATING_KEYS = {
    ("impulsive", "strong"): "GATING_IMPULSIVE_STRONG",
    ("impulsive", "mild"): "GATING_IMPULSIVE_MILD",
    ("stubborn", "strong"): "GATING_STUBBORN_STRONG",
    ("stubborn", "mild"): "GATING_STUBBORN_MILD",
}


def build_decision_gating(mode: str, level: str) -> str:
    """Build the prompt block describing affect's effect on the decision process.

    感情が意思決定の過程に及ぼす影響を説明するプロンプトブロックを構築する.

    熟慮(deliberate)や平常時には空文字列を返し, 何も注入しない.

    Args:
        mode (str): Decision mode from the affect state / 感情に由来する決定モード
        level (str): Intensity level / 強度レベル

    Returns:
        str: Prompt block text, or empty string / プロンプトブロック. 無ければ空文字列
    """
    key = _GATING_KEYS.get((mode, level))
    if key is None:
        return ""
    return getattr(texts.get(), key, "")


def salient_pressurers(
    info: Info,
    talk_history: list[Talk],
    name: str,
    candidates: list[str],
) -> list[str]:
    """List living candidates who recently pressured the agent (most recent first).

    直近に自分を圧迫した生存候補者を, 新しい順で列挙する.

    直近のトークで自分に言及した相手と, 今日自分へ投票した相手を圧力源とみなす. 反射的な
    上書き先の候補に用いる. 重複は除き, 出現順(新しい順)を保つ.

    Args:
        info (Info): Current game info / 現在のゲーム情報
        talk_history (list[Talk]): Talk history / トーク履歴
        name (str): The agent's in-game name / ゲーム内のエージェント名
        candidates (list[str]): Allowed target names / 選べる対象名の一覧

    Returns:
        list[str]: Pressurer names, newest first / 圧力源の名前(新しい順)
    """
    allowed = set(candidates)
    ordered: list[str] = []
    for talk in reversed(talk_history[-_RECENT_TALK_WINDOW:]):
        text = talk.text or ""
        if talk.agent != name and not talk.skip and name and name in text and talk.agent in allowed:
            ordered.append(talk.agent)
    ordered.extend(
        vote.agent
        for vote in info.vote_list or []
        if vote.target == name and vote.agent != name and vote.agent in allowed
    )
    seen: set[str] = set()
    result: list[str] = []
    for pressurer in ordered:
        if pressurer not in seen:
            seen.add(pressurer)
            result.append(pressurer)
    return result


def impulsive_override(salient: list[str], deliberated: str | None) -> str | None:
    """Pick a reactive target to override the deliberated choice, if one differs.

    熟慮した選択を上書きする反射的な対象を選ぶ. 熟慮先と異なる圧力源がある時のみ返す.

    Args:
        salient (list[str]): Pressurer names, newest first / 圧力源の名前(新しい順)
        deliberated (str | None): The deliberated target / 熟慮して選んだ対象

    Returns:
        str | None: The override target, or None if none differs /
            上書き先. 異なる候補が無ければ None
    """
    return next((s for s in salient if s != deliberated), None)

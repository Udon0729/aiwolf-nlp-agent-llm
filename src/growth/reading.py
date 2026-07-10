"""Module that builds the prompt block for reading others' emotional tells.

他者の感情のテルを読むためのプロンプトブロックを構築するモジュール.

自分の感情が漏れる(emotion.py)のと対をなす, 他者の発話から感情・態度の手がかりを
読み取る側. 敵対的な場では表出が演技・誤誘導でありうるため, テルは手がかりに留め
確定事実と併せて判断するよう促す. 構造化した信念の保持と終局照合は後段の課題.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Talk

from growth import texts


def _status_value(status: object) -> str:
    """Return the string value of a status enum (or its str form).

    ステータス列挙体の文字列値を返す.

    Args:
        status (object): Status value from status_map / status_map の値

    Returns:
        str: String value such as "ALIVE" / "ALIVE" などの文字列値
    """
    return str(getattr(status, "value", status))


def _latest_utterances(agent_name: str, talk_history: list[Talk]) -> dict[str, str]:
    """Collect each other speaker's most recent substantive utterance.

    自分以外の各話者の最新の実発言(skip/over/空でないもの)を集める.

    Args:
        agent_name (str): The agent's in-game name / ゲーム内のエージェント名
        talk_history (list[Talk]): Talk history of the game / ゲームのトーク履歴

    Returns:
        dict[str, str]: Mapping of speaker name to their latest utterance /
            話者名から最新発言への対応
    """
    latest: dict[str, str] = {}
    for talk in talk_history:
        if talk.agent == agent_name or not talk.text or talk.skip or talk.over:
            continue
        latest[talk.agent] = talk.text
    return latest


def build_tell_reading(info: Info | None, agent_name: str, talk_history: list[Talk]) -> str:
    """Build the block instructing the agent to read others' emotional tells.

    他者の感情のテルを読むよう促すブロックを構築する.

    発言の内容だけでなく感情・態度(動揺・弁明・回避・不自然な平静など)を手がかりとして
    役職推定に用いるよう促す. ただし表出は演技・誤誘導でありうるため, 確定事実と併せて
    判断するよう注意を添える. 読むべき他者の発言が無い場合は空文字列を返す.

    Args:
        info (Info | None): Current game info / 現在のゲーム情報
        agent_name (str): The agent's in-game name / ゲーム内のエージェント名
        talk_history (list[Talk]): Talk history of the game / ゲームのトーク履歴

    Returns:
        str: Tell-reading block, or empty string if there is nothing to read /
            テル読みブロック. 読む対象が無い場合は空文字列
    """
    if info is None:
        return ""
    t = texts.get()
    alive = {k for k, v in info.status_map.items() if _status_value(v) == "ALIVE"}
    latest = _latest_utterances(agent_name, talk_history)
    lines = [
        t.READING_UTTERANCE_LINE.format(name=name, text=text)
        for name, text in latest.items()
        if name in alive
    ]
    if not lines:
        return ""
    return t.READING_TELL_HEADER + "\n" + "\n".join(lines)

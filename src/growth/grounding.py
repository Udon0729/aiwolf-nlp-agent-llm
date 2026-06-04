"""Module that builds grounding context injected into prompts.

プロンプトに注入する接地情報を構築するモジュール.

「確定事実」(生死・追放・襲撃・投票・占い/霊媒結果)と「自己スタンス」(役職・自分が得た
情報・自分の過去の発言)をテキスト化し, 幻覚と過同調を抑える土台を作る.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Role, Talk

_RECENT_OWN_TALKS = 3


def _status_value(status: object) -> str:
    """Return the string value of a status enum (or its str form).

    ステータス列挙体の文字列値を返す.

    Args:
        status (object): Status value from status_map / status_map の値

    Returns:
        str: String value such as "ALIVE" / "ALIVE" などの文字列値
    """
    return str(getattr(status, "value", status))


def build_confirmed_facts(info: Info | None) -> str:
    """Build the confirmed-facts block from game info.

    ゲーム情報から「確定事実」ブロックを構築する.

    Args:
        info (Info | None): Current game info / 現在のゲーム情報

    Returns:
        str: Confirmed-facts block, or empty string if info is None /
            確定事実ブロック. info が None の場合は空文字列
    """
    if info is None:
        return ""
    lines: list[str] = [f"- 日付: {info.day}日目"]
    alive = [k for k, v in info.status_map.items() if _status_value(v) == "ALIVE"]
    dead = [k for k, v in info.status_map.items() if _status_value(v) != "ALIVE"]
    lines.append(f"- 生存者: {', '.join(alive) if alive else 'なし'}")
    if dead:
        lines.append(f"- 死亡者: {', '.join(dead)}")
    if info.executed_agent is not None:
        lines.append(f"- 直近の追放(吊り): {info.executed_agent}")
    if info.attacked_agent is not None:
        lines.append(f"- 直近の襲撃(噛み): {info.attacked_agent}")
    if info.vote_list is not None:
        lines.append(f"- 投票結果: {info.vote_list}")
    if info.attack_vote_list is not None:
        lines.append(f"- 襲撃投票結果: {info.attack_vote_list}")
    return "## 確定事実(これに基づいて判断すること)\n" + "\n".join(lines)


def build_self_stance(
    agent_name: str,
    role: Role,
    talk_history: list[Talk],
    known_results: list[str],
) -> str:
    """Build the self-stance block to keep the agent consistent.

    エージェントの一貫性を保つための「自己スタンス」ブロックを構築する.

    Args:
        agent_name (str): The agent's in-game name / ゲーム内のエージェント名
        role (Role): The agent's role / エージェントの役職
        talk_history (list[Talk]): Talk history of the game / ゲームのトーク履歴
        known_results (list[str]): Private results obtained (divine/medium) /
            自分が得た占い・霊媒結果

    Returns:
        str: Self-stance block / 自己スタンスブロック
    """
    role_value = str(getattr(role, "value", role))
    lines: list[str] = [f"- あなたの役職: {role_value}(他者はあなたの役職を知らない)"]
    if known_results:
        lines.append("- あなたが得た占い/霊媒結果: " + "; ".join(known_results))
    own_talks = [
        t.text for t in talk_history if t.agent == agent_name and t.text and not t.skip and not t.over
    ]
    if own_talks:
        recent = own_talks[-_RECENT_OWN_TALKS:]
        lines.append("- あなたがこれまで公言したこと: " + " / ".join(f"「{text}」" for text in recent))
    return "## あなたのこれまでの立場(これと矛盾しないこと)\n" + "\n".join(lines)


_ATTENTION = (
    "## 注意\n"
    "- 上記の確定事実を最優先する。確定事実に無い出来事を、起きたことのように語らない。\n"
    "- 他者の発言は誤り・かく乱の可能性がある。根拠なく同調せず、確定事実と照らして判断する。\n"
    "- これまでの自分の主張と矛盾しないこと。"
)


def build_action_grounding(
    info: Info | None,
    agent_name: str,
    role: Role,
    talk_history: list[Talk],
    known_results: list[str],
) -> str:
    """Build the full grounding prefix for an action prompt.

    行動プロンプトに前置する接地情報全体(確定事実+自己スタンス+注意)を構築する.

    Args:
        info (Info | None): Current game info / 現在のゲーム情報
        agent_name (str): The agent's in-game name / ゲーム内のエージェント名
        role (Role): The agent's role / エージェントの役職
        talk_history (list[Talk]): Talk history of the game / ゲームのトーク履歴
        known_results (list[str]): Private results obtained / 自分が得た占い・霊媒結果

    Returns:
        str: Grounding prefix text / 接地情報の前置テキスト
    """
    blocks = [
        build_confirmed_facts(info),
        build_self_stance(agent_name, role, talk_history, known_results),
        _ATTENTION,
    ]
    return "\n\n".join(block for block in blocks if block)

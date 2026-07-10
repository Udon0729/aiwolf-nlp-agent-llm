"""Extract truth-grounded metrics from a server game-record JSON.

サーバのゲーム記録 JSON から, 真値に接地した評価メトリクスを抽出する.

人狼知能サーバが `log/json/*.json` に出力する1ゲームの完全記録を入力に取り, 勝敗・役職・
投票の的中・占いの的中といった, 役職推定の精度を測る指標を計算する. 自己対戦では陣営勝率が
対称な交絡を受けるため, 投票/占いの的中率を主指標, 陣営勝率を従指標として扱う.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# 人狼陣営の役職集合
_WOLF_SIDE = {"WEREWOLF", "POSSESSED"}
# 村人陣営の役職集合
_VILLAGE_SIDE = {"VILLAGER", "SEER", "MEDIUM", "BODYGUARD"}


def _full_role_map(entries: list[dict[str, Any]]) -> dict[str, str]:
    """Return the revealed role map (persona name to role) from a FINISH entry.

    FINISH エントリから, 開示された役職表(名前→役職)を返す.

    Args:
        entries (list[dict[str, Any]]): Game entries / ゲームのエントリ列

    Returns:
        dict[str, str]: Persona name to role / 名前ごとの役職
    """
    for entry in entries:
        request = json.loads(entry["request"])
        if request["request"] == "FINISH":
            role_map = request["info"].get("role_map")
            if role_map:
                return dict(role_map)
    return {}


def _final_status(entries: list[dict[str, Any]]) -> dict[str, str]:
    """Return the final alive/dead status map from a FINISH entry.

    FINISH エントリから, 最終的な生死表を返す.

    Args:
        entries (list[dict[str, Any]]): Game entries / ゲームのエントリ列

    Returns:
        dict[str, str]: Persona name to status / 名前ごとの生死
    """
    for entry in entries:
        request = json.loads(entry["request"])
        if request["request"] == "FINISH":
            status_map = request["info"].get("status_map")
            if status_map:
                return dict(status_map)
    return {}


def _targets(entries: list[dict[str, Any]], request_type: str) -> list[tuple[str, str]]:
    """List (actor, target) pairs for the given target-selecting request type.

    指定した対象選択リクエストの (行為者, 対象) の組を列挙する.

    Args:
        entries (list[dict[str, Any]]): Game entries / ゲームのエントリ列
        request_type (str): Request type such as "VOTE"/"DIVINE" / リクエスト種別

    Returns:
        list[tuple[str, str]]: (actor, target) pairs / (行為者, 対象) の組
    """
    pairs: list[tuple[str, str]] = []
    for entry in entries:
        request = json.loads(entry["request"])
        if request["request"] != request_type:
            continue
        actor = entry.get("agent")
        target = entry.get("response")
        if actor and target:
            pairs.append((actor, str(target).strip()))
    return pairs


def parse_game(path: str | Path) -> dict[str, Any]:
    """Parse a game-record JSON into truth-grounded metrics.

    ゲーム記録 JSON を, 真値接地のメトリクスへ解析する.

    Args:
        path (str | Path): Path to the server game-record JSON / 記録 JSON のパス

    Returns:
        dict[str, Any]: Metrics for the single game / 1ゲーム分のメトリクス
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data["entries"]
    roles = _full_role_map(entries)
    status = _final_status(entries)
    win_side = data.get("win_side")

    votes = _targets(entries, "VOTE")
    village_votes = [(a, t) for a, t in votes if roles.get(a) in _VILLAGE_SIDE]
    vote_total = len(village_votes)
    vote_hit_wolfside = sum(1 for _, t in village_votes if roles.get(t) in _WOLF_SIDE)
    vote_hit_werewolf = sum(1 for _, t in village_votes if roles.get(t) == "WEREWOLF")

    divines = _targets(entries, "DIVINE")
    divine_total = len(divines)
    divine_hit_werewolf = sum(1 for _, t in divines if roles.get(t) == "WEREWOLF")

    attacks = _targets(entries, "ATTACK")
    # 人狼が誤って自陣営(狂人)を襲撃した回数(連携の失敗)
    attack_on_ally = sum(1 for _, t in attacks if roles.get(t) in _WOLF_SIDE)

    return {
        "game_id": data.get("game_id"),
        "win_side": win_side,
        "village_won": win_side == "VILLAGER",
        "roles": roles,
        "final_status": status,
        "vote_total": vote_total,
        "vote_hit_wolfside": vote_hit_wolfside,
        "vote_hit_werewolf": vote_hit_werewolf,
        "divine_total": divine_total,
        "divine_hit_werewolf": divine_hit_werewolf,
        "attack_total": len(attacks),
        "attack_on_ally": attack_on_ally,
    }


def _format(metrics: dict[str, Any]) -> str:
    """Format single-game metrics for human reading.

    1ゲーム分のメトリクスを人間向けに整形する.

    Args:
        metrics (dict[str, Any]): Metrics from parse_game / parse_game のメトリクス

    Returns:
        str: Formatted summary / 整形した要約
    """
    vt = metrics["vote_total"]
    dt = metrics["divine_total"]
    vote_rate = metrics["vote_hit_wolfside"] / vt if vt else 0.0
    divine_rate = metrics["divine_hit_werewolf"] / dt if dt else 0.0
    return (
        f"game={metrics['game_id']} win_side={metrics['win_side']}\n"
        f"  村側投票の人狼陣営的中: {metrics['vote_hit_wolfside']}/{vt} ({vote_rate:.0%}) "
        f"うち人狼本体: {metrics['vote_hit_werewolf']}\n"
        f"  占いの人狼的中: {metrics['divine_hit_werewolf']}/{dt} ({divine_rate:.0%})\n"
        f"  人狼の自陣営誤襲撃: {metrics['attack_on_ally']}/{metrics['attack_total']}"
    )


def main() -> None:
    """Print metrics for each game-record JSON given on the command line.

    コマンドライン引数の各ゲーム記録 JSON のメトリクスを出力する.
    """
    for path in sys.argv[1:]:
        print(_format(parse_game(path)))


if __name__ == "__main__":
    main()

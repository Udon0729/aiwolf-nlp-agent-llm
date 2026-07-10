"""Relate each role-holder's emotional activation to their side's win.

役職ごとに, その役を担ったエージェントの感情の出方とその陣営の勝敗を結びつける.

各対局のエージェントログ(1ディレクトリ=1ゲームの kanolab*.log)は自己完結で, 全役職・
最終生死・各自の感情マーカー(漏洩の強さ・ゲーティング作動・上書き)を含む. これらから
役職別に陣営勝率を出し, さらに「担当者の感情が強く出た対局」と「出なかった対局」で勝率を
比較する. 自己対戦では役職別勝率は陣営勝率に帰着するため, 感情と勝敗の相関を主眼に置く.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_WOLF_SIDE = {"WEREWOLF", "POSSESSED"}
_ROLE_RE = re.compile(r"'([^']+)':\s*<Role\.([A-Z]+):")
_STATUS_RE = re.compile(r"'([^']+)':\s*<Status\.([A-Z]+):")
_AGENT_RE = re.compile(r"agent='([^']+)'")


def _finish_maps(log_text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Extract the full role and status maps from a FINISH packet line.

    FINISH パケット行から全役職表と最終生死表を取り出す.

    Args:
        log_text (str): Full text of one agent log / 1エージェントログの全文

    Returns:
        tuple[dict[str, str], dict[str, str]]: (role_map, status_map) /
            (役職表, 生死表). 取れなければ空辞書
    """
    for line in log_text.splitlines():
        if "Request.FINISH" not in line:
            continue
        role_block = re.search(r"role_map=\{([^}]*)\}", line)
        status_block = re.search(r"status_map=\{([^}]*)\}", line)
        if not role_block:
            continue
        roles = dict(_ROLE_RE.findall(role_block.group(1)))
        if len(roles) < 2:  # noqa: PLR2004
            continue
        statuses = dict(_STATUS_RE.findall(status_block.group(1))) if status_block else {}
        return roles, statuses
    return {}, {}


def _emotion_markers(log_text: str) -> dict[str, int]:
    """Count emotional-activation markers in one agent log.

    1エージェントログ中の感情作動マーカーを数える.

    Args:
        log_text (str): Full text of one agent log / 1エージェントログの全文

    Returns:
        dict[str, int]: Marker counts / マーカーの回数
    """
    return {
        "strong": log_text.count("にじみ出る"),
        "gating": log_text.count("意思決定への影響"),
        "override": log_text.count("反射的に上書き"),
    }


def _parse_dir(game_dir: Path) -> dict | None:
    """Parse one game directory into role-wise outcome and emotion records.

    1ゲームのディレクトリを, 役職ごとの勝敗と感情の記録へ解析する.

    Args:
        game_dir (Path): Directory holding kanolab*.log / kanolab*.log を含むディレクトリ

    Returns:
        dict | None: Game record, or None if unparseable / ゲーム記録. 解析不能なら None
    """
    logs = sorted(game_dir.glob("kanolab*.log"))
    if not logs:
        return None
    texts = {p: p.read_text(encoding="utf-8", errors="ignore") for p in logs}
    roles: dict[str, str] = {}
    statuses: dict[str, str] = {}
    for text in texts.values():
        roles, statuses = _finish_maps(text)
        if roles:
            break
    if not roles:
        return None
    werewolf = next((name for name, role in roles.items() if role == "WEREWOLF"), None)
    village_won = werewolf is not None and statuses.get(werewolf) == "DEAD"
    per_agent: dict[str, dict] = {}
    for text in texts.values():
        agent_match = _AGENT_RE.search(text)
        if not agent_match:
            continue
        persona = agent_match.group(1)
        role = roles.get(persona)
        if role is None:
            continue
        markers = _emotion_markers(text)
        side_won = village_won != (role in _WOLF_SIDE)
        per_agent[persona] = {"role": role, "won": side_won, **markers}
    return {"village_won": village_won, "agents": per_agent}


def main() -> None:
    """Print role-wise win rates and emotion-split win rates over game dirs.

    複数のゲームディレクトリにわたり, 役職別勝率と感情の出方で分けた勝率を出力する.
    """
    roots = [Path(a) for a in sys.argv[1:]] or [Path("log")]
    game_dirs: list[Path] = []
    for root in roots:
        game_dirs.extend(sorted(d for d in root.glob("2*/") if d.is_dir()))
    games = [g for g in (_parse_dir(d) for d in game_dirs) if g]
    if not games:
        print("解析できる対局ログが見つかりません。", file=sys.stderr)
        sys.exit(1)

    # 役職ごとに集計
    stats: dict[str, dict[str, list]] = {}
    for game in games:
        for rec in game["agents"].values():
            role = rec["role"]
            slot = stats.setdefault(role, {"won": [], "emotional": []})
            slot["won"].append(rec["won"])
            slot["emotional"].append(rec["strong"] > 0 or rec["gating"] > 0)

    print(f"================ 役職別の勝敗と感情 ({len(games)}ゲーム) ================")
    print(f"{'role':<12}{'N':>4}{'陣営勝率':>9}{'感情出た時の勝率':>20}{'平静時の勝率':>16}")
    for role in ("WEREWOLF", "POSSESSED", "SEER", "VILLAGER"):
        if role not in stats:
            continue
        won = stats[role]["won"]
        emo = stats[role]["emotional"]
        n = len(won)
        wr = sum(won) / n if n else 0.0
        emo_won = [w for w, e in zip(won, emo, strict=True) if e]
        calm_won = [w for w, e in zip(won, emo, strict=True) if not e]
        emo_wr = f"{sum(emo_won) / len(emo_won) * 100:>3.0f}% (n={len(emo_won)})" if emo_won else "    - (n=0)"
        calm_wr = f"{sum(calm_won) / len(calm_won) * 100:>3.0f}% (n={len(calm_won)})" if calm_won else "    - (n=0)"
        print(f"{role:<12}{n:>4}{wr * 100:>8.0f}%{emo_wr:>20}{calm_wr:>16}")


if __name__ == "__main__":
    main()

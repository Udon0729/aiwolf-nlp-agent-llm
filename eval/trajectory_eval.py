"""Trajectory-aware evaluation: does affect over time match the persona type.

軌跡を見る評価: 感情の時系列がペルソナ類型に合っているかを測る.

一点採点の整合ジャッジは時間的な情動の揺れを見ないため, 遅発の噴出を誤って減点する.
本評価はエージェントログから感情レベルの時系列を復元し, プロフィール由来の類型
(不動型/溜め型/興奮型/中間)が予測する軌跡と一致するかを採点する. 不動型はフラット,
興奮型は即噴出, 溜め型は耐えてから遅れて表面化, を期待値とする. 追加のLLM呼び出しは不要.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from growth.emotion import derive_traits

_AGENT_RE = re.compile(r"agent='([^']+)'")
_PROFILE_RE = re.compile(r"profile='([^']*)'")
_BLOCK_RE = re.compile(r"## あなたの今の感情[^#]{0,250}")
# 感受性の類型しきい値
_SENS_LOW = 0.3
_SENS_HIGH = 0.7
# 強い情動が早期とみなすターン位置
_IMMEDIATE_TURN = 2
# 遅発とみなす最小ターン位置
_DELAYED_TURN = 3


def _level(block: str) -> str:
    """Map an injection block to a one-letter affect level.

    感情注入ブロックを1文字の感情レベルへ写す.

    Args:
        block (str): The injected affect block text / 注入された感情ブロック

    Returns:
        str: One of ".meSB?" / ".meSB?" のいずれか
    """
    if "抑えきれなくなっている" in block:
        return "B"
    if "表向きは平静を保っている" in block:
        return "e"
    if "にじみ出る" in block:
        return "S"
    if "程度でよい" in block:
        return "m"
    if "特段の動揺はない" in block:
        return "."
    return "?"


def _kind(sensitivity: float, *, suppression: bool) -> str:
    """Classify the persona type from derived traits.

    導出した係数からペルソナ類型を分類する.

    Args:
        sensitivity (float): Derived sensitivity / 導出した感受性
        suppression (bool): Whether the persona suppresses / 抑制性向か

    Returns:
        str: One of the persona types / ペルソナ類型
    """
    if suppression:
        return "溜め型"
    if sensitivity <= _SENS_LOW:
        return "不動型"
    if sensitivity >= _SENS_HIGH:
        return "興奮型"
    return "中間"


def _classify(seq: str) -> dict:
    """Summarise an affect-level sequence into trajectory features.

    感情レベルの時系列を, 軌跡の特徴へ要約する.

    Args:
        seq (str): The level sequence such as "..mSS" / レベル系列

    Returns:
        dict: Trajectory features / 軌跡の特徴
    """
    intensity = {".": 0, "m": 1, "e": 1, "S": 2, "B": 2}
    ints = [intensity.get(c, 0) for c in seq]
    max_int = max(ints) if ints else 0
    first_strong = next((i for i, v in enumerate(ints) if v >= 2), None)  # noqa: PLR2004
    used_suppression = "e" in seq or "B" in seq
    if max_int == 0:
        pattern = "flat"
    elif max_int == 1:
        pattern = "mild-only"
    elif first_strong is not None and first_strong <= _IMMEDIATE_TURN:
        pattern = "immediate-burst"
    elif first_strong is not None and first_strong >= _DELAYED_TURN:
        pattern = "slow-burn"
    else:
        pattern = "mid-burst"
    return {
        "pattern": pattern,
        "max_int": max_int,
        "first_strong": first_strong,
        "suppressed": used_suppression,
    }


def _matches(kind: str, feat: dict) -> bool:
    """Decide whether the trajectory matches the persona type's expectation.

    軌跡がペルソナ類型の期待に合うかを判定する.

    Args:
        kind (str): Persona type / ペルソナ類型
        feat (dict): Trajectory features / 軌跡の特徴

    Returns:
        bool: True if appropriate / 期待に合えば True
    """
    pattern = feat["pattern"]
    if kind == "不動型":
        return pattern != "immediate-burst"
    if kind == "興奮型":
        return feat["max_int"] >= 2  # noqa: PLR2004
    if kind == "溜め型":
        return feat["suppressed"] and pattern != "immediate-burst"
    return True


def _scan(roots: list[Path]) -> list[dict]:
    """Extract per-agent trajectory records from agent log directories.

    エージェントログのディレクトリから, エージェントごとの軌跡記録を取り出す.

    Args:
        roots (list[Path]): Roots holding game-log dirs / 対局ログのディレクトリ群

    Returns:
        list[dict]: Per-agent records / エージェントごとの記録
    """
    records: list[dict] = []
    for root in roots:
        for game_dir in sorted(d for d in root.glob("2*/") if d.is_dir()):
            for log_file in sorted(game_dir.glob("kanolab*.log")):
                text = log_file.read_text(encoding="utf-8", errors="ignore")
                agent_match = _AGENT_RE.search(text)
                profile_match = _PROFILE_RE.search(text)
                if not agent_match or not profile_match:
                    continue
                profile = profile_match.group(1).replace("\\n", "\n")
                traits = derive_traits(profile)
                seq = "".join(_level(b) for b in _BLOCK_RE.findall(text))
                if not seq:
                    continue
                feat = _classify(seq)
                kind = _kind(traits.sensitivity, suppression=traits.suppression)
                records.append({
                    "agent": agent_match.group(1),
                    "kind": kind,
                    "seq": seq,
                    "match": _matches(kind, feat),
                    **feat,
                })
    return records


def main() -> None:
    """Report per-type trajectory features and persona-trajectory match rate.

    類型別の軌跡の特徴と, ペルソナと軌跡の一致率を報告する.
    """
    roots = [Path(a) for a in sys.argv[1:]] or [Path("log_verify")]
    records = _scan(roots)
    if not records:
        print("軌跡を抽出できるログが見つかりません。", file=sys.stderr)
        sys.exit(1)

    print(f"================ 軌跡評価 ({len(records)}エージェント) ================")
    print(f"{'類型':<8}{'N':>4}{'一致率':>8}{'即噴出':>8}{'遅発':>8}{'フラット':>10}{'抑制使用':>10}")
    for kind in ("不動型", "溜め型", "興奮型", "中間"):
        rows = [r for r in records if r["kind"] == kind]
        if not rows:
            continue
        n = len(rows)
        match = sum(r["match"] for r in rows) / n
        immediate = sum(r["pattern"] == "immediate-burst" for r in rows) / n
        slow = sum(r["pattern"] == "slow-burn" for r in rows) / n
        flat = sum(r["pattern"] == "flat" for r in rows) / n
        supp = sum(r["suppressed"] for r in rows) / n
        print(f"{kind:<8}{n:>4}{match:>7.0%}{immediate:>8.0%}{slow:>8.0%}{flat:>10.0%}{supp:>10.0%}")
    overall = sum(r["match"] for r in records) / len(records)
    print(f"\n全体のペルソナ-軌跡一致率: {overall:.0%}")
    print("\n期待: 不動型=即噴出しない, 興奮型=強へ到達, 溜め型=抑制を見せ即噴出しない")


if __name__ == "__main__":
    main()

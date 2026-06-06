"""Combine per-worker ablation result JSONs and test the primary metric.

ワーカー別の要因除去結果 JSON を統合し, 主指標の有意性を検定する.

並列実行では各条件のワーカーが個別の結果 JSON を書き出すため, それらを1つの表に統合する.
主指標(村側投票の人狼陣営的中率)について, full と各 leave-one-out の差を2標本比率の
z 検定で評価する(正規近似, 外部ライブラリ非依存). 自己対戦で交絡する陣営勝率は参考併記する.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
# leave-one-out の条件名。表示順
_ORDER = ["full", "no_emotion", "no_reading", "no_gating"]
# 有意性の表示閾値
_P_STRONG = 0.01
_P_MEDIUM = 0.05
_P_WEAK = 0.1


def _latest_per_worker() -> list[Path]:
    """Return the newest result JSON for each worker tag w0..w3.

    各ワーカータグ w0..w3 について最新の結果 JSON を返す.

    Returns:
        list[Path]: Newest path per worker tag / ワーカーごとの最新パス
    """
    found: list[Path] = []
    for tag in ("w0", "w1", "w2", "w3"):
        matches = sorted(
            _RESULTS_DIR.glob(f"ablation_*_{tag}.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if matches:
            found.append(matches[-1])
    return found


def _load(paths: list[str]) -> dict[str, dict]:
    """Merge condition summaries from the given result JSON paths.

    指定した結果 JSON のパスから, 条件ごとの集計を統合する.

    Args:
        paths (list[str]): Result JSON paths / 結果 JSON のパス列

    Returns:
        dict[str, dict]: Condition name to summary / 条件名ごとの集計
    """
    merged: dict[str, dict] = {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for condition, value in data.items():
            merged[condition] = value["summary"]
    return merged


def _two_proportion_z(hit_a: int, n_a: int, hit_b: int, n_b: int) -> tuple[float, float]:
    """Return the z statistic and two-sided p for two proportions.

    2つの比率に対する z 統計量と両側 p 値を返す(正規近似).

    Args:
        hit_a (int): Successes in group A / 群Aの成功数
        n_a (int): Trials in group A / 群Aの試行数
        hit_b (int): Successes in group B / 群Bの成功数
        n_b (int): Trials in group B / 群Bの試行数

    Returns:
        tuple[float, float]: (z, p_two_sided) / (z 統計量, 両側 p 値)
    """
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    p_a = hit_a / n_a
    p_b = hit_b / n_b
    pool = (hit_a + hit_b) / (n_a + n_b)
    denom = math.sqrt(pool * (1 - pool) * (1 / n_a + 1 / n_b))
    if denom == 0:
        return 0.0, 1.0
    z = (p_a - p_b) / denom
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


def main() -> None:
    """Print the combined ablation table and significance of the primary metric.

    統合した要因除去の表と, 主指標の有意性を出力する.
    """
    paths = sys.argv[1:] or [str(p) for p in _latest_per_worker()]
    if not paths:
        print("結果 JSON が見つかりません。", file=sys.stderr)
        sys.exit(1)
    merged = _load(paths)
    order = [c for c in _ORDER if c in merged] + [c for c in merged if c not in _ORDER]

    print("================ 統合結果 ================")
    print(f"{'condition':<12}{'N':>4}{'村勝率':>8}{'投票的中(主)':>16}{'占い的中':>14}{'上書き':>7}{'ゲート注入':>10}")
    for c in order:
        s = merged[c]
        print(
            f"{c:<12}{s['parsed']:>4}{s['village_win_rate']*100:>7.0f}%"
            f"{s['vote_hit_wolfside']:>7}/{s['vote_total']:<4}({s['vote_acc_wolfside']*100:>3.0f}%)"
            f"{s['divine_hit_werewolf']:>5}/{s['divine_total']:<3}({s['divine_acc_werewolf']*100:>3.0f}%)"
            f"{s['override_total']:>7}{s['gating_inject_total']:>10}",
        )

    if "full" not in merged:
        return
    full = merged["full"]
    print("\n--- 主指標(投票的中)の full との差: 2標本比率 z 検定 ---")
    for c in order:
        if c == "full":
            continue
        s = merged[c]
        z, p = _two_proportion_z(
            full["vote_hit_wolfside"], full["vote_total"], s["vote_hit_wolfside"], s["vote_total"],
        )
        sig = "***" if p < _P_STRONG else "**" if p < _P_MEDIUM else "*" if p < _P_WEAK else ""
        drop = full["vote_acc_wolfside"] - s["vote_acc_wolfside"]
        print(f"  full vs {c:<11}: Δ={drop*100:+5.0f}pt  z={z:+.2f}  p={p:.3f} {sig}")


if __name__ == "__main__":
    main()

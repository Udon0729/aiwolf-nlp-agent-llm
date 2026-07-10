"""Run a leave-one-out ablation over growth-layer features and aggregate metrics.

growth 層の機能について leave-one-out の要因除去を行い, メトリクスを集計するランナー.

各条件(full / full-emotion / full-reading / full-gating)ごとに, growth トグルを設定した
一時 config を生成して K ゲーム対局させ, サーバのゲーム記録から真値接地のメトリクスを集計する.
対局間学習(review)はバッチ内の非定常性を避けるため全条件で無効にする. vllm とゲームサーバは
本スクリプトの外で起動済みであることを前提とする. 接続層(main.py)は無変更で, config 切替のみで制御する.

使い方:
    uv run python eval/run_eval.py [K] [condition ...]
    K は1条件あたりのゲーム数(既定 5). condition を指定すると一部条件のみ実行する.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
from parse_game import parse_game

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BASE_CONFIG = _PROJECT_ROOT / "config" / "config.yml"
_RESULTS_DIR = _PROJECT_ROOT / "eval" / "results"
_TMP_DIR = _PROJECT_ROOT / "eval" / "_tmp"
# 1ゲームあたりの上限秒数(これを超えたら異常とみなし打ち切る)
_GAME_TIMEOUT_S = 1200

# 並列ワーカーごとに分離する接続先・出力先(環境変数で上書き可能。既定は逐次実行向け)
_WS_URL = os.environ.get("EVAL_WS_URL", "ws://127.0.0.1:8080/ws")
_VLLM_BASE_URL = os.environ.get("EVAL_VLLM_BASE_URL", "http://localhost:8000/v1")
_SERVER_JSON_DIR = Path(
    os.environ.get("EVAL_SERVER_JSON_DIR", "/diskthalys/ssd14ta/kmunaoka/aiwolf/aiwolf-nlp-server/log/json"),
)
_PROJECT_LOG_DIR = Path(os.environ.get("EVAL_AGENT_LOG_DIR", str(_PROJECT_ROOT / "log")))
# 並列ワーカーの結果ファイル名の衝突を避けるためのタグ
_RESULTS_TAG = os.environ.get("EVAL_RESULTS_TAG", "")
# 感受性スイープ用の全体係数(設定時は emotion.sensitivity_scale に反映)
_SENSITIVITY_SCALE = os.environ.get("EVAL_SENSITIVITY_SCALE", "")
# 全条件で対局間学習を無効化する
_REVIEW_OFF = {"enabled": False}

# leave-one-out の各条件。full から1機能だけ除いた4条件。emotion_only は感受性スイープ用
_CONDITIONS: dict[str, dict[str, dict[str, bool]]] = {
    "full": {"emotion": {"enabled": True}, "reading": {"enabled": True}, "gating": {"enabled": True}},
    "no_emotion": {"emotion": {"enabled": False}, "reading": {"enabled": True}, "gating": {"enabled": True}},
    "no_reading": {"emotion": {"enabled": True}, "reading": {"enabled": False}, "gating": {"enabled": True}},
    "no_gating": {"emotion": {"enabled": True}, "reading": {"enabled": True}, "gating": {"enabled": False}},
    "emotion_only": {"emotion": {"enabled": True}, "reading": {"enabled": False}, "gating": {"enabled": False}},
}


def _write_config(condition: str, growth: dict[str, dict[str, bool]]) -> Path:
    """Write a temp config with the given growth toggles and review disabled.

    指定した growth トグルと review 無効を設定した一時 config を書き出す.

    Args:
        condition (str): Condition name / 条件名
        growth (dict): Growth toggles for the condition / 条件の growth トグル

    Returns:
        Path: Path to the written config / 書き出した config のパス
    """
    base = yaml.safe_load(_BASE_CONFIG.read_text(encoding="utf-8"))
    base["growth"] = {**growth, "review": dict(_REVIEW_OFF)}
    # 感受性スイープ: 設定時は emotion セクションに sensitivity_scale を上書きする
    if _SENSITIVITY_SCALE:
        emotion = dict(base["growth"].get("emotion", {"enabled": True}))
        emotion["sensitivity_scale"] = float(_SENSITIVITY_SCALE)
        base["growth"]["emotion"] = emotion
    # 並列ワーカーごとの接続先・エージェントログ出力先を反映する
    base["web_socket"]["url"] = _WS_URL
    base["openai"]["base_url"] = _VLLM_BASE_URL
    base["log"]["output_dir"] = str(_PROJECT_LOG_DIR)
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    # 並列ワーカーが同一条件名を使う場合(感受性スイープ等)に config が衝突しないよう
    # ワーカータグを名前に含める。タグが無ければ条件名のみ。
    tag = f"_{_RESULTS_TAG}" if _RESULTS_TAG else ""
    path = _TMP_DIR / f"config_{condition}{tag}.yml"
    path.write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _run_one_game(config_path: Path, log_path: Path) -> bool:
    """Launch the 5 agents for one game and wait for completion.

    1ゲーム分の5体のエージェントを起動し, 完了を待つ.

    Args:
        config_path (Path): Config to run with / 使用する config
        log_path (Path): Where to write agent stdout / エージェント標準出力の書き出し先

    Returns:
        bool: True if the game finished normally / 正常終了なら True
    """
    with log_path.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(  # noqa: S603
            ["uv", "run", "src/main.py", "-c", str(config_path)],  # noqa: S607
            cwd=str(_PROJECT_ROOT),
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=_GAME_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return False
        return proc.returncode == 0


def _newest_record(before: set[str]) -> Path | None:
    """Return the game-record JSON created since the snapshot.

    スナップショット以降に作られたゲーム記録 JSON を返す.

    Args:
        before (set[str]): Filenames present before the game / 対局前に存在した名前

    Returns:
        Path | None: The new record, or None / 新規記録. 無ければ None
    """
    current = {p.name for p in _SERVER_JSON_DIR.glob("*.json")}
    new = current - before
    if not new:
        return None
    return max((_SERVER_JSON_DIR / n for n in new), key=lambda p: p.stat().st_mtime)


def _activation_counts() -> dict[str, int]:
    """Count gating/emotion activations in the newest project log directory.

    最新のプロジェクトログから, ゲーティング・感情の作動回数を数える.

    Returns:
        dict[str, int]: Counts of overrides, gating injections, strong-affect /
            上書き・ゲーティング注入・強い感情の回数
    """
    dirs = sorted(_PROJECT_LOG_DIR.glob("2*/"), key=lambda p: p.stat().st_mtime)
    if not dirs:
        return {"override": 0, "gating_inject": 0, "strong_affect": 0}
    latest = dirs[-1]
    override = gating = strong = 0
    for log_file in latest.glob("kanolab*.log"):
        text = log_file.read_text(encoding="utf-8", errors="ignore")
        override += text.count("反射的に上書き")
        gating += text.count("意思決定への影響")
        strong += text.count("発言・選択ににじみ出る")
    return {"override": override, "gating_inject": gating, "strong_affect": strong}


def _aggregate(games: list[dict]) -> dict:
    """Aggregate per-game metrics for one condition.

    1条件分のゲームごとのメトリクスを集計する.

    Args:
        games (list[dict]): Per-game metric dicts / ゲームごとのメトリクス

    Returns:
        dict: Aggregated metrics / 集計したメトリクス
    """
    ok = [g for g in games if g.get("parsed")]
    n = len(ok)
    vt = sum(g["vote_total"] for g in ok)
    dt = sum(g["divine_total"] for g in ok)
    return {
        "games": len(games),
        "parsed": n,
        "village_win_rate": (sum(g["village_won"] for g in ok) / n) if n else 0.0,
        "vote_total": vt,
        "vote_hit_wolfside": sum(g["vote_hit_wolfside"] for g in ok),
        "vote_acc_wolfside": (sum(g["vote_hit_wolfside"] for g in ok) / vt) if vt else 0.0,
        "vote_hit_werewolf": sum(g["vote_hit_werewolf"] for g in ok),
        "divine_total": dt,
        "divine_hit_werewolf": sum(g["divine_hit_werewolf"] for g in ok),
        "divine_acc_werewolf": (sum(g["divine_hit_werewolf"] for g in ok) / dt) if dt else 0.0,
        "override_total": sum(g.get("override", 0) for g in ok),
        "gating_inject_total": sum(g.get("gating_inject", 0) for g in ok),
        "strong_affect_total": sum(g.get("strong_affect", 0) for g in ok),
    }


def main() -> None:
    """Run the ablation and write aggregated results.

    要因除去を実行し, 集計結果を書き出す.
    """
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    selected = sys.argv[2:] or list(_CONDITIONS)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d%H%M%S")
    tag = f"_{_RESULTS_TAG}" if _RESULTS_TAG else ""
    results_path = _RESULTS_DIR / f"ablation_{stamp}{tag}.json"

    all_results: dict[str, dict] = {}
    for condition in selected:
        growth = _CONDITIONS[condition]
        config_path = _write_config(condition, growth)
        print(f"\n=== 条件 {condition} ({k}ゲーム) growth={growth} ===", flush=True)
        games: list[dict] = []
        for g in range(1, k + 1):
            before = {p.name for p in _SERVER_JSON_DIR.glob("*.json")}
            game_log = _TMP_DIR / f"agents_{condition}_{g}.log"
            t0 = time.time()
            finished = _run_one_game(config_path, game_log)
            record = _newest_record(before)
            entry: dict = {"index": g, "finished": finished, "secs": round(time.time() - t0)}
            if record is not None:
                metrics = parse_game(record)
                entry.update(metrics)
                entry.update(_activation_counts())
                entry["parsed"] = True
                print(
                    f"  [{condition} {g}/{k}] {entry['secs']}s win={metrics['win_side']} "
                    f"vote_acc={metrics['vote_hit_wolfside']}/{metrics['vote_total']} "
                    f"override={entry.get('override', 0)}",
                    flush=True,
                )
            else:
                entry["parsed"] = False
                print(f"  [{condition} {g}/{k}] 記録なし(失敗) finished={finished}", flush=True)
            games.append(entry)
            all_results[condition] = {"games_detail": games, "summary": _aggregate(games)}
            results_path.write_text(
                json.dumps(all_results, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )

    print("\n================ 集計 ================", flush=True)
    cols = ("condition", "parsed", "村勝率", "投票的中", "占い的中", "上書き", "ゲート注入")
    print(f"{cols[0]:<12} {cols[1]:>6} {cols[2]:>7} {cols[3]:>9} {cols[4]:>9} {cols[5]:>6} {cols[6]:>9}", flush=True)
    for condition in selected:
        s = all_results[condition]["summary"]
        print(
            f"{condition:<12} {s['parsed']:>6} {s['village_win_rate']:>7.0%} "
            f"{s['vote_acc_wolfside']:>9.0%} {s['divine_acc_werewolf']:>9.0%} "
            f"{s['override_total']:>6} {s['gating_inject_total']:>9}",
            flush=True,
        )
    print(f"\n結果を保存しました: {results_path}", flush=True)


if __name__ == "__main__":
    main()

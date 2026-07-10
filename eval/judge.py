"""LLM judge for human-likeness: persona consistency and logical consistency.

人間らしさのLLMジャッジ: ペルソナ整合と論理整合を採点する.

各ゲーム記録から, エージェントごとに性格プロフィール・役職・全発言を取り出し, 合意した
ルーブリックで判定する. ペルソナ整合は語調・態度・情動表現が性格に合うかを見, 戦略行動の
選択は対象外とする. 論理整合は本人の発言の時間的な一貫性を見, 人狼陣営の戦略的な嘘は
一貫していれば矛盾としない. 判定は手元の gemma を OpenAI 互換エンドポイント経由で呼ぶ
(同一モデルによる自己評価の偏りは留意). 多数の発話列を同時に投げて処理を速める.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gemma-4-31b-it")
_BASE_URL = os.environ.get("EVAL_VLLM_BASE_URL", "http://localhost:8000/v1")
_CONCURRENCY = 10
_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_SERVER_LOG = Path("/diskthalys/ssd14ta/kmunaoka/aiwolf/aiwolf-nlp-server/log")
# 条件名と, その対局記録が入るサーバ側ディレクトリ
_CONDITION_DIRS = {
    "full": "json_w0",
    "no_emotion": "json_w1",
    "no_reading": "json_w2",
    "no_gating": "json_w3",
}
_PERSONA_RE = re.compile(r"ペルソナ整合[:：]\s*([1-5])")  # noqa: RUF001
_LOGIC_RE = re.compile(r"論理整合[:：]\s*([1-5])")  # noqa: RUF001

_RUBRIC = (
    "あなたは人狼ゲームの発話評価者です。あるエージェントの【性格プロフィール】【役職】【全発言】を読み、"
    "次の2軸をそれぞれ1〜5点で採点してください。\n"
    "①ペルソナ整合: 発話の語調・態度・情動表現が性格に合うか。"
    "CO・占い対抗・名指し・投票などの戦略行動の選択は評価に含めない"
    "(控えめな性格でも役職上必要なら行ってよく、控えめさは口調で判定する)。\n"
    "②論理整合: 本人の発言が時間を通じて矛盾しないか。"
    "人狼・狂人の戦略的な嘘(偽CO・偽占い結果)は、一貫していれば矛盾とみなさない。"
    "自分の過去の主張の翻し・占い結果の言い換え・実際には起きていない出来事への言及のみを矛盾とする。\n"
    "次の形式で厳密に出力してください。\n"
    "ペルソナ整合: <1-5の整数>\n"
    "論理整合: <1-5の整数>\n"
    "根拠: <簡潔に、特に減点理由>"
)

_client = OpenAI(base_url=_BASE_URL, api_key="dummy")


def _per_agent(game_path: Path) -> list[dict]:
    """Extract per-agent profile, role, and utterances from a game record.

    ゲーム記録から, エージェントごとの性格・役職・全発言を取り出す.

    Args:
        game_path (Path): Path to the server game-record JSON / 記録 JSON のパス

    Returns:
        list[dict]: One record per agent / エージェントごとの記録
    """
    data = json.loads(game_path.read_text(encoding="utf-8"))
    profiles: dict[str, str] = {}
    roles: dict[str, str] = {}
    utterances: dict[str, list[str]] = {}
    for entry in data["entries"]:
        request = json.loads(entry["request"])
        if request["request"] == "INITIALIZE":
            info = request["info"]
            agent = info["agent"]
            profiles[agent] = info.get("profile", "")
            roles[agent] = (info.get("role_map") or {}).get(agent, "")
        elif request["request"] == "TALK" and entry.get("response"):
            utterances.setdefault(entry["agent"], []).append(entry["response"])
    return [
        {
            "agent": agent,
            "profile": profiles[agent],
            "role": roles.get(agent, ""),
            "utterances": utterances.get(agent, []),
        }
        for agent in profiles
    ]


def _judge(profile: str, role: str, utterances: list[str]) -> dict:
    """Score one agent's utterances on persona and logical consistency.

    あるエージェントの発言列を, ペルソナ整合と論理整合で採点する.

    Args:
        profile (str): The agent's persona profile / 性格プロフィール
        role (str): The agent's role / 役職
        utterances (list[str]): The agent's utterances in order / 順序付きの全発言

    Returns:
        dict: Scores and the raw judge text / スコアと判定本文
    """
    talk = "\n".join(f"{i}. {t}" for i, t in enumerate(utterances, 1))
    prompt = f"{_RUBRIC}\n\n【性格プロフィール】\n{profile}\n\n【役職】{role}\n\n【全発言】\n{talk}"
    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        text = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        return {"persona": None, "logic": None, "text": f"ERROR: {exc}"}
    persona = _PERSONA_RE.search(text)
    logic = _LOGIC_RE.search(text)
    return {
        "persona": int(persona.group(1)) if persona else None,
        "logic": int(logic.group(1)) if logic else None,
        "text": text,
    }


def _mean(values: list[int]) -> float:
    """Return the mean of a list, or 0 for empty.

    リストの平均を返す. 空なら 0.

    Args:
        values (list[int]): Numbers to average / 平均する数値

    Returns:
        float: The mean / 平均
    """
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    """Judge all conditions and report mean persona and logical consistency.

    全条件を判定し, ペルソナ整合と論理整合の平均を報告する.
    """
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tasks: list[tuple[str, dict]] = []
    for condition, subdir in _CONDITION_DIRS.items():
        games = sorted((_SERVER_LOG / subdir).glob("*.json"))
        if sample:
            games = games[:sample]
        for game in games:
            tasks.extend((condition, rec) for rec in _per_agent(game) if rec["utterances"])
    print(f"判定対象: {len(tasks)} エージェント発言列 (並列度 {_CONCURRENCY})", flush=True)

    def run(task: tuple[str, dict]) -> dict:
        condition, rec = task
        verdict = _judge(rec["profile"], rec["role"], rec["utterances"])
        return {"condition": condition, "agent": rec["agent"], "role": rec["role"], **verdict}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        results = list(pool.map(run, tasks))
    print(f"判定完了: {round(time.time() - t0)}秒", flush=True)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _RESULTS_DIR / "humanlike_judge.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n================ 人間らしさ判定 ================")
    print(f"{'condition':<12}{'N':>5}{'ペルソナ整合':>14}{'論理整合':>12}{'解析失敗':>10}")
    for condition in _CONDITION_DIRS:
        rows = [r for r in results if r["condition"] == condition]
        persona = [r["persona"] for r in rows if r["persona"] is not None]
        logic = [r["logic"] for r in rows if r["logic"] is not None]
        failed = sum(1 for r in rows if r["persona"] is None or r["logic"] is None)
        print(f"{condition:<12}{len(rows):>5}{_mean(persona):>13.2f}{_mean(logic):>12.2f}{failed:>10}")
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()

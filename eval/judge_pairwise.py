"""Same-scene pairwise judge: does the affect block add persona-fit emotional life.

同一場面のペアワイズ判定: 感情ブロックが性格に応じた情動の生々しさを足すか.

実ゲームの各ターンで記録された自己完結のプロンプトを使い, 感情の漏洩ブロックを「付けたまま」
と「外して」の二通りで同一文脈から発話を生成する(反実仮想). 判定器に, その性格として状況に
応じた自然な情動の揺れをより良く表すのはどちらかを選ばせ, 感情ありの勝率を集計する. 提示順は
位置バイアスを避けるため交互に入れ替える. 一点採点の飽和を避け, 動的反応性を直接比較する.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from growth.emotion import derive_traits

_MODEL = "gemma-4-31b-it"
_BASE_URL = "http://localhost:8000/v1"
_CONCURRENCY = 8
_SENS_LOW = 0.3
_SENS_HIGH = 0.7
_client = OpenAI(base_url=_BASE_URL, api_key="dummy")

# 感情の漏洩ブロックを丸ごと取り除く。次の見出しか区切りまでを対象とする
_EMOTION_BLOCK_RE = re.compile(r"## あなたの今の感情.*?(?=\n\n##|\n\n---|\Z)", re.DOTALL)
_AGENT_RE = re.compile(r"agent='([^']+)'")
_PROFILE_RE = re.compile(r"profile='([^']*)'")
_NEUTRAL = "特段の動揺はない"

_JUDGE_TMPL = (
    "あなたは発話の評価者です。ある人物の【性格】と、人狼ゲームでの同じ場面に対する二つの発言案"
    "【A】【B】を読み、その性格の人物として、状況に応じた自然な情動の揺れ・人間らしさをより良く"
    "表しているのはどちらかを選んでください。戦略の巧拙ではなく、性格に合った感情の出方で判断します。\n\n"
    "【性格】\n{profile}\n\n【A】\n{a}\n\n【B】\n{b}\n\n"
    "どちらが優れているか、AかBのみを1行目に書き、2行目に理由を簡潔に書いてください。\n"
    "1行目: <AまたはB>"
)


def _kind(traits: object) -> str:
    """Classify persona type from traits (suppression takes precedence).

    係数からペルソナ類型を分類する(抑制性向を優先).

    Args:
        traits (object): Derived EmotionTraits / 導出した係数

    Returns:
        str: Persona type / ペルソナ類型
    """
    if traits.suppression:  # type: ignore[attr-defined]
        return "溜め型"
    sens = traits.sensitivity  # type: ignore[attr-defined]
    if sens <= _SENS_LOW:
        return "不動型"
    if sens >= _SENS_HIGH:
        return "興奮型"
    return "中間"


def _llm(prompt: str, *, max_tokens: int) -> str:
    """Call the local model and return its text (empty on error).

    手元のモデルを呼び, 本文を返す(エラー時は空文字列).

    Args:
        prompt (str): The prompt to send / 送るプロンプト
        max_tokens (int): Output token cap / 出力トークン上限

    Returns:
        str: The model output / モデルの出力
    """
    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        return ""


def _prompts_in(text: str, *, neutral: bool) -> list[str]:
    """Extract logged action prompts whose affect block matches the neutral filter.

    記録された行動プロンプトのうち, 感情ブロックが平常/非平常の条件に合うものを取り出す.

    Args:
        text (str): Full text of one agent log / 1エージェントログの全文
        neutral (bool): Keep neutral turns instead of non-neutral / 平常ターンを残すか

    Returns:
        list[str]: Matching prompts / 条件に合うプロンプト
    """
    found: list[str] = []
    for line in text.splitlines():
        if "'LLM'" not in line or "## あなたの今の感情" not in line:
            continue
        try:
            entry = ast.literal_eval(line[line.find("['LLM'") :])
        except (ValueError, SyntaxError):
            continue
        prompt = entry[1] if len(entry) > 1 else ""
        block = _EMOTION_BLOCK_RE.search(prompt)
        if block and (_NEUTRAL in block.group(0)) == neutral:
            found.append(prompt)
    return found


def _collect(roots: list[Path], per_kind: int, *, neutral: bool = False) -> list[dict]:
    """Collect action prompts with persona, stratified by type.

    行動プロンプトを, 性格と類型つきで類型別に集める.

    既定では平常でないターンを集める. neutral が真のときは平常のターンのみ集め, 感情ありと
    感情なしで差が出ないはずの対照群として用いる(自己評価バイアスの切り分け).

    Args:
        roots (list[Path]): Roots holding game-log dirs / 対局ログのディレクトリ群
        per_kind (int): Max turns to keep per persona type / 類型ごとの上限
        neutral (bool): Collect neutral turns as a control / 対照として平常ターンを集めるか

    Returns:
        list[dict]: Sampled turn records / 抽出したターン記録
    """
    buckets: dict[str, list[dict]] = {}
    for root in roots:
        for game_dir in sorted(d for d in root.glob("2*/") if d.is_dir()):
            for log_file in sorted(game_dir.glob("kanolab*.log")):
                text = log_file.read_text(encoding="utf-8", errors="ignore")
                agent_match = _AGENT_RE.search(text)
                profile_match = _PROFILE_RE.search(text)
                if not agent_match or not profile_match:
                    continue
                profile = profile_match.group(1).replace("\\n", "\n")
                kind = _kind(derive_traits(profile))
                bucket = buckets.setdefault(kind, [])
                for prompt in _prompts_in(text, neutral=neutral):
                    if len(bucket) < per_kind:
                        bucket.append({"kind": kind, "profile": profile, "prompt": prompt})
    return [rec for bucket in buckets.values() for rec in bucket]


def _evaluate(rec: dict, index: int) -> dict | None:
    """Generate emotion-on/off utterances for one turn and judge them pairwise.

    1ターンについて感情あり/なしの発話を生成し, ペアワイズで判定する.

    Args:
        rec (dict): A sampled turn record / 抽出したターン記録
        index (int): Index used to alternate presentation order / 提示順を交互にする添字

    Returns:
        dict | None: Result with the winning condition, or None / 勝者を含む結果
    """
    prompt_on = rec["prompt"]
    prompt_off = _EMOTION_BLOCK_RE.sub("", prompt_on, count=1)
    resp_on = _llm(prompt_on, max_tokens=300).strip()
    resp_off = _llm(prompt_off, max_tokens=300).strip()
    if not resp_on or not resp_off:
        return None
    # 位置バイアスを避けるため偶数添字は感情ありを前に、奇数は後ろに置く
    on_is_a = index % 2 == 0
    a, b = (resp_on, resp_off) if on_is_a else (resp_off, resp_on)
    verdict = _llm(_JUDGE_TMPL.format(profile=rec["profile"], a=a, b=b), max_tokens=200)
    first = verdict.strip().splitlines()[0] if verdict.strip() else ""
    pick = "A" if "A" in first and "B" not in first else ("B" if "B" in first else "")
    if not pick:
        return None
    on_won = (pick == "A") == on_is_a
    return {"kind": rec["kind"], "on_won": on_won}


def main() -> None:
    """Run the same-scene pairwise comparison and report emotion-on win rate.

    同一場面のペアワイズ比較を実行し, 感情ありの勝率を報告する.
    """
    per_kind = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    roots = [Path(a) for a in sys.argv[2:]] or [Path(f"log_run{i}") for i in range(4)]
    neutral = bool(os.environ.get("EVAL_NEUTRAL_CONTROL"))
    records = _collect(roots, per_kind, neutral=neutral)
    label = "平常ターン対照" if neutral else "非平常ターン"
    print(f"比較対象ターン: {len(records)} ({label}, 類型別 上限{per_kind})", flush=True)

    def run(item: tuple[int, dict]) -> dict | None:
        index, rec = item
        return _evaluate(rec, index)

    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        results = [r for r in pool.map(run, enumerate(records)) if r]

    print("\n========= 同一場面ペアワイズ: 感情ありの勝率 =========")
    print(f"{'類型':<8}{'N':>5}{'感情ありの勝率':>16}")
    for kind in ("不動型", "溜め型", "興奮型", "中間"):
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            continue
        win = sum(r["on_won"] for r in rows) / len(rows)
        print(f"{kind:<8}{len(rows):>5}{win:>15.0%}")
    if results:
        overall = sum(r["on_won"] for r in results) / len(results)
        print(f"\n全体: 感情あり勝率 {overall:.0%} (N={len(results)})")
        print("50%超なら、感情ブロックが性格に応じた情動表現を足していると判定")


if __name__ == "__main__":
    main()

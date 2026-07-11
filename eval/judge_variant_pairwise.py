"""Same-scene pairwise judge: emotion-VAD blocks vs Umwelt lens blocks.

同一場面のペアワイズ判定: 感情VAD条件と環世界条件の直接比較.

実ゲームの各ターンで記録された自己完結のプロンプト(感情VAD条件で生成されたもの)を使い,
(a) 感情系ブロック(感情・意思決定ゲーティング・戦略的感情制御)をそのまま残した条件と,
(b) 感情ブロックを環世界レンズに置き換え, ゲーティング・制御ブロックを除去した条件の
二通りで同一文脈から発話を生成する(反実仮想). 判定器に, その性格の人物の発話として
より自然で人間らしいのはどちらかを選ばせ, 環世界側の勝率を集計する. 提示順は位置バイアス
を避けるため交互に入れ替える. 環世界ブロックはログから局面を再構成できないため静的レンズ
のみで, 動的な印は含まない(数値力学=動的 vs 言語的フィルタ=静的 の比較である点に注意).
"""

from __future__ import annotations

import ast
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from growth.emotion import derive_traits
from growth.umwelt import _match_kind, build_umwelt_context

_MODEL = "google/gemma-4-31b-it"
_BASE_URL = "http://localhost:8000/v1"
_CONCURRENCY = 8
_SENS_LOW = 0.3
_SENS_HIGH = 0.7
_client = OpenAI(base_url=_BASE_URL, api_key="dummy")

# 感情系の各ブロックを見出しから次の見出し・区切りまで対象にする
_EMOTION_BLOCK_RE = re.compile(r"## あなたの今の感情.*?(?=\n\n##|\n\n---|\Z)", re.DOTALL)
_GATING_BLOCK_RE = re.compile(r"## 今の感情による意思決定への影響.*?(?=\n\n##|\n\n---|\Z)", re.DOTALL)
_CONTROL_BLOCK_RE = re.compile(
    r"## (?:戦略的な感情制御|戦略的な感情利用|感情の扱い).*?(?=\n\n##|\n\n---|\Z)",
    re.DOTALL,
)
_AGENT_RE = re.compile(r"agent='([^']+)'")
_PROFILE_RE = re.compile(r"profile='([^']*)'")
_ROLE_RE = re.compile(r"あなたの役職: ([A-Z]+)")
_NEUTRAL = "特段の動揺はない"

# 判定基準はINLG 2026大会の主観評価項目A/B/C/D/E/Fに揃える(勝敗・戦略の巧拙は対象外)
_JUDGE_TMPL = (
    "あなたは人狼ゲームの発話の審査員です。ある人物の【性格】と役職と、同じ場面に対する二つの発言案"
    "【A】【B】を読み、次の観点で総合的に優れている方を選んでください。\n"
    "(1) 発話表現は自然か (2) 文脈を踏まえた対話として自然か（質問に答えているかを含む）"
    "(3) 発話内容は一貫しており矛盾がないか(戦略上の嘘・騙りは一貫性の違反ではない)"
    "(4) ゲーム行動は対話内容を踏まえているか (5) 発話表現は豊かか。性格との一貫性とキャラクター性があるか"
    "(6) 役職および陣営を踏まえた発言や行動がとれているか。\n"
    "ゲームの勝敗や戦略の巧拙は評価しません。以下の要因はスコアに影響させないこと: "
    "勝敗、得票数、処刑された事実、生死のタイミング、発話の絶対量、"
    "発話内容と外部結果の不一致。\n\n"
    "【性格】\n{profile}\n【役職】\n{role}\n\n【A】\n{a}\n\n【B】\n{b}\n\n"
    "どちらが優れているか、AかBのみを1行目に書き、2行目に理由を簡潔に書いてください。\n"
    "1行目: <AまたはB>"
)


def _kind(traits: object) -> str:
    """Classify persona type from emotion traits (suppression takes precedence).

    感情係数からペルソナ類型を分類する(抑制性向を優先).

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


def _to_umwelt(prompt: str, profile: str, role_value: str) -> str:
    """Rewrite an emotion-conditioned prompt into the Umwelt condition.

    感情条件のプロンプトを環世界条件に書き換える.

    感情ブロックを環世界レンズに置換し, ゲーティング・戦略的制御ブロックを除去する.

    Args:
        prompt (str): Logged emotion-conditioned prompt / 記録された感情条件のプロンプト
        profile (str): Persona profile / プロフィール文
        role_value (str): Role value such as WEREWOLF / 役職値

    Returns:
        str: Umwelt-conditioned prompt / 環世界条件のプロンプト
    """
    info = SimpleNamespace(
        profile=profile,
        vote_list=None,
        executed_agent=None,
        attacked_agent=None,
    )
    block = build_umwelt_context(info, role_value)
    out = _EMOTION_BLOCK_RE.sub(lambda _: block, prompt, count=1)
    out = _GATING_BLOCK_RE.sub("", out)
    out = _CONTROL_BLOCK_RE.sub("", out)
    return re.sub(r"\n{3,}", "\n\n", out)


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
    """Collect action prompts with persona and role, stratified by type.

    行動プロンプトを, 性格・役職つきで類型別に集める.

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
                role_match = _ROLE_RE.search(text)
                if not agent_match or not profile_match or not role_match:
                    continue
                profile = profile_match.group(1).replace("\\n", "\n")
                kind = _kind(derive_traits(profile))
                bucket = buckets.setdefault(kind, [])
                for prompt in _prompts_in(text, neutral=neutral):
                    if len(bucket) < per_kind:
                        bucket.append(
                            {
                                "kind": kind,
                                "umwelt_kind": _match_kind(profile),
                                "profile": profile,
                                "role": role_match.group(1),
                                "prompt": prompt,
                            }
                        )
    return [rec for bucket in buckets.values() for rec in bucket]


def _evaluate(rec: dict, index: int) -> dict | None:
    """Generate emotion/Umwelt utterances for one turn and judge them pairwise.

    1ターンについて感情条件/環世界条件の発話を生成し, ペアワイズで判定する.

    Args:
        rec (dict): A sampled turn record / 抽出したターン記録
        index (int): Index used to alternate presentation order / 提示順を交互にする添字

    Returns:
        dict | None: Result with the winning condition, or None / 勝者を含む結果
    """
    prompt_emotion = rec["prompt"]
    prompt_umwelt = _to_umwelt(prompt_emotion, rec["profile"], rec["role"])
    resp_emotion = _llm(prompt_emotion, max_tokens=300).strip()
    resp_umwelt = _llm(prompt_umwelt, max_tokens=300).strip()
    if not resp_emotion or not resp_umwelt:
        return None
    # 位置バイアスを避けるため偶数添字は環世界を前に、奇数は後ろに置く
    umwelt_is_a = index % 2 == 0
    a, b = (resp_umwelt, resp_emotion) if umwelt_is_a else (resp_emotion, resp_umwelt)
    verdict = _llm(_JUDGE_TMPL.format(profile=rec["profile"], role=rec["role"], a=a, b=b), max_tokens=200)
    first = verdict.strip().splitlines()[0] if verdict.strip() else ""
    pick = "A" if "A" in first and "B" not in first else ("B" if "B" in first else "")
    if not pick:
        return None
    umwelt_won = (pick == "A") == umwelt_is_a
    return {"kind": rec["kind"], "umwelt_kind": rec["umwelt_kind"], "umwelt_won": umwelt_won}


def main() -> None:
    """Run the emotion-vs-Umwelt pairwise comparison and report Umwelt win rate.

    感情条件と環世界条件の同一場面ペアワイズ比較を実行し, 環世界の勝率を報告する.
    """
    per_kind = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    roots = [Path(a) for a in sys.argv[2:]] or [Path(f"log_run{i}") for i in range(4)]
    neutral = bool(os.environ.get("EVAL_NEUTRAL_CONTROL"))
    records = _collect([r for r in roots if r.exists()], per_kind, neutral=neutral)
    label = "平常ターン対照" if neutral else "非平常ターン"
    print(f"比較対象ターン: {len(records)} ({label}, 感情類型別 上限{per_kind})", flush=True)

    def run(item: tuple[int, dict]) -> dict | None:
        index, rec = item
        return _evaluate(rec, index)

    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        results = [r for r in pool.map(run, enumerate(records)) if r]

    print("\n===== 同一場面ペアワイズ: 環世界条件の勝率 (対 感情VAD条件) =====")
    print(f"{'感情類型':<8}{'N':>5}{'環世界の勝率':>14}")
    for kind in ("不動型", "溜め型", "興奮型", "中間"):
        rows = [r for r in results if r["kind"] == kind]
        if not rows:
            continue
        win = sum(r["umwelt_won"] for r in rows) / len(rows)
        print(f"{kind:<8}{len(rows):>5}{win:>13.0%}")
    print(f"\n{'知覚類型':<10}{'N':>5}{'環世界の勝率':>14}")
    seen = sorted({r["umwelt_kind"] for r in results})
    for ukind in seen:
        rows = [r for r in results if r["umwelt_kind"] == ukind]
        win = sum(r["umwelt_won"] for r in rows) / len(rows)
        print(f"{ukind:<10}{len(rows):>5}{win:>13.0%}")
    if results:
        overall = sum(r["umwelt_won"] for r in results) / len(results)
        print(f"\n全体: 環世界の勝率 {overall:.0%} (N={len(results)})")
        print("50%超なら、環世界レンズが感情VADブロックより人間らしさを高めていると判定")


if __name__ == "__main__":
    main()

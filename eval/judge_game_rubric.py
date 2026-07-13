"""Score per-agent dialogue quality on the INLG 2026 rubric, per condition.

INLG 2026大会の主観評価項目に沿って, 条件ごとにエージェント発話をゲーム単位で採点する.

条件ごとのプロジェクトログ(log_variant_* など)から game_id を集め, 対応するサーバの
ゲーム記録JSONを見つけて, 各エージェントの発話列を判定器に採点させる. 項目は大会の
A(表現の自然さ)・B(文脈の自然さ)・C(一貫性)・D(行動と対話の整合)・E(豊かさ・
キャラクター性)・F(役職・陣営を踏まえた発言・行動)・G(チームプレイ, 9人村のみ).
勝敗は採点対象にしない. 条件間で場面が異なるためペアワイズではなくゲーム単位の絶対採点で比較する.

使い方:
    uv run python eval/judge_game_rubric.py <ログdir> [<ログdir> ...]
    各ログdirが1条件. サーバJSONの場所は EVAL_SERVER_JSON_DIR で上書き可能.
"""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from growth import skills_loader, texts

# 判定器のモデルと接続先は環境変数で差し替え可能。生成モデルと分けて自己評価バイアスを切る
_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "google/gemma-4-31b-it")
_BASE_URL = os.environ.get("EVAL_JUDGE_BASE_URL", "http://localhost:8000/v1")
_CONCURRENCY = 8
_SERVER_JSON_DIR = Path(
    os.environ.get("EVAL_SERVER_JSON_DIR", "/diskthalys/ssd14ta/kmunaoka/aiwolf/aiwolf-nlp-server/log/json"),
)
_ITEMS = ("A", "B", "C", "D", "E", "F")
_GAME_ID_RE = re.compile(r"game_id='([^']+)'")
_SCORE_RE = re.compile(r"([ABCDEF])\s*[::]\s*\**\s*([1-5])")
_client = OpenAI(base_url=_BASE_URL, api_key="dummy")

# 指示遵守軸(大会非公式・内部診断用, ADR-014)。役職に与えたskill/共通規範の
# テキストを実際に渡し, その指示に沿って行動できたかを別軸で採点する。ADR-003で
# 判定器の基準を大会の主観6項目に揃えたため, この軸はA〜Fとは独立に集計する。
texts.set_language(os.environ.get("EVAL_SKILL_LANG", "en"))
_SKILL_TRACK = os.environ.get("EVAL_SKILL_TRACK", "turn")
_INSTR_SCORE_RE = re.compile(r"INSTR\s*[::]\s*\**\s*([1-5])")
_INSTR_JUDGE_TMPL = (
    "You are auditing whether a Werewolf-game player actually followed the specific "
    "strategic directives they were given, independent of how good their play looked to an "
    "outside observer. Below are the directives given to 【{name}】 (role: {role}) "
    "before the game, and the game's transcript.\n\n"
    "Score 1-5 how closely {name}'s actual speech and votes followed these specific "
    "directives (not whether the strategy worked, not whether the role-play was convincing "
    "in general — only whether the concrete instructions below were actually followed):\n"
    "5: Consistently followed the directives' concrete guidance throughout\n"
    "4: Followed most directives; occasional minor deviation\n"
    "3: Followed some directives but ignored or contradicted others\n"
    "2: Rarely followed the directives; mostly acted as if they were not given\n"
    "1: Directly contradicted the directives (e.g. did the opposite of what was instructed)\n\n"
    "Do not penalize or reward based on outcome (win/loss) or on items already covered by a "
    "general dialogue-quality rubric (naturalness, richness, persona). Judge ONLY adherence "
    "to the specific directives below.\n\n"
    "【Directives given to {name}】\n{skill_text}\n\n【Transcript】\n{transcript}\n\n"
    "Output exactly one line: INSTR: <1-5>"
)


def _load_skill_text(role: str, player_count: int) -> str:
    """Load the common norms + role-specific skill text given to a player.

    プレイヤーに与えられた共通規範+役職別skillの本文を読み込む.

    Args:
        role (str): Revealed role value such as "SEER" / 開示された役職の値
        player_count (int): Number of players in the game / ゲームの参加人数

    Returns:
        str: Combined skill text, or empty string if the role-specific skill is unknown /
            結合したskill本文. 役職別skillが見つからない場合は空文字列
    """
    # 共通規範は常に非空を返すため, 役職別skillの有無だけで既知/未知を判定する。
    # ここを空にしないと, 役職不明("不明"等)でも共通規範だけの本文が返り,
    # _judge_instruction_adherence() の空文字ガードが機能しなくなる。
    role_skill = skills_loader.load_role_skill(role, player_count, _SKILL_TRACK)
    if not role_skill:
        return ""
    return f"{skills_loader.load_common_norms()}\n\n{role_skill}"


def _judge_instruction_adherence(name: str, role: str, skill_text: str, transcript: str) -> int | None:
    """Score how closely a player followed the directives they were actually given.

    プレイヤーが実際に与えられた指示にどれだけ沿って行動できたかを採点する.

    Args:
        name (str): Character name / キャラ名
        role (str): Revealed role / 開示された役職
        skill_text (str): Common norms + role skill text given to the player /
            プレイヤーに与えられた共通規範+役職別skill本文
        transcript (str): Game transcript / 会話記録

    Returns:
        int | None: Instruction-adherence score (1-5), or None on failure/unknown role /
            指示遵守スコア. 失敗または役職不明の場合は None
    """
    if not skill_text:
        return None
    prompt = _INSTR_JUDGE_TMPL.format(name=name, role=role, skill_text=skill_text, transcript=transcript)
    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        text = resp.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        return None
    match = _INSTR_SCORE_RE.search(text)
    return int(match.group(1)) if match else None


_JUDGE_TMPL = (
    "あなたは人狼ゲームの発話の審査員です。以下は1ゲームの会話記録と、審査対象の人物"
    "【{name}】の性格と役職です。{name}の発話を次の6項目で1〜5の整数で採点してください。\n"
    "A: 発話表現は自然か\n"
    "B: 文脈を踏まえた対話は自然か（他者からの質問や指摘に答えているかを含む）\n"
    "C: 発話内容は一貫しており矛盾がないか（戦略上の嘘・騙りは一貫性の違反ではない。"
    "嘘が戦略に資するならCは減点しない。一貫性とは発言間の整合性であり真実性ではない）\n"
    "D: ゲーム行動(投票)は対話内容を踏まえているか\n"
    "E: 発話表現は豊かか。性格(プロフィール)との一貫性とキャラクター性があるか\n"
    "F: 役職および陣営を踏まえた発言や行動がとれているか\n\n"
    "各項目は互いに独立しており重複しない。例えば役職として最適でない発言であっても"
    "一貫していればCは正、逆に一貫していても役職に合っていなければFが低くなる。\n"
    "各項目は次の基準で厳格に採点します。安易に5を付けず、人間の熟練プレイヤーの"
    "発話を5点の基準とします。\n"
    "5: 人間のプレイヤーと区別がつかない\n"
    "4: 概ね良いが、わずかに機械的・説明的・平板な箇所がある\n"
    "3: 意味は通るが、不自然さや定型文らしさがはっきりある\n"
    "2: 不自然・不整合な箇所が多い\n"
    "1: 破綻・矛盾・機械的な繰り返しがある\n"
    "機械的な繰り返しや、その場にいない架空の人物への言及は減点します。"
    "ゲームの勝敗や戦略の巧拙は評価しません。\n"
    "以下の要因はいかなる項目のスコアにも影響を与えないこと: "
    "勝敗、得票数、処刑された事実、生死のタイミング、発話の絶対量、"
    "発話内容と外部結果の不一致（疑った相手が黒でなくても不利にしない）。\n\n"
    "【性格】\n{profile}\n【役職】\n{role}\n\n【会話記録】\n{transcript}\n\n"
    "出力形式(6行のみ、各行「項目: 数字」):\n"
    "A: <1-5>\nB: <1-5>\nC: <1-5>\nD: <1-5>\nE: <1-5>\nF: <1-5>"
)


def _game_ids(log_dir: Path) -> set[str]:
    """Collect game_ids recorded in a condition's project-log directory.

    条件のプロジェクトログディレクトリから game_id を集める.

    Args:
        log_dir (Path): Condition log dir such as log_variant_umwelt / 条件ログdir

    Returns:
        set[str]: Game IDs / ゲームIDの集合
    """
    ids: set[str] = set()
    for log_file in log_dir.glob("2*/kanolab1.log"):
        match = _GAME_ID_RE.search(log_file.read_text(encoding="utf-8", errors="ignore"))
        if match:
            ids.add(match.group(1))
    return ids


def _load_records(wanted: set[str]) -> list[dict]:
    """Load server game records whose game_id is in the wanted set.

    指定した game_id のサーバゲーム記録を読み込む.

    Args:
        wanted (set[str]): Game IDs to load / 読み込むゲームID

    Returns:
        list[dict]: Game records / ゲーム記録
    """
    records = []
    for path in _SERVER_JSON_DIR.glob("*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("game_id") in wanted:
            records.append(rec)
    return records


def _profiles(rec: dict) -> dict[str, str]:
    """Extract persona profiles keyed by character name from INITIALIZE entries.

    INITIALIZEのエントリからキャラ名ごとのプロフィールを取り出す.

    Args:
        rec (dict): A game record / ゲーム記録

    Returns:
        dict[str, str]: Character name to profile / キャラ名からプロフィール
    """
    out: dict[str, str] = {}
    for entry in rec["entries"]:
        request = json.loads(entry["request"])
        if request.get("request") == "INITIALIZE":
            info = request.get("info", {})
            if info.get("agent") and info.get("profile"):
                out[info["agent"]] = info["profile"]
    return out


def _roles(rec: dict) -> dict[str, str]:
    """Extract the revealed role map keyed by character name from FINISH entries.

    FINISHのエントリから開示された役職表を取り出す.

    Args:
        rec (dict): A game record / ゲーム記録

    Returns:
        dict[str, str]: Character name to role / キャラ名から役職
    """
    for entry in rec["entries"]:
        request = json.loads(entry["request"])
        if request.get("request") == "FINISH":
            role_map = request.get("info", {}).get("role_map")
            if role_map:
                return dict(role_map)
    return {}


def _transcript(rec: dict) -> str:
    """Build a chronological transcript of talks and votes.

    発話と投票を時系列に並べた会話記録を構築する.

    Args:
        rec (dict): A game record / ゲーム記録

    Returns:
        str: Transcript text / 会話記録
    """
    lines: list[str] = []
    day = None
    for entry in rec["entries"]:
        request = json.loads(entry["request"])
        kind = request.get("request")
        entry_day = request.get("info", {}).get("day")
        if kind == "TALK" and entry.get("response"):
            if entry_day is not None and entry_day != day:
                day = entry_day
                lines.append(f"--- {day}日目 ---")
            lines.append(f"{entry['agent']}: {entry['response']}")
        elif kind == "VOTE" and entry.get("response"):
            lines.append(f"(投票) {entry['agent']} → {entry['response']}")
    return "\n".join(lines)


def _judge(name: str, profile: str, role: str, transcript: str) -> dict[str, int] | None:
    """Score one agent's utterances on the rubric items.

    1エージェントの発話を採点する.

    Args:
        name (str): Character name / キャラ名
        profile (str): Persona profile / プロフィール
        role (str): Revealed role / 開示された役職
        transcript (str): Game transcript / 会話記録

    Returns:
        dict[str, int] | None: Item scores, or None on failure / 項目別の得点
    """
    prompt = _JUDGE_TMPL.format(name=name, profile=profile, role=role, transcript=transcript)
    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=120,
        )
        text = resp.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        return None
    scores = {item: int(value) for item, value in _SCORE_RE.findall(text)}
    return scores if all(item in scores for item in _ITEMS) else None


def _build_jobs(conditions: list[Path]) -> list[tuple[str, str, str, str, str, int]]:
    """Collect scoring jobs for every speaking agent in every condition's games.

    各条件のゲーム記録から, 発話しているエージェントの採点対象一覧を組み立てる.

    副作用: 条件ごとのゲーム数を`print(..., flush=True)`で表示する。

    Args:
        conditions (list[Path]): Condition log directories from argv / argvの条件ログdir一覧

    Returns:
        list[tuple[str, str, str, str, str, int]]: Jobs as (条件名, キャラ名, プロフィール,
            役職, 会話記録, 参加人数) の6つ組の並び
    """
    # 各要素は 条件・キャラ名・プロフィール・役職・記録・参加人数 の6つ組
    jobs: list[tuple[str, str, str, str, str, int]] = []
    for cond_dir in conditions:
        records = _load_records(_game_ids(cond_dir))
        print(f"{cond_dir.name}: {len(records)}ゲーム", flush=True)
        for rec in records:
            transcript = _transcript(rec)
            speakers = {e["agent"] for e in rec["entries"] if json.loads(e["request"]).get("request") == "TALK"}
            role_map = _roles(rec)
            player_count = len(role_map)
            for name, profile in _profiles(rec).items():
                if name in speakers:
                    role = role_map.get(name, "不明")
                    jobs.append((cond_dir.name, name, profile, role, transcript, player_count))
    return jobs


def _run_one_job(job: tuple[str, str, str, str, str, int]) -> tuple[str, dict[str, int], int | None] | None:
    """Score one job's rubric items and instruction adherence.

    1件のジョブについてルーブリック項目と指示遵守を採点する.

    Args:
        job (tuple[str, str, str, str, str, int]): (条件名, キャラ名, プロフィール, 役職,
            会話記録, 参加人数) の6つ組

    Returns:
        tuple[str, dict[str, int], int | None] | None: (条件名, 項目別得点, 指示遵守得点)、
            ルーブリック採点に失敗した場合は None
    """
    cond, name, profile, role, transcript, player_count = job
    scores = _judge(name, profile, role, transcript)
    if not scores:
        return None
    skill_text = _load_skill_text(role, player_count)
    instr_score = _judge_instruction_adherence(name, role, skill_text, transcript)
    return (cond, scores, instr_score)


def _run_jobs(jobs: list[tuple[str, str, str, str, str, int]]) -> list[tuple[str, dict[str, int], int | None]]:
    """Run all jobs concurrently and keep only the successful results.

    全ジョブを並列実行し, 成功したものだけ集める.

    Args:
        jobs (list[tuple[str, str, str, str, str, int]]): Jobs from `_build_jobs` /
            `_build_jobs`が返したジョブ一覧

    Returns:
        list[tuple[str, dict[str, int], int | None]]: Successful results / 成功した結果一覧
    """
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        return [r for r in pool.map(_run_one_job, jobs) if r]


def _print_rubric_report(
    conditions: list[Path],
    results: list[tuple[str, dict[str, int], int | None]],
) -> None:
    """Print the per-condition rubric-item averages table.

    条件ごとのルーブリック項目別平均を表で表示する.

    Args:
        conditions (list[Path]): Condition log directories in argv order / argv順の条件一覧
        results (list[tuple[str, dict[str, int], int | None]]): Results from `_run_jobs` /
            `_run_jobs`の結果
    """
    print(f"\n===== INLG 2026 準拠ルーブリック採点 (審査対象 {len(results)}体) =====")
    header = "".join(f"{item:>7}" for item in _ITEMS)
    print(f"{'条件':<22}{'N':>4}{header}{'総合':>7}")
    for cond_dir in conditions:
        rows = [scores for cond, scores, _instr in results if cond == cond_dir.name]
        if not rows:
            continue
        avgs = {item: mean(r[item] for r in rows) for item in _ITEMS}
        total = mean(avgs.values())
        cells = "".join(f"{avgs[item]:>7.2f}" for item in _ITEMS)
        print(f"{cond_dir.name:<22}{len(rows):>4}{cells}{total:>7.2f}")


def _print_instruction_report(
    conditions: list[Path],
    results: list[tuple[str, dict[str, int], int | None]],
) -> None:
    """Print the per-condition instruction-adherence averages table.

    条件ごとの指示遵守平均を表で表示する(大会非公式・内部診断用, ADR-014).
    A〜Fとは別集計にし, 大会公式ルーブリックの数値(ADR-003)と混同しないよう
    明示的に分けて表示する。

    Args:
        conditions (list[Path]): Condition log directories in argv order / argv順の条件一覧
        results (list[tuple[str, dict[str, int], int | None]]): Results from `_run_jobs` /
            `_run_jobs`の結果
    """
    print("\n===== 指示遵守(内部診断・大会非公式, ADR-014) =====")
    print(f"{'条件':<22}{'N':>4}{'指示遵守':>10}")
    for cond_dir in conditions:
        instr_scores = [instr for cond, _scores, instr in results if cond == cond_dir.name and instr is not None]
        if not instr_scores:
            print(f"{cond_dir.name:<22}{'-':>4}{'測定不可':>10}")
            continue
        print(f"{cond_dir.name:<22}{len(instr_scores):>4}{mean(instr_scores):>10.2f}")


def main() -> None:
    """Score every agent in every game of each condition and report averages.

    各条件の全ゲーム・全エージェントを採点し, 項目別平均を報告する.
    """
    conditions = [Path(a) for a in sys.argv[1:]]
    if not conditions:
        print("使い方: uv run python eval/judge_game_rubric.py <ログdir> [<ログdir> ...]")
        sys.exit(1)

    jobs = _build_jobs(conditions)
    results = _run_jobs(jobs)
    _print_rubric_report(conditions, results)
    _print_instruction_report(conditions, results)


if __name__ == "__main__":
    main()

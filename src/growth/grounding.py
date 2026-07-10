"""Module that builds grounding context injected into prompts.

プロンプトに注入する接地情報を構築するモジュール.

「確定事実」(生死・追放・襲撃・投票・占い/霊媒結果)と「自己スタンス」(役職・自分が得た
情報・自分の過去の発言)をテキスト化し, 幻覚と過同調を抑える土台を作る.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Role, Talk

from growth import texts

_RECENT_OWN_TALKS = 3
# 私的結果(占い/霊媒)を持つ役職。結果ゼロ時は捏造禁止, 結果ありは列挙対象のみに厳密拘束する
_INFO_ROLES = frozenset({"SEER", "MEDIUM"})

# プロフィール書式(「年齢: 28」/「Age: 28」)から年齢を拾う
_AGE_RE = re.compile(r"(?:年齢|Age)\s*[::]\s*(\d+)", re.IGNORECASE)
# 年齢帯の境界(child: 〜12, teen: 13〜19, adult: 20〜59, senior: 60〜)
_AGE_TEEN = 13
_AGE_ADULT = 20
_AGE_SENIOR = 60


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
    t = texts.get()
    lines: list[str] = [t.GROUNDING_DAY_LABEL.format(day=info.day)]
    alive = [k for k, v in info.status_map.items() if _status_value(v) == "ALIVE"]
    dead = [k for k, v in info.status_map.items() if _status_value(v) != "ALIVE"]
    lines.append(t.GROUNDING_ALIVE_LABEL.format(names=", ".join(alive) if alive else t.GROUNDING_ALIVE_NONE))
    if dead:
        lines.append(t.GROUNDING_DEAD_LABEL.format(names=", ".join(dead)))
    if info.executed_agent is not None:
        lines.append(t.GROUNDING_EXECUTED_LABEL.format(agent=info.executed_agent))
    if info.attacked_agent is not None:
        lines.append(t.GROUNDING_ATTACKED_LABEL.format(agent=info.attacked_agent))
    if info.vote_list is not None:
        lines.append(t.GROUNDING_VOTE_LABEL.format(votes=info.vote_list))
    if info.attack_vote_list is not None:
        lines.append(t.GROUNDING_ATTACK_VOTE_LABEL.format(votes=info.attack_vote_list))
    return t.GROUNDING_CONFIRMED_FACTS_HEADER + "\n" + "\n".join(lines)


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
    t = texts.get()
    role_value = str(getattr(role, "value", role))
    lines: list[str] = [t.GROUNDING_ROLE_LABEL.format(role_value=role_value)]
    if known_results:
        lines.append(t.GROUNDING_KNOWN_RESULTS_LABEL.format(results="; ".join(known_results)))
        if role_value in _INFO_ROLES:
            lines.append(t.GROUNDING_RESULTS_STRICT)
    elif role_value in _INFO_ROLES:
        lines.append(t.GROUNDING_NO_RESULTS_YET)
    own_talks = [
        talk.text for talk in talk_history
        if talk.agent == agent_name and talk.text and not talk.skip and not talk.over
    ]
    if own_talks:
        recent = own_talks[-_RECENT_OWN_TALKS:]
        separator = " / "
        quoted = separator.join(f"「{text}」" for text in recent)
        lines.append(t.GROUNDING_OWN_TALKS_LABEL.format(talks=quoted))
        lines.append(t.GROUNDING_NO_REPEAT)
    return t.GROUNDING_SELF_STANCE_HEADER + "\n" + "\n".join(lines)


def get_speech_format_note() -> str:
    """Return the speech-format note for the currently active language.

    有効な言語の発話形式注意書きを返す. agent.py からはこの関数を呼ぶこと.

    Returns:
        str: Speech-format note / 発話形式注意書き
    """
    return texts.get().SPEECH_FORMAT_NOTE


def build_voice_note(profile: str | None) -> str:
    """Build a voice guide from the persona's age and profile.

    プロフィールの年齢から, この人物の声(話し方)のガイドを構築する.
    年齢帯(子供・若者・大人・年配)に応じたfew-shot例文を与え, LLMがスタイルを模倣するようにする.
    指示ベースより例文ベースの方が, モデルの語彙・文体の再現性が高い.
    年齢が読めない場合は共通部分のみ返す.

    Args:
        profile (str | None): Persona profile text / プロフィール文

    Returns:
        str: Voice-guide block (empty if no profile) / 話し方ガイド(プロフィール無しなら空)
    """
    if not profile:
        return ""
    t = texts.get()
    lines = [t.VOICE_NOTE_HEADER]
    age_match = _AGE_RE.search(profile)
    if age_match:
        age = int(age_match.group(1))
        if age < _AGE_TEEN:
            examples = t.VOICE_FEW_SHOT_CHILD
        elif age < _AGE_ADULT:
            examples = t.VOICE_FEW_SHOT_TEEN
        elif age < _AGE_SENIOR:
            examples = t.VOICE_FEW_SHOT_ADULT
        else:
            examples = t.VOICE_FEW_SHOT_SENIOR
        lines.append(examples.format(age=age))
    lines.append(t.VOICE_COMMON)
    return "\n".join(lines)


def build_target_self_exclusion(agent_name: str) -> str:
    """Build a note forbidding the agent from targeting itself.

    投票・占い・護衛・襲撃の対象に自分を選ばないよう促す注意書きを構築する.

    Args:
        agent_name (str): The agent's in-game name / ゲーム内のエージェント名

    Returns:
        str: Self-exclusion note / 自己除外の注意書き
    """
    return texts.get().GROUNDING_TARGET_SELF_EXCLUSION.format(agent_name=agent_name)


def build_divine_guide(
    agent_name: str,
    alive: list[str],
    known_species: dict[str, str],
    talk_history: list[Talk] | None = None,
    seer_claimants: list[str] | None = None,
) -> str:
    """Build a note steering the Seer to divine the most informative grey target.

    占い師に、最も情報価値の高い未確定(灰)の相手を占うよう促す注記を構築する。
    Day 0は情報がないため灰からランダムでよいが、Day 1以降は以下の優先度で占い先を誘導する:
    1. Seer COの対抗者(偽占い師=狂人の可能性、ただし人狼の可能性も残る)
    2. 沈黙・発言回避している者(人狼が潜伏している可能性)
    3. 投票で村人に票を集めた者(人狼が組織票を仕向けた先の周辺)
    4. 上記がなければ灰から選ぶ

    Args:
        agent_name (str): The agent's in-game name / ゲーム内のエージェント名
        alive (list[str]): Names of living agents / 生存エージェント名
        known_species (dict[str, str]): Own-confirmed species / 自分が確定させた種族
        talk_history (list[Talk] | None): Talk history for heuristic targeting /
            占い先選定のヒューリスティックに使う発話履歴
        seer_claimants (list[str] | None): Players who claimed Seer, in order /
            占い師CO者のリスト(順序付き)

    Returns:
        str: Divine guidance note (empty if no grey remains) / 占い誘導の注記(灰が無ければ空)
    """
    grey = [a for a in alive if a != agent_name and a not in known_species]
    if not grey:
        return ""
    t = texts.get()
    # Day 0(情報なし)は単純に灰を列挙するだけ
    if not talk_history or not seer_claimants:
        return t.GROUNDING_DIVINE_GUIDE.format(grey=", ".join(grey))

    # Day 1以降: 発話から優先占い先を推定する
    # 1. 対抗CO者(自分以外のSeer CO者)は偽占い師=狂人or人狼の可能性
    counter_claimants = [c for c in seer_claimants if c != agent_name and c in grey]
    # 2. 発言数が少ない(沈黙気味の)生存者 — 閾値は生存者数に応じて調整
    talk_counts: dict[str, int] = {}
    for talk in talk_history:
        if talk.skip or not talk.text:
            continue
        talk_counts[talk.agent] = talk_counts.get(talk.agent, 0) + 1
    avg_talks = sum(talk_counts.values()) / max(len(talk_counts), 1)
    quiet_threshold = max(1, int(avg_talks * 0.5))
    quiet_players = [a for a in grey if talk_counts.get(a, 0) <= quiet_threshold]
    # 3. 他者を激しく攻撃している者(人狼が注意を逸らすための扇動の可能性)
    aggressive_players: list[str] = []
    attack_keywords = ("wolf", "werewolf", "liar", "lying", "fake", "suspicious", "vote")
    for talk in talk_history:
        if talk.skip or not talk.text or talk.agent == agent_name or talk.agent not in grey:
            continue
        lowered = talk.text.lower()
        attack_count = sum(1 for kw in attack_keywords if kw in lowered)
        if attack_count >= 2 and talk.agent not in aggressive_players:
            aggressive_players.append(talk.agent)
    # 4. 優先度順に候補を組み立てる
    priority: list[str] = []
    for c in counter_claimants:
        if c not in priority:
            priority.append(c)
    for a in aggressive_players:
        if a not in priority:
            priority.append(a)
    for q in quiet_players:
        if q not in priority:
            priority.append(q)
    remaining_grey = [g for g in grey if g not in priority]
    priority.extend(remaining_grey)

    # 注記を構築: 優先候補と理由を明示
    lines = [t.GROUNDING_DIVINE_GUIDE.format(grey=", ".join(grey))]
    if counter_claimants:
        lines.append(t.GROUNDING_DIVINE_PRIORITY_COUNTER.format(
            names=", ".join(counter_claimants)))
    if aggressive_players:
        lines.append(t.GROUNDING_DIVINE_PRIORITY_AGGRESSIVE.format(
            names=", ".join(aggressive_players)))
    if quiet_players:
        lines.append(t.GROUNDING_DIVINE_PRIORITY_QUIET.format(
            names=", ".join(quiet_players)))
    if priority[:3] != grey[:3]:
        lines.append(t.GROUNDING_DIVINE_PRIORITY_TOP.format(
            names=", ".join(priority[:3])))
    return "\n".join(lines)


# 占い師COのキーワード(英・日)
_SEER_CO_KEYWORDS_EN = ("i am the seer", "i'm the seer", "i claim seer", "i'm seer")
_SEER_CO_KEYWORDS_JA = ("占い師です", "占い師を名乗", "占い師co", "占い師CO", "私は占い師")


def _detect_seer_claim_order(talk_history: list[Talk]) -> list[str]:
    """Return the list of Seer claimants in the order they first claimed.

    占い師を名乗ったプレイヤーを、名乗り出た順序で返す。

    Args:
        talk_history (list[Talk]): Talk history of the game / ゲームのトーク履歴

    Returns:
        list[str]: Seer claimants in first-claim order / 占い師COの順序
    """
    seen: set[str] = set()
    order: list[str] = []
    for talk in talk_history:
        if talk.skip or not talk.text:
            continue
        lowered = talk.text.lower()
        if any(kw in lowered for kw in _SEER_CO_KEYWORDS_EN) or any(kw in talk.text for kw in _SEER_CO_KEYWORDS_JA):
            if talk.agent not in seen:
                seen.add(talk.agent)
                order.append(talk.agent)
    return order


def build_seer_priority_claimants(talk_history: list[Talk]) -> list[str]:
    """Return the list of Seer claimants in first-claim order.

    占い師CO者のリストを名乗り順で返す。agent.pyから直接呼ぶための薄いラッパー。

    Args:
        talk_history (list[Talk]): Talk history of the game / ゲームのトーク履歴

    Returns:
        list[str]: Seer claimants in first-claim order / 占い師CO者の順序リスト
    """
    return _detect_seer_claim_order(talk_history)


def build_seer_priority(
    talk_history: list[Talk],
    agent_name: str,
    role_value: str,
) -> str:
    """Build a note on Seer credibility when multiple Seer claims exist.

    複数の占い師COがある場合、先制COを真とみなすヒューリスティックを注入する。
    自分が占い師で対抗が出た場合と、自分が占い師以外で複数COがある場合の両方で機能する。

    Args:
        talk_history (list[Talk]): Talk history of the game / ゲームのトーク履歴
        agent_name (str): The agent's in-game name / ゲーム内のエージェント名
        role_value (str): The agent's role value / エージェントの役職値

    Returns:
        str: Seer priority note (empty if no counter-claim) / 占い師優先の注記(対抗が無ければ空)
    """
    claimants = _detect_seer_claim_order(talk_history)
    if len(claimants) < 2:
        return ""
    return texts.get().GROUNDING_SEER_PRIORITY


def build_werewolf_attack_guide(
    talk_history: list[Talk],
    agent_name: str,
) -> str:
    """Build a note guiding the werewolf to avoid attacking its likely ally.

    人狼が自陣営(狂人の可能性が高い相手)を誤襲撃しないよう誘導する注記を構築する。
    自分を直接・間接に支援している相手(偽占い師CO・自分を追及する相手を攻撃・自分を擁護)
    を襲撃から除外するよう促す。

    Args:
        talk_history (list[Talk]): Talk history of the game / ゲームのトーク履歴
        agent_name (str): The werewolf's in-game name / 人狼のゲーム内名

    Returns:
        str: Attack guide note (empty if no likely ally detected) / 襲撃誘導の注記
    """
    seer_claimants = _detect_seer_claim_order(talk_history)
    # 自分以外の占い師CO者は狂人の可能性がある(5人村では人狼は自分だけなので他のCO者は狂人)
    likely_allies: list[str] = []
    for claimant in seer_claimants:
        if claimant != agent_name and claimant not in likely_allies:
            likely_allies.append(claimant)
    # 自分を追及している相手(=敵)を攻撃している第三者も味方の可能性
    my_name_lower = agent_name.lower()
    my_attackers: list[str] = []
    for talk in talk_history:
        if talk.skip or not talk.text or talk.agent == agent_name:
            continue
        lowered = talk.text.lower()
        if my_name_lower in lowered and any(kw in lowered for kw in (
            "wolf", "werewolf", "suspicious", "liar", "lying", "fake", "vote",
        )):
            if talk.agent not in my_attackers:
                my_attackers.append(talk.agent)
    # 自分を攻撃している相手をさらに攻撃している第三者は味方の可能性
    for talk in talk_history:
        if talk.skip or not talk.text or talk.agent == agent_name:
            continue
        lowered = talk.text.lower()
        for attacker in my_attackers:
            attacker_lower = attacker.lower()
            if attacker_lower in lowered and any(kw in lowered for kw in (
                "wolf", "werewolf", "suspicious", "liar", "lying", "fake", "vote",
            )):
                if talk.agent not in likely_allies and talk.agent != agent_name:
                    likely_allies.append(talk.agent)
    if not likely_allies:
        return ""
    return texts.get().GROUNDING_WEREWOLF_ATTACK_GUIDE


def build_medium_guide(
    agent_name: str,
    talk_history: list[Talk],
    known_results: list[str],
    known_species: dict[str, str],
    info: Info | None,
) -> str:
    """Build a note guiding the Medium on when and how to use their results.

    霊媒師に、結果の開示タイミングと占い師主張との照合を促す注記を構築する。
    霊媒結果は処刑されたプレイヤーの種族(人狼/人間)を確定する。
    これを占い師の黒/白判定と照合し、真偽を判定する材料を提供する。

    Args:
        agent_name (str): The medium's in-game name / 霊媒師のゲーム内名
        talk_history (list[Talk]): Talk history / 発話履歴
        known_results (list[str]): Accumulated medium results / 蓄積された霊媒結果
        known_species (dict[str, str]): Confirmed species by target / 確定種族
        info (Info | None): Current game info / 現在のゲーム情報

    Returns:
        str: Medium guidance note (empty if no medium results yet) / 霊媒誘導の注記
    """
    # 霊媒結果がなければ何も注入しない(Day 0や結果が出る前)
    medium_results = [r for r in known_results if r.startswith("Medium")]
    if not medium_results or info is None:
        return ""

    t = texts.get()
    lines: list[str] = [t.GROUNDING_MEDIUM_GUIDE]

    # 処刑されたプレイヤーの種族を特定
    executed = getattr(info, "executed_agent", None)
    if executed and executed in known_species:
        species = known_species[executed]
        if species == "WEREWOLF":
            lines.append(t.GROUNDING_MEDIUM_CONFIRMED_WOLF.format(name=executed))
        else:
            lines.append(t.GROUNDING_MEDIUM_CONFIRMED_HUMAN.format(name=executed))

    # 占い師CO者の主張と照合するよう促す
    seer_claimants = _detect_seer_claim_order(talk_history)
    if seer_claimants:
        lines.append(t.GROUNDING_MEDIUM_CROSSCHECK.format(
            seers=", ".join(seer_claimants)))

    return "\n".join(lines)


def build_guard_guide(
    agent_name: str,
    alive: list[str],
    talk_history: list[Talk],
    seer_claimants: list[str] | None = None,
    known_species: dict[str, str] | None = None,
) -> str:
    """Build a note guiding the Bodyguard to protect the most valuable player.

    騎士に、最も価値の高い生存者を護衛するよう促す注記を構築する。
    優先度: 確定白(自分の占いで白が出た者) > 先制COした占い師 > 対抗COした占い師 >
    霊媒CO者 > 発言が鋭い村人。確定黒は護衛しない。

    Args:
        agent_name (str): The bodyguard's in-game name / 騎士のゲーム内名
        alive (list[str]): Names of living agents / 生存エージェント名
        talk_history (list[Talk]): Talk history / 発話履歴
        seer_claimants (list[str] | None): Seer CO claimants in order / 占い師CO者の順序
        known_species (dict[str, str] | None): Own-confirmed species / 自分が確定させた種族

    Returns:
        str: Guard guidance note / 護衛誘導の注記
    """
    t = texts.get()
    confirmed_humans: list[str] = []
    confirmed_wolves: set[str] = set()
    if known_species:
        confirmed_humans = [a for a in alive if known_species.get(a) == "HUMAN"]
        confirmed_wolves = {a for a in alive if known_species.get(a) == "WEREWOLF"}

    # 護衛候補を優先度順に組み立てる(自分以外の生存者)
    candidates = [a for a in alive if a != agent_name and a not in confirmed_wolves]
    priority: list[str] = []

    # 1. 確定白(最優先 — 人狼が最も襲いたい確定村人)
    for a in confirmed_humans:
        if a in candidates and a not in priority:
            priority.append(a)

    # 2. 先制COした占い師(対抗COがいる場合は先制者が本物の可能性が高い)
    if seer_claimants:
        for c in seer_claimants:
            if c in candidates and c not in priority:
                priority.append(c)

    # 3. 霊媒CO者
    medium_keywords = ("i am the medium", "i'm the medium", "medium result")
    for talk in talk_history:
        if talk.skip or not talk.text:
            continue
        lowered = talk.text.lower()
        if any(kw in lowered for kw in medium_keywords):
            if talk.agent in candidates and talk.agent not in priority:
                priority.append(talk.agent)

    # 4. 残りの候補
    for a in candidates:
        if a not in priority:
            priority.append(a)

    if not priority:
        return ""

    lines = [t.GROUNDING_GUARD_GUIDE]
    if seer_claimants:
        first_seer = seer_claimants[0]
        if first_seer in candidates:
            lines.append(t.GROUNDING_GUARD_PRIORITY_SEER.format(name=first_seer))
    if confirmed_humans:
        lines.append(t.GROUNDING_GUARD_PRIORITY_CONFIRMED.format(
            names=", ".join(confirmed_humans)))
    lines.append(t.GROUNDING_GUARD_PRIORITY_TOP.format(
        names=", ".join(priority[:3])))
    return "\n".join(lines)


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
        texts.get().GROUNDING_ATTENTION,
    ]
    return "\n\n".join(block for block in blocks if block)

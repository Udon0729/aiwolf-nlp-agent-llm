"""Module that deduces the game-theoretic situation from confirmed facts.

確定事実から人狼のゲーム論的な状況を演繹するモジュール(ADR-008 の層0).

勝利条件(人狼生存数 >= 人間生存数で人狼勝ち / 人狼0で市民勝ち)・パリティ・吊り残機・
情報閉包を, LLM を呼ばず算術で計算する純関数群. grounding.py の「確定事実」が状態の
「列挙」であるのに対し, 本モジュールはそこから盤面の「含意」を演繹し, 意思決定プロンプトへ
注入する土台を作る. 数値の典拠はサーバ仕様(doc/ja/logic.md の勝利条件, config.md の村構成).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Role, Setting

from growth import texts

# 種族が人狼の役職(勝利条件の頭数計算に用いる). 狂人は種族が人間なので含めない.
_WEREWOLF_ROLE = "WEREWOLF"
# 人狼陣営(勝敗を共有する役職). 狂人は種族人間だが陣営は人狼側.
_WOLF_SIDE_ROLES = frozenset({"WEREWOLF", "POSSESSED"})
# role_num_map が読めない場合の, 人数からの人狼数フォールバック(reference §3.2, server default_13).
_WOLVES_BY_COUNT = {5: 1, 9: 2, 13: 3}


def _value(obj: object) -> str:
    """列挙体ふうのオブジェクトの文字列値を返す(なければ str 化)."""
    return str(getattr(obj, "value", obj))


def _alive_names(info: Info) -> list[str]:
    """生存中のプレイヤー名を安定順(status_map の順)で返す."""
    return [k for k, v in info.status_map.items() if _value(v) == "ALIVE"]


def _total_werewolves(setting: Setting | None, agent_count: int) -> int:
    """このゲームの人狼の総数を返す(典拠 = role_num_map, 読めなければ人数表)."""
    role_num_map = getattr(setting, "role_num_map", None) if setting else None
    if role_num_map:
        try:
            total = sum(
                int(n) for r, n in dict(role_num_map).items() if _value(r) == _WEREWOLF_ROLE
            )
            if total > 0:
                return total
        except (TypeError, ValueError):
            pass
    return _WOLVES_BY_COUNT.get(agent_count, max(1, agent_count // 4))


def _worst_case_lynch_budget(humans_alive: int, wolves_alive: int) -> int:
    """村側の最悪ケースで, パリティ到達までに村が持つ追放(吊り)の回数.

    最悪ケース = 毎昼の追放が人間に当たり, 毎夜の襲撃も人間を1名奪う(護衛は数えない).
    人狼生存数 >= 人間生存数 で村は敗北するため, その直前までに残る吊り機会を数える.
    5人村(人間4/狼1)では 2, 9人村(人間7/狼2)では 3 を返す.

    Args:
        humans_alive (int): 生存する人間の数 / Living humans
        wolves_alive (int): 生存する人狼の数 / Living werewolves

    Returns:
        int: 残る追放回数 / Remaining lynches
    """
    humans, wolves = humans_alive, wolves_alive
    budget = 0
    while wolves < humans:
        humans -= 1  # 昼の追放(最悪ケース = 人間に当たる)
        budget += 1
        if wolves >= humans:
            break
        humans -= 1  # 夜の襲撃(最悪ケース = 人間が1名減る)
    return budget


@dataclass
class Situation:
    """確定事実から演繹した盤面の状況(ADR-008 の層0の出力).

    Attributes:
        n_alive: 生存者数 / Living count
        total_wolves: このゲームの人狼総数 / Total werewolves in the game
        wolves_alive_max: 生存しうる人狼数の上限 / Upper bound of living werewolves
        humans_alive_min: 生存する人間数の下限 / Lower bound of living humans
        lynch_budget: 村側の最悪ケースの吊り残機 / Worst-case village lynch budget
        confirmed_wolves: 自分が確定させた生存中の黒 / Own-confirmed living blacks
        confirmed_humans: 自分が確定させた生存中の白 / Own-confirmed living whites
        grey: 未確定の生存者(自分以外) / Unconfirmed living others
        closure_wolf: 消去法で人狼と確定する1名(無ければ None) / Forced werewolf by elimination
    """

    n_alive: int
    total_wolves: int
    wolves_alive_max: int
    humans_alive_min: int
    lynch_budget: int
    confirmed_wolves: list[str]
    confirmed_humans: list[str]
    grey: list[str]
    closure_wolf: str | None


def build_situation(
    info: Info,
    setting: Setting | None,
    agent_name: str,
    known_species: dict[str, str],
) -> Situation:
    """確定事実と自分の占い/霊媒結果から盤面の状況を演繹する.

    Args:
        info (Info): 現在のゲーム情報 / Current game info
        setting (Setting | None): ゲーム設定(村構成の典拠) / Game setting
        agent_name (str): 自分のゲーム内の名前 / The agent's in-game name
        known_species (dict[str, str]): 自分が確定させた種族(対象名 -> "WEREWOLF"/"HUMAN") /
            Own-confirmed species by target name

    Returns:
        Situation: 演繹した盤面の状況 / The deduced situation
    """
    alive = _alive_names(info)
    alive_set = set(alive)
    n_alive = len(alive)
    total_wolves = _total_werewolves(setting, len(info.status_map))

    dead = {k for k, v in info.status_map.items() if _value(v) != "ALIVE"}
    known_dead_wolves = sum(
        1 for target, sp in known_species.items() if sp == _WEREWOLF_ROLE and target in dead
    )
    wolves_alive_max = max(0, total_wolves - known_dead_wolves)

    confirmed_wolves = [
        t for t, sp in known_species.items() if sp == _WEREWOLF_ROLE and t in alive_set
    ]
    confirmed_humans = [t for t, sp in known_species.items() if sp == "HUMAN" and t in alive_set]
    grey = [n for n in alive if n != agent_name and n not in known_species]

    humans_alive_min = max(0, n_alive - wolves_alive_max)
    lynch_budget = _worst_case_lynch_budget(humans_alive_min, wolves_alive_max)

    # 情報閉包: 灰に潜む未確定の狼の数. 灰が1名でそこに狼が残るなら, その1名が確定.
    unknown_wolves = max(0, wolves_alive_max - len(confirmed_wolves))
    closure_wolf = grey[0] if (len(grey) == 1 and unknown_wolves >= 1) else None

    return Situation(
        n_alive=n_alive,
        total_wolves=total_wolves,
        wolves_alive_max=wolves_alive_max,
        humans_alive_min=humans_alive_min,
        lynch_budget=lynch_budget,
        confirmed_wolves=confirmed_wolves,
        confirmed_humans=confirmed_humans,
        grey=grey,
        closure_wolf=closure_wolf,
    )


def build_rule_model_block(
    info: Info | None,
    setting: Setting | None,
    role: Role,
    agent_name: str,
    known_species: dict[str, str],
) -> str:
    """盤面の算術(層0)を意思決定プロンプトへ注入するブロックを構築する.

    村側は「残り人狼数・吊り残機・確定/灰・消去法」を, 人狼側は「パリティまでの猶予と
    非公開の徹底」を, それぞれの視点で言語化する. 状態が無ければ空文字列を返す.

    Args:
        info (Info | None): 現在のゲーム情報 / Current game info
        setting (Setting | None): ゲーム設定 / Game setting
        role (Role): 自分の役職 / The agent's role
        agent_name (str): 自分のゲーム内の名前 / The agent's in-game name
        known_species (dict[str, str]): 自分が確定させた種族 / Own-confirmed species

    Returns:
        str: 注入ブロック. 状態が無ければ空文字列 / The block, empty if no state
    """
    if info is None or not info.status_map:
        return ""
    sit = build_situation(info, setting, agent_name, known_species)
    if sit.n_alive <= 0:
        return ""
    t = texts.get()
    role_value = _value(role)
    lines: list[str] = [t.RULE_MODEL_HEADER, t.RULE_MODEL_WIN_CONDITION]

    if role_value in _WOLF_SIDE_ROLES:
        lines.append(t.RULE_MODEL_WOLF_LINE.format(k=sit.lynch_budget))
        if role_value == "POSSESSED":
            lines.append(t.RULE_MODEL_POSSESSED_LINE)
    else:
        lines.append(t.RULE_MODEL_WOLVES_REMAINING.format(w=sit.wolves_alive_max, n=sit.n_alive))
        lines.append(t.RULE_MODEL_LYNCH_BUDGET.format(k=sit.lynch_budget))
        if sit.lynch_budget <= 1:
            lines.append(t.RULE_MODEL_MUST_LYNCH_NOW)
        if sit.confirmed_wolves:
            lines.append(t.RULE_MODEL_CONFIRMED_WOLF.format(names=", ".join(sit.confirmed_wolves)))
        if sit.confirmed_humans:
            lines.append(t.RULE_MODEL_CONFIRMED_HUMAN.format(names=", ".join(sit.confirmed_humans)))
        # 灰/消去法の言語化は, 自分に確定情報(占い/霊媒)があり, かつ未確定の狼が実際に
        # 残っている(unknown_wolves>=1)ときだけ意味を持つ. 全狼を自分で確定済みなら灰に
        # 狼はおらず, 灰への誘導や消去法の断定は確定狼への集中を妨げる誤誘導になる.
        unknown_wolves = max(0, sit.wolves_alive_max - len(sit.confirmed_wolves))
        if known_species and sit.grey and unknown_wolves >= 1:
            lines.append(t.RULE_MODEL_GREY.format(names=", ".join(sit.grey)))
        if sit.closure_wolf is not None and unknown_wolves >= 1:
            lines.append(t.RULE_MODEL_CLOSURE.format(name=sit.closure_wolf))

    lines.append(t.RULE_MODEL_USAGE)
    return "\n".join(lines)

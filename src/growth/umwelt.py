"""Build Uexkull-inspired subjective-world prompt blocks.

ユクスキュルの環世界論に着想を得た, 主観的世界のプロンプトブロックを構築する.

ここではプロフィールを数値的な感情係数に変換しない. 代わりに, その人物にとって
どの手がかりが意味担体になり, どう知覚され, どんな行動可能性を誘うかを
言語的な知覚-行動フィルタとして与える. さらに機能環(知覚→行動→環境→知覚)を
実装するため, いまの局面で成立している印(自分への言及・票・死亡・場の沈黙)を
類型ごとのレンズで意味づけして注入する. 同じ出来事でも, 人物像が違えば
違う印として知覚される. 文言は texts_ja / texts_en に集約している.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from growth import texts

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Talk

# 印の検出に使う直近トークの観測窓
_RECENT_TALKS = 15
# 沈黙の印を判定するのに必要な最小トーク数と, 非skip発言の許容比率(1/3以下で沈黙)
_SILENCE_MIN_TALKS = 5
_SILENCE_ACTIVE_RATIO = 3


def _match_kind(profile: str) -> str:
    """Pick the perceptual kind with the most keyword matches.

    キーワードの最多マッチで知覚類型を選ぶ(同数なら先に定義された類型).

    Args:
        profile (str): Persona profile text / プロフィール文

    Returns:
        str: Kind key such as "cautious" / 類型キー
    """
    t = texts.get()
    lowered = profile.lower()
    best_kind, best_count = "default", 0
    for kind, words in t.UMWELT_KIND_WORDS:
        count = sum(1 for word in words if word.lower() in lowered)
        if count > best_count:
            best_kind, best_count = kind, count
    return best_kind


@lru_cache(maxsize=64)
def _lens_block(profile: str, role_value: str, lang: str) -> tuple[str, str]:
    """Build the static lens part of the block (cached per profile/role/language).

    ブロックの静的なレンズ部分を構築する(プロフィール・役職・言語ごとにキャッシュ).

    Args:
        profile (str): Persona profile text / プロフィール文
        role_value (str): Role value such as WEREWOLF / 役職値
        lang (str): Active language code (cache key) / 言語コード(キャッシュのキー)

    Returns:
        tuple[str, str]: (kind, lens block) / 類型キーと静的レンズ部分
    """
    del lang  # キャッシュのキーとしてのみ用いる(texts はグローバルな言語状態を持つ)
    t = texts.get()
    kind = _match_kind(profile)
    lens = t.UMWELT_LENSES.get(kind, t.UMWELT_LENSES["default"])
    role_lens = t.UMWELT_ROLE_LENSES.get(role_value, t.UMWELT_ROLE_LENSES["default"])
    return kind, t.UMWELT_TMPL_LENS.format(role_lens=role_lens, **lens)


def _present_signs(info: Info, talk_history: list[Talk], name: str) -> set[str]:
    """Detect which environmental signs hold in the current scene.

    いまの局面で成立している環境の印を検出する.

    素材は, 直近トークでの自分への言及, 自分に入った票, 直近の死亡,
    場の沈黙(skipの多さ)の4つ. どれが「見える」かは類型側で選ぶ.

    Args:
        info (Info): Current game info / 現在のゲーム情報
        talk_history (list[Talk]): Talk history / トーク履歴
        name (str): The agent's in-game name / ゲーム内のエージェント名

    Returns:
        set[str]: Present sign keys / 成立している印のキー集合
    """
    signs: set[str] = set()
    recent = talk_history[-_RECENT_TALKS:]
    others = [talk for talk in recent if talk.agent != name and not talk.skip]
    if name and any(name in talk.text for talk in others):
        signs.add("mentions")
    if any(vote.target == name for vote in (info.vote_list or [])):
        signs.add("votes")
    if info.executed_agent is not None or info.attacked_agent is not None:
        signs.add("death")
    if len(recent) >= _SILENCE_MIN_TALKS:
        active = sum(1 for talk in recent if not talk.skip)
        if active * _SILENCE_ACTIVE_RATIO <= len(recent):
            signs.add("silence")
    return signs


def _sign_line(kind: str, info: Info, talk_history: list[Talk], name: str) -> str:
    """Verbalise the first sign this kind perceives (empty if none).

    その類型のレンズで最初に知覚される印を1行にする(なければ空).

    類型の優先リストにない印はその環世界に存在しない(知覚されない).

    Args:
        kind (str): Kind key / 類型キー
        info (Info): Current game info / 現在のゲーム情報
        talk_history (list[Talk]): Talk history / トーク履歴
        name (str): The agent's in-game name / ゲーム内のエージェント名

    Returns:
        str: One sign line or "" / 印の1行(なければ空文字列)
    """
    t = texts.get()
    present = _present_signs(info, talk_history, name)
    if not present:
        return ""
    for sign_key, phrase in t.UMWELT_SIGNS.get(kind, t.UMWELT_SIGNS["default"]):
        if sign_key in present:
            return t.UMWELT_SIGN_LINE.format(sign=phrase)
    return ""


def build_umwelt_context(
    info: Info | None,
    role_value: str,
    talk_history: list[Talk] | None = None,
    name: str = "",
    *,
    reads_others: bool = True,
) -> str:
    """Build the persona-specific subjective-world prompt block.

    プロフィールごとの環世界ブロックを構築する.

    静的なレンズ(意味担体・知覚世界・作用世界・盲点・補正・役職)に,
    いまの局面の印(あれば)と機能環の読み直し指示を加える. 発話や対象選択など
    他者を読む場面では, レンズで扱う論点を選ぶ持ちかけの行も付ける.

    Args:
        info (Info | None): Game info holding the profile and sign sources /
            プロフィールと印の素材を持つゲーム情報
        role_value (str): Role value such as WEREWOLF / 役職値
        talk_history (list[Talk] | None): Talk history / トーク履歴
        name (str): The agent's in-game name / ゲーム内のエージェント名
        reads_others (bool): Whether the request reads other players / 他者を読むリクエストか

    Returns:
        str: Prompt block / プロンプトブロック
    """
    t = texts.get()
    profile = (info.profile if info is not None else None) or ""
    kind, lens_block = _lens_block(profile, role_value, texts.get_language())
    lines = [lens_block]
    if info is not None:
        sign = _sign_line(kind, info, talk_history or [], name)
        if sign:
            lines.append(sign)
    lines.append(t.UMWELT_CIRCLE_LINE)
    if reads_others:
        lines.append(t.UMWELT_ENGAGE_LINE)
    lines.append(t.UMWELT_TMPL_TAIL)
    return "\n".join(lines)

"""Module that tracks per-player observations across turns.

ターンをまたいだプレイヤーごとの観察を蓄積するモジュール.

テル読み(reading.py)は毎ターン直近の発言しか参照しないため, 一貫性の変化・投票パターン・
主張の矛盾といった時系列の情報が失われる. このモジュールは LLM を呼ばず, 純粋な記録として
各プレイヤーの発言履歴・投票先・主張内容を保持し, テル読みに蓄積された文脈を与える.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Talk

from growth import texts

_MAX_UTTERANCES_PER_PLAYER = 5
_MAX_VOTES_PER_PLAYER = 5


@dataclass
class PlayerRecord:
    """Accumulated observations for a single player.

    ひとりのプレイヤーについて蓄積した観察.
    """

    utterances: list[str] = field(default_factory=list[str])
    vote_targets: list[str] = field(default_factory=list[str])
    claims: list[str] = field(default_factory=list[str])


_CO_KEYWORDS_JA = ("占い師", "霊能者", "騎士", "狩人", "人狼", "狂人", "CO", "カミングアウト")
_CO_KEYWORDS_EN = (
    "seer",
    "medium",
    "bodyguard",
    "hunter",
    "werewolf",
    "possessed",
    "coming out",
    "CO",
    "claim",
    "I am the",
)


def _extract_claims(text: str) -> list[str]:
    """Detect role claims or divination/medium result announcements.

    役職の主張や占い・霊媒結果の公表を検出する.

    Args:
        text (str): Utterance text / 発言テキスト

    Returns:
        list[str]: Detected claims / 検出された主張
    """
    claims: list[str] = []
    keywords = _CO_KEYWORDS_JA if texts.get_language() == "ja" else _CO_KEYWORDS_EN
    for kw in keywords:
        if kw.lower() in text.lower():
            claims.append(text)
            break
    return claims


class BeliefTracker:
    """Tracks observations about all players across game turns.

    ゲームのターンをまたいで全プレイヤーの観察を蓄積する.
    """

    def __init__(self, self_name: str) -> None:
        """Initialize the tracker.

        Args:
            self_name (str): The agent's own name / 自分のエージェント名
        """
        self._self_name = self_name
        self._records: dict[str, PlayerRecord] = {}
        self._last_talk_idx: int = -1
        self._last_vote_day: int = -1

    def _get(self, name: str) -> PlayerRecord:
        if name not in self._records:
            self._records[name] = PlayerRecord()
        return self._records[name]

    def update(self, info: Info | None, talk_history: list[Talk]) -> None:
        """Ingest new talks and vote results since last update.

        前回以降の新しい発言と投票結果を取り込む.

        Args:
            info (Info | None): Current game info / 現在のゲーム情報
            talk_history (list[Talk]): Talk history / トーク履歴
        """
        for talk in talk_history:
            if talk.idx <= self._last_talk_idx:
                continue
            if talk.agent == self._self_name or not talk.text or talk.skip or talk.over:
                continue
            rec = self._get(talk.agent)
            rec.utterances.append(talk.text)
            if len(rec.utterances) > _MAX_UTTERANCES_PER_PLAYER:
                rec.utterances = rec.utterances[-_MAX_UTTERANCES_PER_PLAYER:]
            for claim in _extract_claims(talk.text):
                if claim not in rec.claims:
                    rec.claims.append(claim)
        if talk_history:
            self._last_talk_idx = max(t.idx for t in talk_history)

        if info is None:
            return
        if info.day != self._last_vote_day and info.vote_list:
            self._last_vote_day = info.day
            for vote in info.vote_list:
                if vote.agent == self._self_name:
                    continue
                rec = self._get(vote.agent)
                rec.vote_targets.append(vote.target)
                if len(rec.vote_targets) > _MAX_VOTES_PER_PLAYER:
                    rec.vote_targets = rec.vote_targets[-_MAX_VOTES_PER_PLAYER:]

    def build_belief_summary(self, alive: set[str]) -> str:
        """Build a prompt block summarising accumulated observations.

        蓄積された観察をまとめたプロンプトブロックを構築する.

        Args:
            alive (set[str]): Names of currently alive players / 現在生存中のプレイヤー名

        Returns:
            str: Summary block, or empty string if no observations /
                要約ブロック. 観察が無ければ空文字列
        """
        t = texts.get()
        lines: list[str] = []
        for name in sorted(alive):
            if name == self._self_name or name not in self._records:
                continue
            rec = self._records[name]
            if not rec.utterances and not rec.vote_targets and not rec.claims:
                continue
            parts: list[str] = [t.BELIEFS_CLAIM_LINE.format(name=name, claim=claim) for claim in rec.claims]
            parts.extend(t.BELIEFS_VOTE_LINE.format(voter=name, target=target) for target in rec.vote_targets)
            if len(rec.utterances) > 1:
                parts.append(
                    t.BELIEFS_UTTERANCE_SUMMARY.format(
                        name=name,
                        count=len(rec.utterances),
                    )
                )
            if parts:
                lines.extend(parts)
        if not lines:
            return ""
        return t.BELIEFS_HEADER + "\n" + "\n".join(lines)

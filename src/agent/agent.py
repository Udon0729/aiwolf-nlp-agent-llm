"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import threading
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from growth import texts as _growth_texts
from growth.beliefs import BeliefTracker
from growth.emotion import EmotionDynamics, build_strategic_control
from growth.game_model import build_rule_model_block
from growth.gating import build_decision_gating, impulsive_override, salient_pressurers
from growth.grounding import (
    REPEAT_MIN_COUNT,
    build_action_grounding,
    build_divine_guide,
    build_guard_guide,
    build_medium_guide,
    build_repetition_guard,
    build_seer_priority,
    build_seer_priority_claimants,
    build_target_self_exclusion,
    build_voice_note,
    build_werewolf_attack_guide,
    find_repeated_phrase,
    get_speech_format_note,
)
from growth.reading import build_tell_reading
from growth.review import load_calibration, load_lessons, run_post_game_review
from growth.skills_loader import load_common_norms, load_role_skill
from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")

# 先頭の初期化文脈を保持しつつメッセージ履歴を剪定する上限.
# 確定事実と直近発言は各リクエストで再供給されるため, 古い巨大な行動プロンプトは強く落とす.
_MAX_HISTORY_MESSAGES = 4
_KEEP_HEAD_MESSAGES = 2
_SPEECH_PARENTHETICAL_RE = re.compile(r"\([^()\n]{0,120}\)|\uFF08[^\uFF08\uFF09\n]{0,120}\uFF09")
_SPEECH_EMOTE_RE = re.compile(r"[*\uFF0A][^*\uFF0A\n]{1,80}[*\uFF0A]")
_SPEECH_ELLIPSIS_RE = re.compile(r"(?:…+|‥+|\.{3,}|(?:・\s*){3,}|･{3,})")
_SPEECH_DOUBLE_PERIOD_RE = re.compile(r"\.{2,}")
_SPEECH_OVER_ONLY_RE = re.compile(r"\s*over\s*[。.!\uFF01]?\s*", re.IGNORECASE)
_SPEECH_OVER_TOKEN_RE = re.compile(r"(?<![A-Za-z])Over(?![A-Za-z])", re.IGNORECASE)
_SPEECH_SPACES_RE = re.compile(r"[ \t　]{2,}")
_SPEECH_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([、。,.!?\uFF01\uFF1F])")
_SPEECH_REMOVE_CHARS = str.maketrans({
    "「": "",
    "」": "",
    "『": "",
    "』": "",
    "\uFF08": "",
    "\uFF09": "",
    "(": "",
    ")": "",
})

_MAX_SPEECH_COUNTED_CHARS = 125
_SPEECH_LEADING_MENTION_RE = re.compile(r"^(@\S+\s+)(.*)$", re.DOTALL)
_SPEECH_SENTENCE_END_RE = re.compile(r"[.!?。！？](?=\s|$)")  # noqa: RUF001


def _counted_speech_len(text: str) -> int:
    """Count characters the way the server does (spaces between words are not counted).

    サーバと同じ数え方(単語間の空白を除いた文字数)で長さを数える.
    """
    return len(text) - text.count(" ")


def _trim_to_char_budget(text: str, budget: int = _MAX_SPEECH_COUNTED_CHARS) -> str:
    """Trim speech to the server's per-utterance budget at a safe (sentence/word) boundary.

    サーバの1発話文字数上限(空白を除く, 既定125)に収まるよう, 文末→語末の順の無害な境界で
    刈り取る. 超過するとサーバがブロードキャスト時に語中で切り捨て, 相手に届く発話が壊れて
    A/B/C項目を損なうため, その前に自前で丸める. 先頭メンション(@名前)は上限に数えられない
    ため対象から外す. budget はサーバ設定(base_length)や remain_length で上書きされる.

    Args:
        text (str): Sanitized utterance / 整形済みの発話
        budget (int): Server per-utterance char budget (spaces excluded) / 空白非算入の文字数上限

    Returns:
        str: Utterance within the character budget / 文字数上限内に収めた発話
    """
    mention = ""
    body = text
    m = _SPEECH_LEADING_MENTION_RE.match(text)
    if m:
        mention, body = m.group(1), m.group(2)
    if _counted_speech_len(body) <= budget:
        return text
    counted = 0
    cut = len(body)
    for i, ch in enumerate(body):
        if ch != " ":
            counted += 1
        if counted > budget:
            cut = i
            break
    head = body[:cut]
    ends = list(_SPEECH_SENTENCE_END_RE.finditer(head))
    if ends:
        head = head[: ends[-1].end()]
    else:
        space = head.rfind(" ")
        if space > 0:
            head = head[:space]
    return f"{mention}{head}".strip()


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.track = str(config.get("agent", {}).get("track", "turn"))
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role
        # グループチャット方式
        self.in_talk_phase = False
        self.in_whisper_phase = False

        self.sent_talk_count: int = 0
        self.sent_whisper_count: int = 0
        self.llm_model: BaseChatModel | None = None
        self.llm_message_history: list[BaseMessage] = []
        # growth層で参照する、自分が得た占い・霊媒結果のリスト
        self.known_results: list[str] = []
        # growth層(game_model, 層0)が参照する、自分が確定させた種族: 対象名 -> "WEREWOLF"/"HUMAN"
        self._known_species: dict[str, str] = {}
        # growth層: 性格(プロフィール)で個体化された感情の力学(ゲーム開始時に構築)
        self.emotion: EmotionDynamics | None = None
        # growth層: プレイヤーごとの観察蓄積(INITIALIZE時に構築)  # noqa: ERA001
        self.belief_tracker: BeliefTracker | None = None
        # 言語別プロンプトテンプレート(initialize()で設定)
        self._prompts: dict[str, str] | None = None

        load_dotenv(Path(__file__).parent.joinpath("./../../config/.env"))

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            self.info = packet.info
            self._record_known_results(packet.info)
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)

        # グループチャット方式
        if packet.new_talk:
            self.talk_history.append(packet.new_talk)
            self.on_talk_received(packet.new_talk)
        if packet.new_whisper:
            self.whisper_history.append(packet.new_whisper)
            self.on_whisper_received(packet.new_whisper)

        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
            self.llm_message_history: list[BaseMessage] = []
            self.known_results = []
            self._known_species = {}
            self.emotion = None
            self.belief_tracker = None
        self._update_emotion()
        self._update_beliefs()
        self.agent_logger.logger.debug(packet)

    def _record_known_results(self, info: Info) -> None:
        """Accumulate private results (divine/medium) obtained by the agent.

        エージェントが得た占い・霊媒結果を蓄積する.

        Args:
            info (Info): Game info that may contain results / 結果を含みうるゲーム情報
        """
        t = _growth_texts.get()
        for label, value in ((t.RESULT_LABEL_DIVINE, info.divine_result), (t.RESULT_LABEL_MEDIUM, info.medium_result)):
            if value is not None:
                entry = f"{label}: {value}"
                if entry not in self.known_results:
                    self.known_results.append(entry)
                # 層0(game_model)が参照する構造化した確定種族: 対象名 -> "WEREWOLF"/"HUMAN"
                target = getattr(value, "target", None)
                species = str(getattr(getattr(value, "result", None), "value", "")) or None
                if target and species in ("WEREWOLF", "HUMAN"):
                    self._known_species[target] = species

    def _growth_config(self, section: str) -> Any:  # noqa: ANN401
        """Return a growth-layer config section (safe defaults).

        growth層の設定セクションを返す(未設定でも空辞書を返す).

        Args:
            section (str): Section name such as "emotion"/"reading" / セクション名

        Returns:
            Any: Config mapping for the section (supports .get) / 設定(.get可能)
        """
        growth: Any = self.config.get("growth", {})
        return growth.get(section, {})

    def _update_emotion(self) -> None:
        """Build and advance the affect dynamics from the current packet.

        現在のパケットから感情の力学を構築・更新する.

        感情の漏洩(emotion)とゲーティング(gating)のどちらかが有効なら状態を更新する. 感情は
        プロフィールで個体化され, ゲーム中の出来事(言及・投票・死亡)で更新される. ゲーティングは
        この状態に依存するため, 漏洩を切ってもゲーティングが有効なら状態は保つ. 接続層には影響しない.
        """
        emotion_on = bool(self._growth_config("emotion").get("enabled", True))
        gating_on = bool(self._growth_config("gating").get("enabled", True))
        if not (emotion_on or gating_on):
            return
        if self.info is None:
            return
        if self.emotion is None:
            scale = float(self._growth_config("emotion").get("sensitivity_scale", 1.0))
            cal = load_calibration(self.role.value, len(self.info.status_map), self.track)
            self.emotion = EmotionDynamics.from_profile(
                self.info.profile, sensitivity_scale=scale, calibration_adj=cal,
            )
        name = self.info.agent or self.agent_name
        self.emotion.update(self.info, self.talk_history, name)

    def _update_beliefs(self) -> None:
        """Update the belief tracker with new observations.

        信念蓄積器を新しい観察で更新する.
        """
        if self.info is None:
            return
        if self.belief_tracker is None:
            name = self.info.agent or self.agent_name
            self.belief_tracker = BeliefTracker(name)
        self.belief_tracker.update(self.info, self.talk_history)

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def _is_self_dead(self) -> bool:
        """Return whether this agent has already been killed.

        このエージェントが既に死亡しているかを返す.

        死亡後はサーバから DAILY_INITIALIZE 等が届くが, 発言・思考を生成すべきではない.

        Returns:
            bool: True if the agent is dead / 死亡していれば True
        """
        if self.info is None:
            return False
        name = self.info.agent or self.agent_name
        status = self.info.status_map.get(name)
        return status is not None and status != Status.ALIVE

    def _alive_others(self) -> list[str]:
        """Get alive agents excluding oneself.

        自分以外の生存エージェントの一覧を取得する.

        Returns:
            list[str]: Alive agents other than oneself / 自分以外の生存エージェント名のリスト
        """
        name = self.info.agent if self.info else self.agent_name
        return [a for a in self.get_alive_agents() if a != name]

    def _select_target(self) -> str:
        """Select an action target, never choosing oneself.

        対象選択アクション(投票・占い・護衛・襲撃)の対象を返す. 自分自身は選ばない.
        LLMが文章を返した場合は, 生存候補の名前を含むか探し, 含まなければ候補から無作為に選ぶ.

        Returns:
            str: Target agent name / 対象エージェント名
        """
        response = self._send_message_to_llm(self.request)
        name = self.info.agent if self.info else self.agent_name
        candidates = self._alive_others() or self.get_alive_agents()
        if not candidates:
            return response or ""
        if not response:
            return random.choice(candidates)  # noqa: S311
        response = response.strip()
        # LLMが名前のみを返した場合はそのまま使う
        if response == name:
            self.agent_logger.logger.warning("自己を対象に選んだため再選択します: %s", self.request)
            return random.choice(candidates)  # noqa: S311
        if response in candidates:
            override = self._gated_override(response, candidates)
            return override or response
        # LLMが文章を返した場合: 生存候補の名前を含むか探す
        for candidate in candidates:
            if candidate in response:
                self.agent_logger.logger.info(
                    "対象選択の応答から名前を抽出しました: %s -> %s (%s)",
                    response[:60], candidate, self.request,
                )
                override = self._gated_override(candidate, candidates)
                return override or candidate
        # 名前が見つからなければ候補から無作為に選ぶ
        self.agent_logger.logger.warning(
            "対象選択の応答に生存者の名前が含まれず無作為に選びます: %s (%s)",
            response[:60], self.request,
        )
        return random.choice(candidates)  # noqa: S311

    def _gated_override(self, response: str, candidates: list[str]) -> str | None:
        """Override the deliberated target reactively under strong agitation.

        強い動揺の下で, 熟慮した対象を反射的な相手へ上書きする(限定合理性の構造的な摂動).

        感情ゲーティングが無効, 感情が衝動モードでない, または上書きが起きなかった場合は
        None を返す(その場合は熟慮した対象がそのまま使われる).

        Args:
            response (str): The deliberated target / 熟慮して選んだ対象
            candidates (list[str]): Allowed target names / 選べる対象名の一覧

        Returns:
            str | None: The override target, or None / 上書き先. 無ければ None
        """
        if self.info is None or self.emotion is None:
            return None
        if not bool(self._growth_config("gating").get("enabled", True)):
            return None
        mode, level = self.emotion.decision_gate()
        if mode != "impulsive" or level != "strong":
            return None
        prob = self.emotion.perturb_probability()
        if prob <= 0.0 or random.random() >= prob:  # noqa: S311
            return None
        name = self.info.agent or self.agent_name
        salient = salient_pressurers(self.info, self.talk_history, name, candidates)
        alt = impulsive_override(salient, response)
        if alt:
            self.agent_logger.logger.info(
                "感情により対象を反射的に上書きしました: %s -> %s (%s)",
                response,
                alt,
                self.request,
            )
        return alt

    def on_talk_received(self, talk: Talk) -> None:
        """Called when a new talk is received (freeform mode).

        新しいトークを受信した時に呼ばれる (グループチャット方式用).

        Args:
            talk (Talk): Received talk / 受信したトーク
        """

    def on_whisper_received(self, whisper: Talk) -> None:
        """Called when a new whisper is received (freeform mode).

        新しい囁きを受信した時に呼ばれる (グループチャット方式用).

        Args:
            whisper (Talk): Received whisper / 受信した囁き
        """

    async def handle_talk_phase(self, send: Callable[[str], None]) -> None:
        """Handle talk phase in freeform mode.

        グループチャット方式でのトークフェーズ処理.

        Args:
            send (Callable[[str], None]): Send function / 送信関数
        """
        while self.in_talk_phase:
            if self.info and self.info.remain_count is not None and self.info.remain_count <= 0:
                break

            text = self.talk()
            if not self.in_talk_phase:
                break
            send(text)
            await asyncio.sleep(5)

    async def handle_whisper_phase(self, send: Callable[[str], None]) -> None:
        """Handle whisper phase in freeform mode.

        グループチャット方式での囁きフェーズ処理.

        Args:
            send (Callable[[str], None]): Send function / 送信関数
        """
        while self.in_whisper_phase:
            if self.info and self.info.remain_count is not None and self.info.remain_count <= 0:
                break

            text = self.whisper()
            if not self.in_whisper_phase:
                break
            send(text)
            await asyncio.sleep(5)

    _ACTION_REQUESTS = (
        Request.TALK,
        Request.WHISPER,
        Request.VOTE,
        Request.DIVINE,
        Request.GUARD,
        Request.ATTACK,
        Request.DAILY_FINISH,
    )
    # 対象(エージェント名)を選ぶリクエスト
    _TARGET_REQUESTS = (
        Request.VOTE,
        Request.DIVINE,
        Request.GUARD,
        Request.ATTACK,
    )

    def _apply_growth(self, request: Request | None, prompt: str) -> str:
        """Inject growth-layer context into the rendered prompt.

        描画済みプロンプトにgrowth層の文脈(規範・戦略skill・接地情報)を注入する.

        INITIALIZE時は共通規範と役職別戦略skillを末尾に付加し, 行動リクエスト時は
        確定事実・自己スタンス・注意を前置する. 接続層には影響しない.

        Args:
            request (Request | None): The request type / リクエストタイプ
            prompt (str): The rendered prompt / 描画済みプロンプト

        Returns:
            str: The augmented prompt / 拡張後のプロンプト
        """
        if request == Request.INITIALIZE:
            parts = [load_common_norms()]
            if self.info is not None:
                player_count = len(self.info.status_map)
                parts.append(load_role_skill(self.role.value, player_count, self.track))
                # ペルソナの話し方(few-shot)をINITIALIZE時に注入し、
                # 会話履歴の先頭でスタイルを確定させる(TALK時の注入だけでは最初の発話に反映されない)
                voice_note = build_voice_note(self.info.profile)
                if voice_note:
                    parts.append(voice_note)
                if bool(self._growth_config("review").get("enabled", True)):
                    parts.append(load_lessons(self.role.value, player_count, track=self.track))
            suffix = "\n\n".join(part for part in parts if part)
            return f"{prompt}\n\n{suffix}" if suffix else prompt
        if request in self._ACTION_REQUESTS:
            prefix = self._build_action_prefix(request)
            if prefix:
                return f"{prefix}\n\n---\n\n{prompt}"
        return prompt

    def _build_action_prefix(self, request: Request | None) -> str:
        """Build the growth-layer prefix prepended to an action prompt.

        行動リクエストのプロンプトに前置するgrowth層の文脈を構築する.

        確定事実の接地, 自己除外の注意, 他者のテル読み, 感情の漏洩・意思決定ゲーティングを
        この順で組み立てる.

        Args:
            request (Request | None): The request type / リクエストタイプ

        Returns:
            str: The prefix block (may be empty) / 前置ブロック(空のこともある)
        """
        name = self.info.agent if self.info else self.agent_name
        prefix = build_action_grounding(
            self.info,
            name,
            self.role,
            self.talk_history,
            self.known_results,
        )
        # 盤面の算術(勝利条件・パリティ・吊り残機・情報閉包)を演繹して注入する(ADR-008 層0/1)
        if bool(self._growth_config("game_model").get("enabled", True)):
            rule_block = build_rule_model_block(
                self.info,
                self.setting,
                self.role,
                name,
                self._known_species,
            )
            if rule_block:
                prefix = f"{prefix}\n\n{rule_block}" if prefix else rule_block
        if request in self._TARGET_REQUESTS:
            note = build_target_self_exclusion(name)
            prefix = f"{prefix}\n\n{note}" if prefix else note
        # 占い(DIVINE)は確定済みを再占いせず灰を優先させる(評価D: 行動を状況に基づかせる)
        if request == Request.DIVINE:
            seer_claimants = build_seer_priority_claimants(self.talk_history)
            guide = build_divine_guide(
                name, self.get_alive_agents(), self._known_species,
                talk_history=self.talk_history,
                seer_claimants=seer_claimants,
            )
            if guide:
                prefix = f"{prefix}\n\n{guide}" if prefix else guide
        # 護衛(GUARD)は占い師(先制CO)・確定白・霊媒師を優先護衛するよう誘導する
        if request == Request.GUARD:
            seer_claimants = build_seer_priority_claimants(self.talk_history)
            guard_guide = build_guard_guide(
                name, self.get_alive_agents(), self.talk_history,
                seer_claimants=seer_claimants,
                known_species=self._known_species,
            )
            if guard_guide:
                prefix = f"{prefix}\n\n{guard_guide}" if prefix else guard_guide
        # 襲撃(ATTACK)は味方(狂人の可能性)を誤襲撃しないよう誘導する
        if request == Request.ATTACK:
            attack_guide = build_werewolf_attack_guide(self.talk_history, name)
            if attack_guide:
                prefix = f"{prefix}\n\n{attack_guide}" if prefix else attack_guide
            # 襲撃時にSELF/STEER教訓を注入する(whisper合意の遵守・味方誤襲撃の禁止等)
            if bool(self._growth_config("review").get("enabled", True)):
                player_count = len(self.info.status_map) if self.info else None
                attack_lessons = load_lessons(
                    self.role.value, player_count, tags=("SELF", "STEER"),
                    situations=("ATTACK", "WHISPER", "GENERAL"), track=self.track)
                if attack_lessons:
                    prefix = f"{prefix}\n\n{attack_lessons}" if prefix else attack_lessons
        # 複数の占い師COがある場合、先制COを真とみなすヒューリスティックを注入する
        # 発話(TALK)と投票(VOTE)で機能する(占い師本人には対抗への対処を促す)
        if request in (Request.TALK, Request.VOTE):
            seer_note = build_seer_priority(self.talk_history, name, self.role.value)
            if seer_note:
                prefix = f"{prefix}\n\n{seer_note}" if prefix else seer_note
        # 霊媒師(MEDIUM)のTALK時に、結果の開示と占い師主張との照合を促す
        if request == Request.TALK and self.role.value == "MEDIUM":
            medium_guide = build_medium_guide(
                name, self.talk_history, self.known_results,
                self._known_species, self.info,
            )
            if medium_guide:
                prefix = f"{prefix}\n\n{medium_guide}" if prefix else medium_guide
        # 投票(VOTE)時にSELF/STEER教訓を注入する(Seerの白結果を尊重・Medium結果で対抗判定等)
        if request == Request.VOTE:
            if bool(self._growth_config("review").get("enabled", True)):
                player_count = len(self.info.status_map) if self.info else None
                # 終盤(生存者<=4)はENDGAME教訓を優先
                alive_count = len(self.get_alive_agents())
                vote_sits = ("ENDGAME", "VOTE", "COUNTER", "GENERAL") if alive_count <= 4 else ("VOTE", "COUNTER", "GENERAL")
                vote_lessons = load_lessons(
                    self.role.value, player_count, tags=("SELF", "STEER"),
                    situations=vote_sits, track=self.track)
                if vote_lessons:
                    prefix = f"{prefix}\n\n{vote_lessons}" if prefix else vote_lessons
        # 発話(TALK/WHISPER)では口に出す言葉だけを書かせる(対象選択は名前のみ返すので付けない)
        if request in (Request.TALK, Request.WHISPER):
            speech_note = get_speech_format_note()
            prefix = f"{prefix}\n\n{speech_note}" if prefix else speech_note
            # 年齢・プロフィールに応じた声(話し方)のガイドを添える
            voice_note = build_voice_note(self.info.profile if self.info else None)
            if voice_note:
                prefix = f"{prefix}\n\n{voice_note}"
            # 自分の既出フレーズの反復を機械的に検出し、再利用しないよう注意する
            if bool(self._growth_config("repetition_guard").get("enabled", True)):
                repetition_note = build_repetition_guard(name, self.talk_history + self.whisper_history)
                if repetition_note:
                    prefix = f"{prefix}\n\n{repetition_note}"
            # 発話時にSTEER(誘導)・PERSONA(ペルソナ)教訓を注入する
            if bool(self._growth_config("review").get("enabled", True)):
                player_count = len(self.info.status_map) if self.info else None
                # TALKはCO/COUNTER/DEFENSE/VOTEを優先、WHISPERはWHISPERを優先
                if request == Request.WHISPER:
                    talk_sits = ("WHISPER", "ATTACK", "GENERAL")
                else:
                    alive_count = len(self.get_alive_agents())
                    talk_sits = ("ENDGAME", "DEFENSE", "CO", "COUNTER", "VOTE", "GENERAL") if alive_count <= 4 else ("CO", "COUNTER", "DEFENSE", "VOTE", "GENERAL")
                steer_lessons = load_lessons(
                    self.role.value, player_count, tags=("STEER", "PERSONA"),
                    situations=talk_sits, track=self.track)
                if steer_lessons:
                    prefix = f"{prefix}\n\n{steer_lessons}" if prefix else steer_lessons
        reads_others = request == Request.TALK or request in self._TARGET_REQUESTS
        # 蓄積された他者の行動パターンを注入する
        if reads_others and self.belief_tracker is not None:
            alive_set = set(self.get_alive_agents())
            belief_block = self.belief_tracker.build_belief_summary(alive_set)
            if belief_block:
                prefix = f"{prefix}\n\n{belief_block}" if prefix else belief_block
        # 他者の感情のテルを読み役職推定の手がかりにする
        if reads_others and bool(self._growth_config("reading").get("enabled", True)):
            read_block = build_tell_reading(self.info, name, self.talk_history)
            if read_block:
                prefix = f"{prefix}\n\n{read_block}" if prefix else read_block
        return self._append_emotion_blocks(prefix, reads_others=reads_others)

    def _append_emotion_blocks(self, prefix: str, *, reads_others: bool) -> str:
        """Append affect leakage and decision-gating blocks to the prefix.

        感情の漏洩と意思決定ゲーティングのブロックを前置文に付加する.

        漏洩(emotion)は語調へのにじみ, ゲーティング(gating)は意思決定の過程への作用で,
        それぞれ独立に設定で切れる. 感情状態が未構築, または平常・熟慮の場合は足さない.

        Args:
            prefix (str): The prefix built so far / これまでに組み立てた前置文
            reads_others (bool): Whether this is a speech or target request /
                発言または対象選択のリクエストか

        Returns:
            str: The prefix with affect blocks appended / 感情ブロックを付加した前置文
        """
        if self.emotion is None:
            return prefix
        # 感情の漏洩(語調へのにじみ)を前置する
        if bool(self._growth_config("emotion").get("enabled", True)):
            suppress = bool(self._growth_config("emotion").get("expressive_suppression", False))
            emotion_block = self.emotion.injection_text(suppress=suppress)
            prefix = f"{prefix}\n\n{emotion_block}" if prefix else emotion_block
        # 感情が意思決定の過程に及ぼす影響(限定合理性のゲーティング)を前置する
        if reads_others and bool(self._growth_config("gating").get("enabled", True)):
            mode, level = self.emotion.decision_gate()
            gating_block = build_decision_gating(mode, level)
            if gating_block:
                prefix = f"{prefix}\n\n{gating_block}" if prefix else gating_block
        # 役職に応じた戦略的な感情制御を前置する
        if bool(self._growth_config("emotion").get("enabled", True)):
            control_block = build_strategic_control(self.role.value)
            if control_block:
                prefix = f"{prefix}\n\n{control_block}" if prefix else control_block
        return prefix

    def _trim_history(self) -> None:
        """Trim the LLM message history to bound the prompt context length.

        LLMメッセージ履歴を剪定し, プロンプトのコンテキスト長を抑える.

        先頭のINITIALIZE文脈(共通規範・役職skill)は常に残し, それ以外は直近のみ保持する.
        確定事実はgrounding層が毎ターン再供給するため, 古い履歴を落としても整合は保たれる.
        gemmaは user/assistant の厳密な交互を要求するため, tail は HumanMessage 境界
        (偶数index)から開始させて交互構造を壊さない.
        """
        history = self.llm_message_history
        if len(history) <= _MAX_HISTORY_MESSAGES:
            return
        start = len(history) - (_MAX_HISTORY_MESSAGES - _KEEP_HEAD_MESSAGES)
        if start % 2 != 0:
            start += 1
        self.llm_message_history = history[:_KEEP_HEAD_MESSAGES] + history[start:]

    @staticmethod
    def _sanitize_speech_response(response: str, budget: int = _MAX_SPEECH_COUNTED_CHARS) -> str:
        """Remove surface forms that should never be sent as natural speech.

        自然な発話として送ってはいけない表層表現を機械的に除去する.
        """
        text = response.strip()
        if not text:
            return ""
        if _SPEECH_OVER_ONLY_RE.fullmatch(text):
            return "Over"

        for _ in range(4):
            sanitized = _SPEECH_PARENTHETICAL_RE.sub("", text)
            sanitized = _SPEECH_EMOTE_RE.sub("", sanitized)
            if sanitized == text:
                break
            text = sanitized

        text = text.translate(_SPEECH_REMOVE_CHARS)
        text = _SPEECH_ELLIPSIS_RE.sub("", text)
        # 3個以上のピリオドは上のellipsis正規表現で既に除去済み。ちょうど2個だけ残る
        # 二重ピリオド(モデルの生成揺れ)を単一のピリオドに畳む。
        text = _SPEECH_DOUBLE_PERIOD_RE.sub(".", text)
        text = _SPEECH_OVER_TOKEN_RE.sub("", text)
        text = text.replace("\r", " ").replace("\n", " ")
        text = text.replace(",", " ")  # 半角カンマは規約で禁止。決定的に空白へ置換して保証する
        text = _SPEECH_SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        text = _SPEECH_SPACES_RE.sub(" ", text)
        text = _trim_to_char_budget(text.strip(), budget)
        return text or "Over"

    def _refine_speech(self, draft: str) -> str:
        """Re-read a draft utterance once and rewrite it if needed (two-pass generation).

        発話の下書きを一度だけ読み直し, 必要なら書き直す(2段生成の自己推敲).

        既出論点の復唱・説明的な言い回し・唐突さ・字数超過を, 声に出す前の自己点検で
        直す. 推敲呼び出しは単発で行い, 会話履歴には積まない. 失敗時は下書きを返す.

        Args:
            draft (str): Draft utterance from the first pass / 1段目の発話の下書き

        Returns:
            str: Refined (or original) utterance / 推敲後(または元)の発話
        """
        if not draft or draft.strip() == "Over" or self.llm_model is None:
            return draft
        t = _growth_texts.get()
        recent = [talk for talk in self.talk_history[-8:] if not talk.skip]
        context = "\n".join(f"{talk.agent}: {talk.text}" for talk in recent)
        profile = (self.info.profile if self.info else None) or ""
        # 文脈やプロフィールが空だと推敲プロンプト自体へのメタ応答を誘発するため見送る
        if not context or not profile:
            return draft
        prompt = t.REFINE_SPEECH_TMPL.format(profile=profile, context=context, draft=draft)
        try:
            refined = (self.llm_model | StrOutputParser()).invoke([HumanMessage(content=prompt)])
        except Exception:
            self.agent_logger.logger.exception("Failed to refine speech")
            return draft
        return refined.strip() or draft

    def _rewrite_repeated_phrase(self, draft: str, phrase: str) -> str:
        """Force a rewrite when the draft reuses one of the agent's own overused phrases.

        下書きが自分の既出フレーズを再利用している場合, 書き直しを強制する.
        プロンプトでの注意喚起だけでは遵守が保証されないため, 実際に検出された場合に
        機械的にこの書き直しパスを発動する(ADR-011).

        Args:
            draft (str): Draft utterance that reused the phrase / フレーズを再利用した下書き
            phrase (str): The overused phrase detected / 検出された既出フレーズ

        Returns:
            str: Rewritten utterance, or the draft unchanged on failure /
                書き直した発話. 失敗時は下書きをそのまま返す
        """
        if self.llm_model is None:
            return draft
        t = _growth_texts.get()
        prompt = t.REPETITION_REWRITE_TMPL.format(count=REPEAT_MIN_COUNT, phrase=phrase, draft=draft)
        try:
            rewritten = (self.llm_model | StrOutputParser()).invoke([HumanMessage(content=prompt)])
        except Exception:
            self.agent_logger.logger.exception("Failed to rewrite repeated phrase")
            return draft
        return rewritten.strip() or draft

    def _speech_char_budget(self) -> int:
        """Return the server's per-utterance character budget (spaces excluded).

        1発話あたりの文字数上限(空白非算入)を返す. 125 を決め打ちしないための単一の
        解決点(リファレンス§4.1). サーバが渡す残り消費可能文字数(remain_length,
        ターン式のTALK時)と設定の base_length の, いずれも正のものの最小を取り, どちらも
        無ければ既定 125 を用いる. freeform では remain_length が無く base_length が効く.

        Returns:
            int: Per-utterance char budget / 1発話の文字数上限
        """
        candidates: list[int] = []
        info = self.info
        remain = getattr(info, "remain_length", None) if info is not None else None
        if isinstance(remain, int) and remain > 0:
            candidates.append(remain)
        try:
            base = self.setting.talk.max_length.base_length  # type: ignore[union-attr]
        except AttributeError:
            base = None
        if isinstance(base, int) and base > 0:
            candidates.append(base)
        return min(candidates) if candidates else _MAX_SPEECH_COUNTED_CHARS

    def _postprocess_speech(self, response: str) -> str:
        """Refine (if enabled) and sanitize a speech response.

        発話応答に自己推敲(有効時)と整形を適用する.

        Args:
            response (str): Raw speech response / 生の発話応答

        Returns:
            str: Post-processed speech / 後処理済みの発話
        """
        if bool(self._growth_config("refine").get("enabled", False)):
            refined = self._refine_speech(response)
            if refined != response:
                self.agent_logger.logger.info(["SPEECH_REFINED", response, refined])
                response = refined
        if bool(self._growth_config("repetition_guard").get("enabled", True)):
            own_name = self.info.agent if self.info else self.agent_name
            phrase = find_repeated_phrase(own_name, self.talk_history + self.whisper_history, response)
            if phrase:
                rewritten = self._rewrite_repeated_phrase(response, phrase)
                if rewritten != response:
                    self.agent_logger.logger.info(["SPEECH_DEREPEATED", response, rewritten, phrase])
                    response = rewritten
        sanitized = self._sanitize_speech_response(response, self._speech_char_budget())
        if sanitized != response:
            self.agent_logger.logger.info(["SPEECH_SANITIZED", response, sanitized])
        return sanitized

    @staticmethod
    def _strip_reasoning(text: str | None) -> str | None:
        """Remove <think>...</think> reasoning blocks from an LLM response.

        Hermes等のreasoningモデルが思考を<think></think>で出力した場合に除去し,
        最終回答のみを残す. 閉じ忘れ(生成途中で切れた思考)にも対応する.

        LLM応答から<think></think>の思考ブロックを除去する.
        """
        if not text:
            return text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        return text.strip()

    def _send_message_to_llm(self, request: Request | None) -> str | None:
        """Send message to LLM and get response.

        LLMにメッセージを送信して応答を取得する.

        Args:
            request (Request | None): The request type to process / 処理するリクエストタイプ

        Returns:
            str | None: LLM response or None if error occurred / LLMの応答またはエラー時はNone
        """
        if request is None:
            return None
        # 死亡したエージェントは発言・思考を生成しない(死亡後のDAILY_INITIALIZE等で
        # 「死亡しており発言できません」のような応答を吐くのを防ぐ)。INITIALIZE時は生存。
        if request != Request.INITIALIZE and self._is_self_dead():
            return None
        prompts = self._prompts or self.config["prompt"]
        if request.lower() not in prompts:
            return None
        prompt = prompts[request.lower()]
        if float(self.config["llm"]["sleep_time"]) > 0:
            sleep(float(self.config["llm"]["sleep_time"]))
        key = {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
        }
        template: Template = Template(prompt)
        prompt = template.render(**key).strip()
        prompt = self._apply_growth(request, prompt)
        if self.llm_model is None:
            self.agent_logger.logger.error("LLM is not initialized")
            return None
        try:
            self.llm_message_history.append(HumanMessage(content=prompt))
            self._trim_history()
            response = (self.llm_model | StrOutputParser()).invoke(self.llm_message_history)
            response = self._strip_reasoning(response)
            if request in (Request.TALK, Request.WHISPER):
                response = self._postprocess_speech(response)
            self.llm_message_history.append(AIMessage(content=response))
            self.agent_logger.logger.info(["LLM", prompt, response])
        except Exception:
            self.agent_logger.logger.exception("Failed to send message to LLM")
            return None
        else:
            return response

    @timeout
    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
        """
        return self.agent_name

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        """
        if self.info is None:
            return

        # 設定から言語コードを読み込み, growth層のテキスト定数を切り替える
        lang = str(self.config.get("agent", {}).get("language", "ja"))
        _growth_texts.set_language(lang)
        self._prompts = _growth_texts.get().PROMPTS

        model_type = str(self.config["llm"]["type"])
        match model_type:
            case "openai":
                # base_urlが設定されていればvLLM等のOpenAI互換エンドポイントを使用する
                base_url = self.config["openai"].get("base_url")
                # max_tokens: reasoningモデルが思考でtokenを消費するため、発話の短さに
                # 関わらず十分な予算を確保する。未設定なら既定(1024)を使う。
                max_tokens = int(self.config["openai"].get("max_tokens", 1024))
                # enable_thinking: reasoningモデル(Qwythos等)の思考ブロックを切る。
                # 思考が出力tokenを消費してlatencyが膨らむのを防ぐ。未設定ならTrue(思考あり)。
                enable_thinking = bool(self.config["openai"].get("enable_thinking", True))
                self.llm_model = ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=float(self.config["openai"]["temperature"]),
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                    base_url=str(base_url) if base_url else None,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
            case "google":
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["google"]["model"]),
                    temperature=float(self.config["google"]["temperature"]),
                    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            case "ollama":
                self.llm_model = ChatOllama(
                    model=str(self.config["ollama"]["model"]),
                    temperature=float(self.config["ollama"]["temperature"]),
                    base_url=str(self.config["ollama"]["base_url"]),
                )
            case _:
                raise ValueError(model_type, "Unknown LLM type")
        self.llm_model = self.llm_model
        self._send_message_to_llm(self.request)

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.

        昼開始の応答は送信されない(このメソッドは戻り値を持たない)。以前はここでLLMを
        呼んでいたが, grounding未適用かつ占い結果が未確定な初日に存在しない占い結果を
        捏造し, LLM会話履歴を汚染して後続の発話に矛盾を伝播させていた。状況は後続のTALKの
        grounding(確定事実・自己スタンス・未占い時の捏造禁止)で正しく注入されるため呼ばない。
        """

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        response = self._send_message_to_llm(self.request)
        self.sent_whisper_count = len(self.whisper_history)
        return response or ""

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        response = self._send_message_to_llm(Request.TALK)
        self.sent_talk_count = len(self.talk_history)
        return response or ""

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """
        self._send_message_to_llm(self.request)

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        return self._select_target()

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        return self._select_target()

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return self._select_target()

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        return self._select_target()

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.

        終局後の自己レビュー(真値接地の自己改善)をバックグラウンドスレッドで起動する.
        スレッドは非デーモンのため, プロセスは終了時にレビュー完了を待つ(接続層は無変更).
        """
        if self.info is None or self.llm_model is None:
            return
        if not bool(self._growth_config("review").get("enabled", True)):
            return
        role_map = self.info.role_map
        if not role_map:
            return
        name = self.info.agent or self.agent_name
        thread = threading.Thread(
            target=run_post_game_review,
            args=(
                self.llm_model,
                self.role.value,
                name,
                self.info,
                self.talk_history,
                self.track,
            ),
        )
        thread.start()

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None

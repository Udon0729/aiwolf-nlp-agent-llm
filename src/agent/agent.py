"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import asyncio
import os
import random
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

from growth.emotion import EmotionDynamics
from growth.gating import build_decision_gating, impulsive_override, salient_pressurers
from growth.grounding import SPEECH_FORMAT_NOTE, build_action_grounding, build_target_self_exclusion
from growth.reading import build_tell_reading
from growth.review import load_lessons, run_post_game_review
from growth.skills_loader import load_common_norms, load_role_skill
from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")

# 先頭の初期化文脈を保持しつつメッセージ履歴を剪定する上限
_MAX_HISTORY_MESSAGES = 12
_KEEP_HEAD_MESSAGES = 2


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
        # growth層: 性格(プロフィール)で個体化された感情の力学(ゲーム開始時に構築)
        self.emotion: EmotionDynamics | None = None

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
            self.emotion = None
        self._update_emotion()
        self.agent_logger.logger.debug(packet)

    def _record_known_results(self, info: Info) -> None:
        """Accumulate private results (divine/medium) obtained by the agent.

        エージェントが得た占い・霊媒結果を蓄積する.

        Args:
            info (Info): Game info that may contain results / 結果を含みうるゲーム情報
        """
        for label, value in (("占い", info.divine_result), ("霊媒", info.medium_result)):
            if value is not None:
                entry = f"{label}: {value}"
                if entry not in self.known_results:
                    self.known_results.append(entry)

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
            self.emotion = EmotionDynamics.from_profile(self.info.profile)
        name = self.info.agent or self.agent_name
        self.emotion.update(self.info, self.talk_history, name)

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

        Returns:
            str: Target agent name / 対象エージェント名
        """
        response = self._send_message_to_llm(self.request)
        name = self.info.agent if self.info else self.agent_name
        candidates = self._alive_others() or self.get_alive_agents()
        if not candidates:
            return response or ""
        if not response or response.strip() == name:
            if response:
                self.agent_logger.logger.warning("自己を対象に選んだため再選択します: %s", self.request)
            return random.choice(candidates)  # noqa: S311
        override = self._gated_override(response, candidates)
        return override or response

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
                parts.append(load_role_skill(self.role.value, player_count))
                if bool(self._growth_config("review").get("enabled", True)):
                    parts.append(load_lessons(self.role.value, player_count))
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
        if request in self._TARGET_REQUESTS:
            note = build_target_self_exclusion(name)
            prefix = f"{prefix}\n\n{note}" if prefix else note
        # 発話(TALK/WHISPER)では口に出す言葉だけを書かせる(対象選択は名前のみ返すので付けない)
        if request in (Request.TALK, Request.WHISPER):
            prefix = f"{prefix}\n\n{SPEECH_FORMAT_NOTE}" if prefix else SPEECH_FORMAT_NOTE
        # 他者の感情のテルを読み役職推定の手がかりにする
        reads_others = request == Request.TALK or request in self._TARGET_REQUESTS
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
        if request.lower() not in self.config["prompt"]:
            return None
        prompt = self.config["prompt"][request.lower()]
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

        model_type = str(self.config["llm"]["type"])
        match model_type:
            case "openai":
                # base_urlが設定されていればvLLM等のOpenAI互換エンドポイントを使用する
                base_url = self.config["openai"].get("base_url")
                self.llm_model = ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=float(self.config["openai"]["temperature"]),
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                    base_url=str(base_url) if base_url else None,
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
        """
        self._send_message_to_llm(self.request)

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

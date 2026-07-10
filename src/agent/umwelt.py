"""Uexkull-inspired agent variant.

環世界論に基づくエージェント variant を定義するモジュール.

接続層の契約は通常の Agent と同じまま、感情VADの代わりにプロフィール由来の
「何が目立ち、何を脅威と見なし、何を見落としやすいか」をプロンプトへ注入する.
ゲーム状態(自分への言及・票・死亡・場の沈黙)を印としてレンズに通すことで,
機能環(知覚→行動→環境→知覚)を実装する.
"""

from __future__ import annotations

from agent.agent import Agent
from growth.umwelt import build_umwelt_context


class UmweltAgent(Agent):
    """Agent variant that acts through a persona-specific subjective world.

    プロフィールごとの主観的な環世界を通して判断する Agent variant.
    """

    def _update_emotion(self) -> None:
        """Disable VAD emotion dynamics for the Umwelt variant.

        環世界 variant では感情VAD力学を使わない.
        """

    def _append_emotion_blocks(self, prefix: str, *, reads_others: bool) -> str:
        """Append the Umwelt lens instead of affect/gating blocks.

        感情・ゲーティングのブロックではなく, 主観的環世界のブロックを付加する.

        いまの局面の印を検出するためゲーム状態も渡す. 他者を読む場面(発話・対象選択)
        でのみ, レンズで論点を選ぶ持ちかけの行が付く.

        Args:
            prefix (str): The prefix built so far / これまでに組み立てた前置文
            reads_others (bool): Whether the request reads other players / 他者を読むリクエストか

        Returns:
            str: Prefix with Umwelt block / 環世界ブロック付き前置文
        """
        if not bool(self._growth_config("umwelt").get("enabled", True)):
            return prefix
        name = (self.info.agent if self.info is not None else None) or self.agent_name
        block = build_umwelt_context(
            self.info,
            self.role.value,
            self.talk_history,
            name,
            reads_others=reads_others,
        )
        return f"{prefix}\n\n{block}" if prefix else block

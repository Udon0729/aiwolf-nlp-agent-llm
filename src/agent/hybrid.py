"""Hybrid agent variant: Umwelt lens plus affect expression.

環世界レンズと感情表出を併用するハイブリッド variant を定義するモジュール.

実対局比較(eval/results/variant_match_20260703.json)で, 環世界レンズは行動と対話の
整合(D項目)・投票的中に強く, 感情VADはキャラクター性(E項目)・一貫性(C項目)に
強いという相補性が観察された. 本 variant は環世界レンズで知覚と行動選択を性格づけ,
感情VADで語調の表出を重ねることで, 勝率部門と評価部門の両立を狙う.

感情の意思決定ゲーティング(gating)はD項目を下げる要因のため, 既定では config 側で
無効にして使う(eval/run_eval.py の hybrid 条件を参照).
"""

from __future__ import annotations

from agent.agent import Agent
from growth.umwelt import build_umwelt_context


class HybridAgent(Agent):
    """Agent variant that combines the Umwelt lens with affect expression.

    環世界レンズと感情表出を組み合わせる Agent variant.
    """

    def _append_emotion_blocks(self, prefix: str, *, reads_others: bool) -> str:
        """Prepend the Umwelt lens, then the standard affect blocks.

        環世界ブロックを先に付け, その後に通常の感情系ブロックを付加する.

        環世界レンズが知覚と行動選択の枠を与え, 感情の漏洩が語調を性格づける.
        感情・ゲーティング・戦略的制御の各ブロックは親クラスの実装と config に従う.

        Args:
            prefix (str): The prefix built so far / これまでに組み立てた前置文
            reads_others (bool): Whether the request reads other players / 他者を読むリクエストか

        Returns:
            str: Prefix with Umwelt and affect blocks / 環世界と感情のブロック付き前置文
        """
        if bool(self._growth_config("umwelt").get("enabled", True)):
            name = (self.info.agent if self.info is not None else None) or self.agent_name
            block = build_umwelt_context(
                self.info,
                self.role.value,
                self.talk_history,
                name,
                reads_others=reads_others,
            )
            prefix = f"{prefix}\n\n{block}" if prefix else block
        return super()._append_emotion_blocks(prefix, reads_others=reads_others)

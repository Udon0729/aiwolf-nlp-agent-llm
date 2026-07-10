"""Module that models an agent's affective state as a small dynamical system.

エージェントの感情状態を低次元の力学系として表すモジュール.

感情を VAD(valence 感情価, arousal 覚醒, dominance 支配)の連続値で保持し,
ゲーム中の出来事(appraisal)で更新, 性格(プロフィール)由来の係数で個体化する.
現在の感情はプロンプトに前置されて発言・選択ににじみ出る(漏洩, leakage). 表出抑制
(expressive suppression)を有効にすると, 感情を表に出すまいとするが完全には隠れない.

焦りは一事例にすぎず, 怒り・自信・安心・落ち込みなども同じ VAD 空間上の領域として扱う.
これは最小スライス(第1段)であり, 性格→係数の写像は決定論的ヒューリスティックである.
意思決定への限定合理性ゲーティング(QRE-lambda/level-k)と他者の感情の読みは後段で追加する.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Talk

# describe で用いる感情領域の境界(comparison の magic value を避けるため定数化)
_AROUSAL_HIGH = 0.5
_AROUSAL_CALM = 0.4
_VALENCE_NEG = -0.15
_VALENCE_NEG_STRONG = -0.2
_VALENCE_POS = 0.2
# 自信とみなす際に感情価が負でも許容する下限
_VALENCE_MILD_POS = -0.1
_DOMINANCE_HIGH = 0.2
# これ未満を平常とみなす逸脱量の閾値
_NEUTRAL_DEVIATION = 0.25
# これ以上はペルソナを上書きしてよいほど強い感情とみなす
_STRONG_DEVIATION = 0.5
# 1回の更新で数える自己への言及数の上限(過大な振れを防ぐ)
_MAX_MENTIONS_PER_UPDATE = 3
# 意思決定が拙速・反射的になり始めるとみなす覚醒の下限
_AROUSAL_ELEVATED = 0.35
# 構造的な行動上書きが起こりうる覚醒の下限。衝動モードの覚醒に揃える
_PERTURB_MIN_AROUSAL = 0.35
# 行動上書き確率の上限
_PERTURB_MAX_PROB = 0.5
# 感受性に乗ずる行動上書き確率の利得
_PERTURB_GAIN = 0.8


def _clip(value: float, low: float, high: float) -> float:
    """Clip a value into the closed interval [low, high].

    値を閉区間 [low, high] に収める.

    Args:
        value (float): Value to clip / 収める値
        low (float): Lower bound / 下限
        high (float): Upper bound / 上限

    Returns:
        float: Clipped value / 収めた値
    """
    return max(low, min(high, value))


@dataclass
class Vad:
    """A point in valence-arousal-dominance affect space.

    valence-arousal-dominance 感情空間上の一点.

    valence は [-1, 1](不快から快), arousal は [0, 1](平静から高覚醒),
    dominance は [-1, 1](服従から支配)の範囲を取る.
    """

    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0


@dataclass
class EmotionTraits:
    """Personality-derived parameters governing affect dynamics.

    感情の力学を支配する, 性格(プロフィール)由来のパラメータ.

    sensitivity は出来事への感受性(gain), retention は baseline への引き戻しの遅さ
    (1 に近いほど感情が長く残り蓄積する), baseline は性格的な平常時の感情点,
    suppression は表出抑制の性向(内心では感じても表に出すまいとする).
    """

    sensitivity: float = 0.5
    retention: float = 0.6
    baseline: Vad = field(default_factory=Vad)
    suppression: bool = False


# プロフィール文のキーワードから性格係数を導出するための語彙群(ヒューリスティック v0)
# 不動型は動じにくく蓄積もしない。低感受性で減衰が速くフラットに保たれる
_UNFLAPPABLE_WORDS = ("冷静", "落ち着", "沈着", "泰然", "物静か", "動じ")
# 溜め型はすぐには反応しないが内に溜め込み、抑えきれず遅れて噴出する。低感受性で減衰が遅く抑制的
_BOTTLING_WORDS = (
    "我慢", "辛抱", "忍耐", "こらえ", "寡黙", "口数が少な", "内に秘め",
    "感情を表に出さ", "感情を表に出すことは少な", "感情を抑え", "感情を出さ",
)
_EXCITABLE_WORDS = (
    "短気", "せっかち", "おっちょこちょい", "そそっかし", "心配", "不安",
    "臆病", "慌", "気が小さい", "神経質", "焦",
)
_CONFIDENT_WORDS = ("自信", "大胆", "強気", "堂々", "豪胆", "負けず嫌い", "プライド")
_AGGRESSIVE_WORDS = ("怒", "攻撃的", "好戦", "気が強", "激")
_CHEERFUL_WORDS = ("明る", "陽気", "楽観", "ポジティブ", "朗らか")
_GLOOMY_WORDS = ("暗い", "悲観", "ネガティブ", "内向", "陰")


def derive_traits(profile: str | None) -> EmotionTraits:
    """Derive emotion-dynamics traits from a free-text profile.

    自由記述のプロフィールから感情力学の係数を導出する(決定論的ヒューリスティック).

    プロフィールが無い場合は中立的な既定値を返す. LLM による係数推定は後段の課題.

    Args:
        profile (str | None): The persona profile text / ペルソナのプロフィール文

    Returns:
        EmotionTraits: Derived traits / 導出した係数
    """
    traits = EmotionTraits()
    if not profile:
        return traits
    text = profile
    is_bottling = any(word in text for word in _BOTTLING_WORDS)
    if is_bottling:
        # 溜め型はすぐには反応しないが内に溜め込み, 表に出すまいと抑える
        traits.sensitivity -= 0.2
        traits.retention += 0.3
        traits.suppression = True
    elif any(word in text for word in _UNFLAPPABLE_WORDS):
        # 不動型は刺激に動じにくく動揺してもすぐ平常へ戻るよう減衰を強める.
        # 溜め型語彙がある場合はそちらを優先し, ここは適用しない
        traits.sensitivity -= 0.3
        traits.retention -= 0.3
    if any(word in text for word in _EXCITABLE_WORDS):
        traits.sensitivity += 0.3
        traits.retention += 0.25
    if any(word in text for word in _CONFIDENT_WORDS):
        traits.baseline.dominance += 0.3
        traits.baseline.valence += 0.1
    if any(word in text for word in _AGGRESSIVE_WORDS):
        traits.baseline.dominance += 0.2
        traits.sensitivity += 0.15
    if any(word in text for word in _CHEERFUL_WORDS):
        traits.baseline.valence += 0.25
    if any(word in text for word in _GLOOMY_WORDS):
        traits.baseline.valence -= 0.25
    traits.sensitivity = _clip(traits.sensitivity, 0.1, 1.0)
    traits.retention = _clip(traits.retention, 0.2, 0.95)
    traits.baseline.valence = _clip(traits.baseline.valence, -0.5, 0.5)
    traits.baseline.dominance = _clip(traits.baseline.dominance, -0.5, 0.5)
    return traits


def _describe(state: Vad) -> tuple[str, str]:
    """Describe the current affect as a Japanese phrase and an intensity level.

    現在の感情を日本語の語句と強度レベルで表現する.

    強度レベルは "neutral"(平常)/"mild"(軽微)/"strong"(強い)の3段階で, 注入の強さを
    調整するために用いる. 焦りに限らず, 怒り・自信・安心・落ち込みなども領域として扱う.

    Args:
        state (Vad): Current affect point / 現在の感情点

    Returns:
        tuple[str, str]: (phrase, level) with level in
            "neutral"/"mild"/"strong" / (語句, 強度レベル)
    """
    deviation = max(abs(state.valence), state.arousal, abs(state.dominance))
    if deviation < _NEUTRAL_DEVIATION:
        return "ほぼ平静で、特段の動揺はない", "neutral"
    level = "strong" if deviation >= _STRONG_DEVIATION else "mild"
    degree = "強く" if level == "strong" else "やや"

    if state.arousal >= _AROUSAL_HIGH and state.valence <= _VALENCE_NEG:
        core = "苛立ち・怒りを覚えている" if state.dominance >= _DOMINANCE_HIGH else "焦り・動揺を感じている"
    elif state.dominance >= _DOMINANCE_HIGH and state.valence >= _VALENCE_MILD_POS:
        core = "落ち着いて自信を保っている" if state.arousal < _AROUSAL_CALM else "強気になっている"
    elif state.valence >= _VALENCE_POS and state.arousal < _AROUSAL_CALM:
        core = "安心して落ち着いている"
    elif state.valence <= _VALENCE_NEG_STRONG and state.arousal < _AROUSAL_CALM:
        core = "気分が沈み、落ち込んでいる"
    elif state.valence < 0:
        core = "気がかりを感じている"
    else:
        core = "気持ちが高ぶっている"
    return f"{degree}{core}", level


@dataclass
class EmotionDynamics:
    """Stateful affect dynamics for a single agent over one game.

    1ゲーム分の, 単一エージェントの感情力学(状態を保持する).
    """

    traits: EmotionTraits = field(default_factory=EmotionTraits)
    state: Vad = field(default_factory=Vad)
    _last_talk_idx: int = -1
    _last_vote_day: int = -1
    _last_self_vote_count: int = 0
    _seen_death: bool = False

    @classmethod
    def from_profile(cls, profile: str | None, sensitivity_scale: float = 1.0) -> EmotionDynamics:
        """Build dynamics initialised to the personality baseline.

        性格の baseline に初期化した感情力学を構築する.

        sensitivity_scale は感受性に乗じる全体係数で, トレードオフ曲線を描くために感情の
        反応の強さを一律に振るのに用いる(1.0 が既定). 0 に近いほど感情は動かなくなる.

        Args:
            profile (str | None): The persona profile text / ペルソナのプロフィール文
            sensitivity_scale (float): Global gain on sensitivity / 感受性への全体係数

        Returns:
            EmotionDynamics: Initialised dynamics / 初期化した感情力学
        """
        traits = derive_traits(profile)
        traits.sensitivity = _clip(traits.sensitivity * sensitivity_scale, 0.0, 1.5)
        base = traits.baseline
        return cls(traits=traits, state=Vad(base.valence, base.arousal, base.dominance))

    def _stimulus(self, info: Info, talk_history: list[Talk], name: str) -> tuple[Vad, bool]:
        """Compute the appraisal stimulus from new events since last update.

        前回更新以降の新規イベントから appraisal の刺激ベクトルを計算する.

        自分への言及(疑い・追及の代理), 自分への投票, 新たな死亡(緊張)を契機とする. 併せて
        新規トークを観測したか(時間が進んだか)を返す. 新規イベントの無い空のパケットでは
        時間を進めず感情を保持するため, この真偽値を用いる.

        Args:
            info (Info): Current game info / 現在のゲーム情報
            talk_history (list[Talk]): Talk history / トーク履歴
            name (str): The agent's in-game name / ゲーム内のエージェント名

        Returns:
            tuple[Vad, bool]: (stimulus, saw_new_talk). 刺激ベクトルと新規トーク観測の有無
        """
        stim = Vad()

        mentions = 0
        saw_new_talk = False
        for talk in talk_history:
            if talk.idx <= self._last_talk_idx:
                continue
            saw_new_talk = True
            if talk.agent != name and name and not talk.skip and name in talk.text:
                mentions += 1
        if talk_history:
            self._last_talk_idx = max(t.idx for t in talk_history)
        mentions = min(mentions, _MAX_MENTIONS_PER_UPDATE)
        stim.valence += -0.15 * mentions
        stim.arousal += 0.15 * mentions

        current_self_votes = sum(1 for vote in (info.vote_list or []) if vote.target == name)
        if info.day != self._last_vote_day:
            # 投票結果は日ごとにリセットされるため, 日が変わったら基準も戻す
            self._last_vote_day = info.day
            self._last_self_vote_count = 0
        delta_votes = max(0, current_self_votes - self._last_self_vote_count)
        if delta_votes > 0:
            self._last_self_vote_count = current_self_votes
            stim.valence += -0.3 * delta_votes
            stim.arousal += 0.35 * delta_votes
            stim.dominance += -0.25 * delta_votes

        death = info.executed_agent is not None or info.attacked_agent is not None
        if death and not self._seen_death:
            self._seen_death = True
            stim.valence += -0.1
            stim.arousal += 0.1
        return stim, saw_new_talk

    def _advance(self, current: float, base: float, stim: float, low: float, high: float) -> float:
        """Advance one affect axis: retention pull-back plus scaled stimulus.

        感情の1軸を更新する(baseline への引き戻し + 感受性倍した刺激, 範囲に収める).

        Args:
            current (float): Current axis value / 現在の軸の値
            base (float): Baseline value for the axis / その軸の baseline
            stim (float): Stimulus for the axis / その軸の刺激
            low (float): Lower bound / 下限
            high (float): Upper bound / 上限

        Returns:
            float: Updated axis value / 更新後の軸の値
        """
        ret = self.traits.retention
        sens = self.traits.sensitivity
        return _clip(base + ret * (current - base) + sens * stim, low, high)

    def update(self, info: Info | None, talk_history: list[Talk], name: str) -> None:
        """Advance the affect state by one appraisal step.

        appraisal を1ステップ進めて感情状態を更新する.

        baseline への引き戻し(retention)を適用したうえで, 今ターンの刺激を sensitivity 倍
        して加え, 各軸を範囲内に収める.

        Args:
            info (Info | None): Current game info / 現在のゲーム情報
            talk_history (list[Talk]): Talk history / トーク履歴
            name (str): The agent's in-game name / ゲーム内のエージェント名
        """
        if info is None:
            return
        stim, saw_new_talk = self._stimulus(info, talk_history, name)
        has_event = saw_new_talk or stim.valence != 0.0 or stim.arousal != 0.0 or stim.dominance != 0.0
        if not has_event:
            # 新規イベントの無い空のパケットでは時間を進めず, 直近の感情を保持する.
            # 議論直後の投票に, その日の高ぶりが減衰せず持ち越されるようにする.
            return
        base = self.traits.baseline
        self.state.valence = self._advance(self.state.valence, base.valence, stim.valence, -1.0, 1.0)
        self.state.arousal = self._advance(self.state.arousal, base.arousal, stim.arousal, 0.0, 1.0)
        self.state.dominance = self._advance(self.state.dominance, base.dominance, stim.dominance, -1.0, 1.0)

    def injection_text(self, *, suppress: bool) -> str:
        """Build the prompt block that lets the current affect leak (or be suppressed).

        現在の感情を発言・選択ににじませる(あるいは抑制する)プロンプトブロックを構築する.

        表出抑制が有効な場合は強度で振る舞いを変える. 軽微なうちは表に出さず耐え, 抑えきれない
        ほど強まると, 普段感情を出さない人物の語調の端々に, 溜めていたものが噴き出す.

        Args:
            suppress (bool): Whether expressive suppression is active /
                表出抑制を有効にするか(性格由来の抑制とどちらかが真なら抑制する)

        Returns:
            str: Prompt block text / プロンプトブロックのテキスト
        """
        phrase, level = _describe(self.state)
        if level == "neutral":
            return f"## あなたの今の感情\n- いまあなたは{phrase}。"
        if suppress or self.traits.suppression:
            if level == "mild":
                # 耐えている段階。内心では感じるが表向きは平静を保つ
                return (
                    "## あなたの今の感情と自制\n"
                    f"- いまあなたは内心で{phrase}が、表に出すまいと抑えており、表向きは平静を保っている。\n"
                    "- 語調はあくまで普段通りに保ち、感情はごくわずかに端ににじむ程度に留めること。"
                )
            # 抑えきれず噴出する段階
            return (
                "## あなたの今の感情と自制の限界\n"
                f"- いまあなたは内心で{phrase}。普段は感情を表に出さないあなたが、抑えきれなくなっている。\n"
                "- これまで抑えていたものが、語調の端々や、誰を疑い誰に投票するかの選択に、はっきりと漏れ出す。\n"
                "- 基本的な人物像(プロフィール)は保ったまま、抑えが効かなくなった人物として話すこと。"
                "感情は声に出す言葉と語調だけで表し、しぐさや心情を括弧書きで描写しない。"
            )
        if level == "mild":
            # 軽微な感情はペルソナを上書きしない。語調にわずかに表れる程度に留める
            return (
                "## あなたの今の感情\n"
                f"- いまあなたは{phrase}。\n"
                "- この気配は、あなたの性格(プロフィール)の範囲内で語調にわずかに表れる程度でよい。"
                "感情に飲まれて人物像から外れた言動をしないこと。"
            )
        return (
            "## あなたの今の感情(発言・選択ににじみ出る)\n"
            f"- いまあなたは{phrase}。\n"
            "- この感情は強く、発言の語調や、誰を疑い誰に投票するかの選択に自然とにじみ出る。\n"
            "- 基本的な人物像(プロフィール)は保ったまま、その感情を抱いた人物として話すこと。"
            "感情は声に出す言葉と語調だけで表し、しぐさや心情を括弧書きで描写しない。"
        )

    def decision_gate(self) -> tuple[str, str]:
        """Classify how the current affect bounds the decision process.

        現在の感情が意思決定の過程をどう束縛するかを分類する(限定合理性のゲーティング).

        高覚醒かつ負の感情価のとき, 支配が高ければ強気の決め打ち, 低ければ拙速で反射的な
        振る舞いになるとみなす. それ以外は熟慮とみなしゲーティングを掛けない.

        Returns:
            tuple[str, str]: (mode, level). mode は
                "deliberate"/"impulsive"/"stubborn", level は
                "neutral"/"mild"/"strong" / (モード, 強度)
        """
        state = self.state
        deviation = max(abs(state.valence), state.arousal, abs(state.dominance))
        if deviation < _NEUTRAL_DEVIATION:
            return "deliberate", "neutral"
        level = "strong" if deviation >= _STRONG_DEVIATION else "mild"
        if state.arousal >= _AROUSAL_ELEVATED and state.valence <= _VALENCE_NEG:
            mode = "stubborn" if state.dominance >= _DOMINANCE_HIGH else "impulsive"
        else:
            mode = "deliberate"
        return mode, level

    def perturb_probability(self) -> float:
        """Return the probability that affect overrides the deliberated choice.

        熟慮した選択を感情が反射的に上書きする確率を返す(覚醒と感受性に応じて高まる).

        覚醒が下限に満たなければ 0 を返す. 上限で頭打ちにし, 暴走を防ぐ.

        Returns:
            float: Override probability in the closed range up to the cap /
                上限までの上書き確率
        """
        arousal = self.state.arousal
        if arousal < _PERTURB_MIN_AROUSAL:
            return 0.0
        scaled = (arousal - _PERTURB_MIN_AROUSAL) / (1.0 - _PERTURB_MIN_AROUSAL)
        prob = _PERTURB_GAIN * self.traits.sensitivity * scaled
        return _clip(prob, 0.0, _PERTURB_MAX_PROB)

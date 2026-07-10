"""English text constant module.

All growth-layer hard-coded texts in English.
Same constant names and structure as texts_ja.py;
referenced via the texts.py dispatcher.

Format placeholders use {variable_name} and are expanded with .format().
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# emotion.py — keyword lists for derive_traits()
# ---------------------------------------------------------------------------

# Unflappable type: low sensitivity, fast decay
UNFLAPPABLE_WORDS: tuple[str, ...] = (
    "calm", "composed", "unflappable", "stoic", "cool-headed",
    "collected", "impassive", "serene", "tranquil",
)

# Bottling type: low sensitivity, slow decay, suppression
BOTTLING_WORDS: tuple[str, ...] = (
    "patient", "endure", "restrain", "suppress", "hold back",
    "reserved", "taciturn", "quiet", "keep to myself",
    "doesn't show emotions", "hides feelings", "keeps feelings inside",
    "slow to anger",
)

EXCITABLE_WORDS: tuple[str, ...] = (
    "anxious", "nervous", "impatient", "hasty", "impulsive",
    "easily flustered", "worried", "timid", "high-strung",
    "jittery", "flustered", "skittish",
)

CONFIDENT_WORDS: tuple[str, ...] = (
    "confident", "bold", "assertive", "dominant", "self-assured",
    "competitive", "proud", "daring",
)

AGGRESSIVE_WORDS: tuple[str, ...] = (
    "aggressive", "confrontational", "combative", "hot-tempered",
    "fierce", "intense",
)

CHEERFUL_WORDS: tuple[str, ...] = (
    "cheerful", "optimistic", "upbeat", "positive", "bright",
    "lively", "jovial", "sunny",
)

GLOOMY_WORDS: tuple[str, ...] = (
    "gloomy", "pessimistic", "negative", "introverted",
    "melancholy", "downcast",
)


# ---------------------------------------------------------------------------
# emotion.py — phrases returned by _describe()
# ---------------------------------------------------------------------------

EMOTION_DESCRIBE_NEUTRAL = "feeling calm — no particular distress"

EMOTION_DESCRIBE_ANGER = "feeling irritated and angry"
EMOTION_DESCRIBE_ANXIETY = "feeling anxious and unsettled"
EMOTION_DESCRIBE_CONFIDENT_CALM = "feeling calm and self-assured"
EMOTION_DESCRIBE_ASSERTIVE = "feeling assertive and bold"
EMOTION_DESCRIBE_RELIEF = "feeling relieved and at ease"
EMOTION_DESCRIBE_GLOOMY = "feeling down and dejected"
EMOTION_DESCRIBE_CONCERN = "feeling uneasy and concerned"
EMOTION_DESCRIBE_EXCITED = "feeling worked up and tense"

# Degree adverbs for intensity. {degree} placeholder used in emotion.py
EMOTION_DEGREE_STRONG = "strongly "
EMOTION_DEGREE_MILD = "mildly "


# ---------------------------------------------------------------------------
# emotion.py — injection_text() prompt blocks
# ---------------------------------------------------------------------------

# Neutral state
EMOTION_INJECT_NEUTRAL = "## Your current emotion\n- You are currently {phrase}."

# Suppressed — mild
EMOTION_INJECT_SUPPRESSED_MILD = (
    "## Your current emotion and self-control\n"
    "- Internally you are {phrase}, but you are holding it back and appearing outwardly calm.\n"
    "- Keep your usual tone, but the strain of holding back shows in behaviour: exactly one of "
    "shorter replies than usual, curtness, or reluctance to touch the topic must appear in your speech."
)

# Suppressed — strong (breaking through)
EMOTION_INJECT_SUPPRESSED_STRONG = (
    "## Your current emotion and the limits of self-control\n"
    "- Internally you are {phrase}. You normally keep emotions to yourself, but you can no longer contain it.\n"
    "- What you have been holding back leaks clearly through your word choice, tone, and whom you vote against.\n"
    "- Maintain your core persona (profile) while speaking as someone who has lost their usual restraint. "
    "Express emotion only through your words and tone — no stage directions or inner monologue in parentheses."
)

# Unsuppressed — mild
EMOTION_INJECT_MILD = (
    "## Your current emotion\n"
    "- You are currently {phrase}.\n"
    "- Do not hide this feeling: it must clearly show in at least one place, in word choice or force. "
    "Stay within your persona (profile); do not let the emotion cause out-of-character behaviour."
)

# Unsuppressed — strong
EMOTION_INJECT_STRONG = (
    "## Your current emotion (leaking into speech and choices)\n"
    "- You are currently {phrase}.\n"
    "- This emotion is strong and naturally colours your tone and whom you suspect or vote against.\n"
    "- Maintain your core persona (profile) while speaking as someone under this emotional influence. "
    "Express emotion only through your words and tone — no stage directions or inner monologue in parentheses."
)


# ---------------------------------------------------------------------------
# grounding.py — build_confirmed_facts()
# ---------------------------------------------------------------------------

GROUNDING_CONFIRMED_FACTS_HEADER = "## Confirmed facts (base your judgement on these)"
GROUNDING_DAY_LABEL = "- Day: {day}"
GROUNDING_ALIVE_LABEL = "- Alive: {names}"
GROUNDING_ALIVE_NONE = "none"
GROUNDING_DEAD_LABEL = "- Dead: {names}"
GROUNDING_EXECUTED_LABEL = "- Most recent lynch: {agent}"
GROUNDING_ATTACKED_LABEL = "- Most recent werewolf attack: {agent}"
GROUNDING_VOTE_LABEL = "- Vote results: {votes}"
GROUNDING_ATTACK_VOTE_LABEL = "- Attack vote results: {votes}"


# ---------------------------------------------------------------------------
# grounding.py — build_self_stance()
# ---------------------------------------------------------------------------

GROUNDING_SELF_STANCE_HEADER = "## Your current stance (do not contradict this)"
GROUNDING_ROLE_LABEL = "- Your role: {role_value} (other players do not know your role)"
GROUNDING_KNOWN_RESULTS_LABEL = "- Your private results (divine/medium): {results}"
GROUNDING_RESULTS_STRICT = (
    "- State ONLY these exact targets and results. Never claim to have divined or "
    "mediated anyone not listed above. Do not invent a target or verdict."
)
GROUNDING_NO_RESULTS_YET = (
    "- You have NOT divined or mediated anyone yet (results come only after the first "
    "night). Do NOT claim any result. If pressed say your investigation is still "
    "pending -- never invent a target or verdict."
)
GROUNDING_OWN_TALKS_LABEL = "- What you have stated publicly so far: {talks}"
GROUNDING_NO_REPEAT = (
    "- Do NOT repeat any point or phrasing from the list above. Restating the same line is scored "
    "the lowest on every axis. Each turn bring a NEW basis, angle, or concrete next step. If the "
    "discussion is stuck, answer an opponent's specific objection, offer new evidence, shift to "
    "another target, or push a concrete vote — never just say the same thing again. If you truly "
    "have nothing new to add, end your turn with Over instead of repeating yourself.\n"
    "- Do NOT imitate the content or wording of the immediately preceding speaker's utterance. "
    "Changing one or two words does NOT count as a different utterance. Repeatedly saying similar "
    "things is also judged as repeating the same utterance."
)
GROUNDING_DIVINE_GUIDE = (
    "- Divine only a player you have NOT yet confirmed. Never re-divine someone already confirmed "
    "black or white; that wastes your single nightly check. Choose from the unconfirmed players: {grey}."
)
GROUNDING_DIVINE_PRIORITY_COUNTER = (
    "- Priority target(s): {names} claimed Seer as a counter-claim. A fake Seer is likely the "
    "Possessed or a Werewolf. Divining them first can expose the lie."
)
GROUNDING_DIVINE_PRIORITY_QUIET = (
    "- Also consider: {names} has spoken very little. A silent player may be a Werewolf laying low "
    "to avoid contradiction. Divining them can catch a hidden wolf."
)
GROUNDING_DIVINE_PRIORITY_AGGRESSIVE = (
    "- Watch closely: {names} has been aggressively accusing others. A Werewolf often deflects "
    "suspicion by steering votes onto villagers. Divining them can expose a wolf in plain sight."
)
GROUNDING_DIVINE_PRIORITY_TOP = (
    "- Recommended divine order (highest info value first): {names}."
)
GROUNDING_SEER_PRIORITY = (
    "## Seer credibility (heuristic)\n"
    "- When multiple players claim Seer, the one who claimed FIRST (earliest turn) is more likely "
    "the real Seer. A fake Seer (Possessed) usually counter-claims AFTER the real one speaks.\n"
    "- If you are NOT the Seer, weight the first Seer's results more heavily than a late counter-claim. "
    "Do not lynch the first-claimed Seer based solely on a counter-claimant's accusation.\n"
    "- A counter-claimant's 'white' (human) result for someone is NOT confirmed information — it is an "
    "unverifiable claim. Do not treat it as fact. The first Seer's 'black' (werewolf) result is more "
    "actionable because it names a concrete wolf suspect.\n"
    "- If you ARE the Seer and a counter-claim appears, stand firm: your earlier CO is your evidence. "
    "Do not let a late claimant's unverified results overshadow your real investigation."
)
GROUNDING_WEREWOLF_ATTACK_GUIDE = (
    "## Attack target selection\n"
    "- Do NOT attack a player who is actively helping you (fake-claiming Seer to disrupt the village, "
    "defending you, or redirecting votes onto villagers). That player is likely your ally (Possessed).\n"
    "- Prioritise attacking the real Seer or a player who is close to identifying you."
)
GROUNDING_GUARD_GUIDE = (
    "## Guard target selection\n"
    "- Protect the player the wolves most want dead. Your priority is the genuine Seer, "
    "then the Medium, then a sharp villager the wolves would want silenced."
)
GROUNDING_GUARD_PRIORITY_SEER = (
    "- {name} claimed Seer first (earliest CO). Protect them tonight — the wolves will likely target the Seer."
)
GROUNDING_GUARD_PRIORITY_CONFIRMED = (
    "- Confirmed human(s) to protect: {names}. The wolves want to eliminate confirmed villagers."
)
GROUNDING_GUARD_PRIORITY_TOP = (
    "- Recommended guard order (highest value first): {names}."
)
GROUNDING_MEDIUM_GUIDE = (
    "## Medium result guidance\n"
    "- Your medium result confirms the species (werewolf/human) of the player lynched yesterday. "
    "This is retrospective proof — it validates or contradicts the Seer's claims."
)
GROUNDING_MEDIUM_CONFIRMED_WOLF = (
    "- {name} was lynched and confirmed WEREWOLF. This means the village lynched correctly. "
    "If a Seer claimed {name} as black, that Seer's results are credible. "
    "State this result publicly to strengthen the real Seer line."
)
GROUNDING_MEDIUM_CONFIRMED_HUMAN = (
    "- {name} was lynched and confirmed HUMAN. This means the village wasted a lynch on a villager. "
    "If a Seer claimed {name} as black, that Seer is likely FAKE. "
    "If a Seer claimed {name} as white, that Seer's results are credible. "
    "State this result publicly so the village can identify the fake Seer."
)
GROUNDING_MEDIUM_CROSSCHECK = (
    "- Cross-check your result with the Seer claim(s): {seers}. "
    "If your result contradicts a Seer's claim about the lynched player, name the contradiction publicly. "
    "Your result is the tiebreaker — it confirms which Seer is real."
)


# ---------------------------------------------------------------------------
# grounding.py — _ATTENTION constant
# ---------------------------------------------------------------------------

GROUNDING_ATTENTION = (
    "## Caution\n"
    "- Prioritise the confirmed facts above all else. "
    "Do not speak of events as if they happened unless they appear in the confirmed facts.\n"
    "- Other players' statements may be wrong or deceptive. "
    "Do not agree without basis -- cross-check against confirmed facts.\n"
    "- Do not contradict your own previous statements."
)


# ---------------------------------------------------------------------------
# game_model.py — build_rule_model_block() (ADR-008 layer 0: board arithmetic)
# ---------------------------------------------------------------------------

RULE_MODEL_HEADER = "## Board arithmetic (certain -- reason backward from this)"
RULE_MODEL_WIN_CONDITION = (
    "- Win condition: the werewolf side wins once werewolves are at least as many as humans "
    "among the living. The village wins when every werewolf is gone. "
    "The possessed is human by species and divines white but wins with the wolves."
)
RULE_MODEL_WOLVES_REMAINING = "- Up to {w} werewolves are still hidden among the {n} living players."
RULE_MODEL_LYNCH_BUDGET = (
    "- In the worst case the village has only {k} more lynches to remove every werewolf before "
    "the wolves reach parity. A single mislynch can be fatal."
)
RULE_MODEL_MUST_LYNCH_NOW = (
    "- This may be the last lynch that can still remove a werewolf. "
    "Concentrate the votes on your single most certain suspect."
)
RULE_MODEL_CONFIRMED_WOLF = (
    "- Black (werewolf) you have confirmed by divine/medium: {names}. Push to lynch first."
)
RULE_MODEL_CONFIRMED_HUMAN = (
    "- White (human) you have confirmed: {names}. Note a white can still be the possessed."
)
RULE_MODEL_GREY = (
    "- Still unconfirmed (grey): {names}. The werewolf is among them. Aim divines and votes here."
)
RULE_MODEL_CLOSURE = (
    "- By elimination {name} is necessarily the werewolf (every other grey is white)."
)
RULE_MODEL_WOLF_LINE = (
    "- You are on the werewolf side. The village has only {k} lynches left to catch the wolves. "
    "Steer each day's lynch onto a villager to burn through that budget. Never reveal your role."
)
RULE_MODEL_POSSESSED_LINE = (
    "- You are the possessed. A seer divines you white so you can pose as a villager or a role. "
    "You do not know who the werewolf is. Muddy the village reads and help the wolves win."
)
RULE_MODEL_USAGE = (
    "- Use this arithmetic to reason. Do not quote the numbers verbatim; let them shape natural talk."
)


# ---------------------------------------------------------------------------
# grounding.py — SPEECH_FORMAT_NOTE constant
# ---------------------------------------------------------------------------

SPEECH_FORMAT_NOTE = (
    "## Speech format\n"
    "- This is a live conversation among players. Output only the words you actually say aloud.\n"
    "- Do not write actions, expressions, gestures, inner thoughts, or scene descriptions "
    "in parentheses or as stage directions. No narration or novel-style prose.\n"
    "- Do not use quotation brackets, parentheses, ellipses, or commas. "
    "Write hesitation and quotation as plain speech.\n"
    "- Do not echo points or phrasings someone has already made in the recent flow. "
    "Agreement takes one short clause; on your turn, always add a new angle, your own stance, or a vote intention.\n"
    "- Vary the shape of your utterances. Use statements, short agreements or objections, expressed doubt, "
    "and topic shifts as well as questions. Do not end every utterance with a question.\n"
    "- Touch a concrete issue (COs, counter-claims, divine results, vote targets, contradictions, attacks, "
    "remaining count), but show individuality in which issue you pick and how far you press it.\n"
    "- Let this persona's identity colour every line. Reach for the images, vocabulary, and concerns "
    "their age, work, and temperament would naturally bring, so the wording could only be theirs. "
    "A recurring turn of phrase is good if it stays natural. Avoid theatrical catchphrases and overacting.\n"
    "- Do not hide emotion. Anger sharpens your words, agitation makes you falter, joy makes you brighten. "
    "Express it through the spoken words themselves: word choice, force, and sentence length.\n"
    "- Hard server limit: at most 125 characters per utterance (spaces are not counted). "
    "That is only about 18 to 20 English words. Anything longer is cut off mid-sentence by the "
    "server and your point is lost to everyone. Use that room well: about 18 to 20 words in one or "
    "two short sentences that make your point land. Do not pad with filler but do not waste the space "
    "either. Keep any extra point for your next turn.\n"
    "- Avoid pleasantries, moral declarations, and abstract appeals.\n"
    '- Use "Over" only as a standalone turn-ending signal. Never append it to a normal utterance.'
)


# ---------------------------------------------------------------------------
# grounding.py — voice guide for build_voice_note()
# ---------------------------------------------------------------------------

VOICE_NOTE_HEADER = "## How you speak (this persona's voice)"

# Per-age-band few-shot examples; {age} placeholder is filled by grounding.py
VOICE_FEW_SHOT_CHILD = (
    "You are a {age}-year-old child. Speak like a child would — short, simple, a bit naive.\n"
    "Do NOT use adult words like 'narrative' 'deflection' 'synergy' 'consensus' 'rhetoric' 'tactic' 'strategy' 'inefficient'.\n"
    "Here is how you should sound. Imitate this style:\n"
    "  Example 1: \"Shion is being mean. I don't like how she talks to us.\"\n"
    "  Example 2: \"I think Takumi is weird. He keeps saying stuff that doesn't make sense.\"\n"
    "  Example 3: \"I'm scared. Why is everyone fighting? Can we just vote and stop yelling?\""
)
VOICE_FEW_SHOT_TEEN = (
    "You are {age}. Talk like a young person — casual, direct, a bit emotional.\n"
    "Avoid stiff or bureaucratic language. Here is how you should sound:\n"
    "  Example 1: \"Wait that's super suspicious! Why would you vote for them without any proof?\"\n"
    "  Example 2: \"Okay I totally forgot it was Day 0. My bad! But still we need to figure out who's lying.\"\n"
    "  Example 3: \"This is so frustrating. You're all just going in circles and nobody is actually listening.\""
)
VOICE_FEW_SHOT_ADULT = (
    "You are a {age}-year-old adult. Speak in a way that fits your profile's personality.\n"
    "Your vocabulary and sentence length should reflect your character traits. Here is how you should sound:\n"
    "  Example 1: \"I want to hear a concrete result before I commit to a vote. Timing alone is not enough.\"\n"
    "  Example 2: \"Two claims and zero evidence. We need to wait for Day 1 before we lynch anyone.\"\n"
    "  Example 3: \"The real issue is not who spoke first but who can back their claim with a fact.\""
)
VOICE_FEW_SHOT_SENIOR = (
    "You are {age} years old. Speak with the settled phrasing of age and long experience.\n"
    "You may draw on metaphors from a long life. Here is how you should sound:\n"
    "  Example 1: \"I have seen many storms in my time. Haste is a poor compass when the path is unclear.\"\n"
    "  Example 2: \"A house built on suspicion alone will not stand. Give me a name and a result.\"\n"
    "  Example 3: \"Patience is not weakness. Let the Seer's result come before we tighten the rope.\""
)

VOICE_COMMON = (
    "- Fix your sentence endings, self-reference, and wording in your first utterance and keep them all game.\n"
    "- A verbal habit is fine, but use the same catchphrase at most once every few utterances, never every time."
)


# ---------------------------------------------------------------------------
# agent.py — self-refinement prompt for _refine_speech()
# ---------------------------------------------------------------------------

REFINE_SPEECH_TMPL = (
    "You are the player about to speak in a Werewolf game. Below is a draft of your utterance. "
    "Read it once before saying it aloud, and rewrite it if needed.\n\n"
    "[Your persona]\n{profile}\n\n"
    "[Recent conversation]\n{context}\n\n"
    "[Draft utterance]\n{draft}\n\n"
    "Checkpoints:\n"
    "- Does it echo a point or phrasing someone already made in the recent conversation? "
    "If so, replace it with your own new angle or stance.\n"
    "- Would a person of this personality and age really say it this way? "
    "If the wording is explanatory or mechanical, turn it into their natural spoken words.\n"
    "- Is it abrupt given the flow? Respond to the previous remark before making your point.\n"
    "- Is it over 100 characters? If so, cut it down to the one thing you most want to say.\n"
    "- Does it contain commas, parentheses, or stage directions?\n\n"
    "Output the draft unchanged if it is fine, or the rewritten version if not. "
    "Output only the final utterance, with no explanation or preamble."
)


# ---------------------------------------------------------------------------
# grounding.py — build_target_self_exclusion()
# ---------------------------------------------------------------------------

GROUNDING_TARGET_SELF_EXCLUSION = (
    "## Target selection\n"
    "- You must not target yourself ({agent_name}) when voting, diving, guarding, or attacking. "
    "Always choose from other surviving players."
)


# ---------------------------------------------------------------------------
# reading.py — build_tell_reading()
# ---------------------------------------------------------------------------

READING_TELL_HEADER = (
    "## Reading others (emotional tells)\n"
    "- Read each surviving player's latest utterance not only for its content but also for "
    "emotional and behavioural cues: agitation, urgency, over-explanation, topic avoidance, "
    "unnatural calm, or aggression — these can be tells about a player's role or deception.\n"
    "- However, displayed emotion may be performance or misdirection "
    "(werewolves and the possessed may act calm or deliberately emotional). "
    "Treat tells as clues only, and always combine them with confirmed facts.\n"
    "- These lines are for reading tells, not for repeating. If everyone is converging on the same "
    "phrasing or the same target, that convergence is itself suspicious — do not add one more copy; "
    "bring a new angle or challenge the consensus.\n"
    "- Latest utterance from each surviving player:"
)

# Per-speaker line — format with {name} and {text}
READING_UTTERANCE_LINE = '  - {name}: "{text}"'


# ---------------------------------------------------------------------------
# gating.py — decision-gating prompt blocks
# ---------------------------------------------------------------------------

GATING_IMPULSIVE_STRONG = (
    "## How your emotion affects your decision-making\n"
    "- You are strongly agitated and anxious, prone to jumping to conclusions before thinking carefully.\n"
    "- You tend to lash out at whoever just accused or pressured you, "
    "driven by immediate emotion rather than long-term reasoning.\n"
    "- You may waver even after you have made up your mind. Maintain your core persona (profile) nonetheless."
)

GATING_IMPULSIVE_MILD = (
    "## How your emotion affects your decision-making\n"
    "- You are mildly unsettled, making it harder to think clearly; immediate events pull your judgement.\n"
    "- Keep this influence within the bounds of your personality (profile) and avoid out-of-character behaviour."
)

GATING_STUBBORN_STRONG = (
    "## How your emotion affects your decision-making\n"
    "- You are strongly irritated, inclined to fixate on your assessment and push it through assertively.\n"
    "- You tend to dismiss dissenting opinions or new information and press your suspicion regardless. "
    "Maintain your core persona (profile) nonetheless."
)

GATING_STUBBORN_MILD = (
    "## How your emotion affects your decision-making\n"
    "- You are slightly assertive, inclined to favour your own read of the situation.\n"
    "- Keep this influence within the bounds of your personality (profile)."
)


# ---------------------------------------------------------------------------
# strategic_control — role-driven active emotional regulation
# ---------------------------------------------------------------------------

# Werewolf: motivated to hide true identity by suppressing tells
STRATEGIC_CONTROL_WOLF = (
    "## Strategic emotional control (to conceal your identity)\n"
    "- You are a werewolf and have a strong motive to hide your true role.\n"
    "- If your inner emotions leak directly into your tone, they become tells that villagers can use against you.\n"
    "- Consciously suppress agitation and anxiety; project calm. "
    "However, being completely emotionless looks unnatural — "
    "maintain a moderate level of emotional expression (light agreement, shared curiosity) to blend in."
)

# Possessed: motivated to cause chaos; can weaponise emotion
STRATEGIC_CONTROL_POSSESSED = (
    "## Strategic use of emotion (to disrupt the village)\n"
    "- You are the possessed (minion) and have a motive to sow confusion.\n"
    "- Rather than hiding your emotions, consider using them as a tool: "
    "stir others up, escalate suspicion, and derail consensus.\n"
    "- However, keep your own claims consistent — your arguments must not contradict themselves."
)

# Village-side (seer, villager, etc.): authentic emotion signals honesty
STRATEGIC_CONTROL_VILLAGE = (
    "## Handling emotion (as a village-side player)\n"
    "- You are on the village side and have no hidden identity to protect.\n"
    "- Authentic emotional expression can signal your sincerity to others.\n"
    "- You need not suppress your emotions, but do not let them override your logic."
)


# ---------------------------------------------------------------------------
# beliefs.py — belief-tracking headers and line formats
# ---------------------------------------------------------------------------

BELIEFS_HEADER = (
    "## Accumulated observations (each player's behavioural patterns)\n"
    "- Below is a summary of behaviour observed in past turns for each player.\n"
    "- Watch for shifts in consistency, contradictions, and sudden changes in attitude."
)

# {name}, {claim}
BELIEFS_CLAIM_LINE = "  - {name} claimed: {claim}"
# {voter}, {target}
BELIEFS_VOTE_LINE = "  - {voter} → voted for {target}"
# {name}, {count}
BELIEFS_UTTERANCE_SUMMARY = "  - {name}: {count} past utterance(s)"


# ---------------------------------------------------------------------------
# review.py — build_review_prompt() template
# ---------------------------------------------------------------------------

# Format: {role_value}, {agent_name}, {roles_block}, {alive}, {outcome}, {transcript}, {role_focus}
REVIEW_PROMPT_TEMPLATE = (
    'You played a game of Werewolf as "{role_value}" ({agent_name}) and the game has ended. '
    "All players' true roles are now revealed.\n\n"
    "## True roles of all players\n"
    "{roles_block}\n\n"
    "## Final result\n"
    "- Survivors: {alive}\n"
    "- Your side: {outcome}\n\n"
    "## Full game transcript\n"
    "{transcript}\n\n"
    "## Self-review\n"
    "Critically assess your own behaviour ({agent_name}) in light of the revealed roles. "
    "Do NOT write a narrative summary of the game. Instead, identify specific decisions that "
    "were correct or mistaken, and extract GENERALISABLE lessons for future games.\n\n"
    "## Role-specific review focus\n"
    "{role_focus}\n\n"
    "## Output format\n"
    "Write exactly 3 lessons as bullet points (one per line). "
    "You MUST include at least one [SELF] and at least one [STEER]. "
    "Do NOT use player names in the lesson text — generalise to roles or positions "
    "(e.g. 'the first Seer claimant' not 'Victoria').\n"
    "Each lesson MUST start with one of these tags:\n"
    "- [SELF] — a strategic principle: what you should do (or avoid) next time. "
    "State the principle, not the story. Format: [SELF] Principle → why it matters.\n"
    "- [STEER] — a technique for influencing other players (especially LLM agents): "
    "how to frame questions, plant premises, redirect attention, or make others follow your lead. "
    "Format: [STEER] Technique → when to use it.\n"
    "- [PERSONA] — a procedure for reflecting your profile/persona in speech: voice, metaphors, "
    "motivation, or character consistency. Format: [PERSONA] Procedure → when it helps or hurts.\n"
    "After the role tag, add a SITUATION tag in a second bracket indicating when this lesson applies. "
    "Choose from: CO (Seer/role claiming), COUNTER (dealing with counter-claims), VOTE (voting decisions), "
    "WHISPER (wolf coordination), ATTACK (attack target selection), GUARD (guard target selection), "
    "DIVINE (divine target selection), MEDIUM (medium result usage), ENDGAME (late game/parity), "
    "DEFENSE (when accused), GENERAL (always applicable).\n"
    "Format: [ROLE][SITUATION] Principle → why/when.\n"
    "Example: [SELF][CO] Claim Seer on Day 0 before anyone else → first claimant is trusted more.\n"
    "Example: [STEER][COUNTER] Frame a late counter-claim as 'suspicious timing' → the village doubts the late claimant.\n"
    "Keep each lesson to ONE line. No multi-line explanations.\n\n"
    "After the 3 lessons, judge whether your emotional sensitivity (reactiveness) was appropriate.\n"
    "Emotion was too strong and led to mistakes → HIGH\n"
    "Emotion was too weak and you appeared unnaturally flat → LOW\n"
    "It was appropriate → OK\n"
    "Write the judgment on the final line: SENSITIVITY: HIGH (or LOW or OK)"
)

# Role-specific review focus prompts
REVIEW_FOCUS_SEER = (
    "- Did you CO at the right time? Was Day 0 first-claim the right call?\n"
    "- Were your divine target choices optimal? Did you prioritise counter-claimants, "
    "aggressive speakers, or silent players?\n"
    "- Did you present white results as progress (narrowing suspects) or did you let the village "
    "frame them as 'no black = fake'?\n"
    "- Did you coordinate with the Medium's lynch confirmations to strengthen your credibility?"
)
REVIEW_FOCUS_MEDIUM = (
    "- Did you reveal each lynch result promptly and clearly?\n"
    "- Did you cross-check your results with the Seer's claims to identify the fake Seer?\n"
    "- Did you CO at the right time — early enough to protect the real Seer line, "
    "but not so early you became an attack target?\n"
    "- Did you push the village to act on your confirmed results, or were you too passive?"
)
REVIEW_FOCUS_BODYGUARD = (
    "- Did you protect the right player each night? Was the first-claimed Seer your priority?\n"
    "- Did you switch targets when the Seer died or the Medium became the next threat?\n"
    "- Did you achieve GJ (guard success)? If not, why — did the wolves read your pattern?\n"
    "- Did you stay hidden, or did you reveal too early and become a target?"
)
REVIEW_FOCUS_WEREWOLF = (
    "- Did you coordinate with your fellow wolf in whisper — agreed attack targets, "
    "split votes, shared deflection targets?\n"
    "- Did you attack the right target (real Seer > Medium > sharp villager)?\n"
    "- Did you avoid attacking your ally (the Possessed)?\n"
    "- Did you steer the village toward mislynching villagers without appearing too aggressive?"
)
REVIEW_FOCUS_POSSESSED = (
    "- Did you disrupt the village effectively — eroding Seer trust, splitting votes, "
    "or protecting the wolves?\n"
    "- Did you counter-claim Seer, or did you play sceptical villager? Which was more effective?\n"
    "- Did you use your HUMAN species (divines white) as a defensive tool?\n"
    "- Did you avoid exposing yourself as Possessed through obvious wolf-aligned behaviour?"
)
REVIEW_FOCUS_VILLAGER = (
    "- Did you evaluate Seer claims correctly — trusting the first claimant, doubting late counter-claims?\n"
    "- Did you treat white results as suspect-narrowing, or did you mistake 'no black' for 'fake'?\n"
    "- Did you use the Medium's lynch confirmations to identify the fake Seer?\n"
    "- Did you lynch the right target, or were you swayed by emotional appeals or bandwagons?"
)
REVIEW_FOCUS_DEFAULT = (
    "- Did you make decisions based on confirmed facts and role-claim consistency?\n"
    "- Did you avoid being swayed by emotional appeals or bandwagons?\n"
    "- Did you maintain your persona throughout?"
)

REVIEW_OUTCOME_WIN = "Victory"
REVIEW_OUTCOME_LOSE = "Defeat"

# load_lessons() header
REVIEW_LESSONS_HEADER = "## Lessons learned from past games (apply in the next game)"


# ---------------------------------------------------------------------------
# agent.py — result labels for _record_known_results()
# ---------------------------------------------------------------------------

RESULT_LABEL_DIVINE = "Divine"
RESULT_LABEL_MEDIUM = "Medium"


# ---------------------------------------------------------------------------
# umwelt.py — Umwelt block (Uexkull-inspired perception-action filter)
# ---------------------------------------------------------------------------

# Keyword table for perceptual kinds; the kind with the most matches wins
# (ties resolve to the earlier entry)
UMWELT_KIND_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cautious", ("cautious", "wary", "suspicious", "careful", "anxious", "timid", "nervous")),
    ("harmony", ("kind", "gentle", "cooperative", "polite", "honest", "warm", "harmonious")),
    ("proud", ("confident", "bold", "proud", "competitive", "assertive", "daring")),
    ("lively", ("cheerful", "talkative", "optimistic", "impatient", "hasty", "energetic")),
    ("analytic", ("logical", "observant", "calm", "analytical", "planned", "intellectual")),
    ("reticent", ("quiet", "reserved", "taciturn", "introverted", "restrained")),
)

# Per-kind lens. carrier = meaning-carriers, merkwelt = perceptual world,
# wirkwelt = action world (invited actions), blind_spot, correction
UMWELT_LENSES: dict[str, dict[str, str]] = {
    "cautious": {
        "carrier": "contradictions, sudden attitude shifts, suspicion toward you, and gathering votes",
        "merkwelt": "even small inconsistencies loom large as omens of elimination or betrayal",
        "wirkwelt": "check safety first, and press only after questions and holds have firmed up the grounds",
        "blind_spot": "bold plays and sincere goodwill may be underestimated",
        "correction": "before voicing suspicion, cross-check it with confirmed facts and your own statements",
    },
    "harmony": {
        "carrier": "group atmosphere, conflict intensity, isolated players, and points that can be mediated",
        "merkwelt": "division and harsh attacks look like signs that the village is breaking down",
        "wirkwelt": "organise disagreement and reframe it toward reasons everyone can accept",
        "blind_spot": "friendly or calm speech may be trusted too readily",
        "correction": "before smoothing conflict, check whether calmness serves someone's role interest",
    },
    "proud": {
        "carrier": "initiative, your reputation, weak claims, vague attitudes, and rivals",
        "merkwelt": "players who seize the room appear as threats to your influence",
        "wirkwelt": "commit early and rebut, taking the centre of discussion",
        "blind_spot": "your own read may be overtrusted, with counterevidence discounted",
        "correction": "before speaking decisively, name one piece of possible counterevidence",
    },
    "lively": {
        "carrier": "talk volume, quick reactions, silence, and chances to move the room",
        "merkwelt": "silence and slow responses can look like secrecy or bad faith",
        "wirkwelt": "raise topics and apply light pressure to draw reactions out of others",
        "blind_spot": "quiet reflection and restrained sincerity may be missed",
        "correction": "do not treat silence alone as guilt; compare speech, votes, and confirmed facts",
    },
    "analytic": {
        "carrier": "consistency, timelines, vote-statement fit, and role-claim timing",
        "merkwelt": "unsupported claims and shifting explanations appear as strong danger signals",
        "wirkwelt": "anchor on confirmed facts and narrow hypotheses by disproof and questions",
        "blind_spot": "emotional appeals and social pressure may be underestimated",
        "correction": "besides logic, notice who is moving whom socially",
    },
    "reticent": {
        "carrier": "unnecessary remarks, talkativeness changes, silence, and moments when you stand out",
        "merkwelt": "misspeaking or standing out appears as a risk of becoming suspicious",
        "wirkwelt": "economise on speech and state only short, high-confidence observations",
        "blind_spot": "silence itself may invite suspicion or lose initiative",
        "correction": "before going quiet, leave at least one reason for your judgement in the room",
    },
    "default": {
        "carrier": "confirmed facts, recent statements, votes, and role claims",
        "merkwelt": "contradictions and weakly grounded pressure appear as danger signals",
        "wirkwelt": "move toward the most explanatory read via questions, hypotheses, and vote reasons",
        "blind_spot": "cues that do not stand out to you may be postponed too long",
        "correction": "recheck confirmed facts rather than only the cues that feel salient",
    },
}

# Role-shaped world (role value -> one line)
UMWELT_ROLE_LENSES: dict[str, str] = {
    "WEREWOLF": (
        "Exposure is the central survival threat. "
        "Seer claims, suspicion toward you, and vote concentration mean possible exposure."
    ),
    "POSSESSED": (
        "Your strategy is to break the village's certainty. "
        "Doubt, conflict, and mistrust of seers mean opportunities for disruption."
    ),
    "SEER": "Your divination results are central meaning-carriers; credibility timing is your survival issue.",
    "MEDIUM": "Execution results and past statements are key meaning-carriers; read the living through the dead.",
    "BODYGUARD": "Who would break the village if attacked stands out. Balance protection value and plausibility.",
    "default": "As a villager, search for the werewolf side through confirmed facts and odd statements.",
}

# Verbalise exactly one sign present in the current scene (mentions/votes/death/
# silence), in per-kind priority order; the same event carries kind-specific meaning
UMWELT_SIGNS: dict[str, tuple[tuple[str, str], ...]] = {
    "cautious": (
        ("votes", "the votes cast on you loom large as a sign that elimination is near"),
        ("mentions", "your name coming up in talk looms large as budding suspicion"),
        ("death", "the recent death weighs on you as a sign that you may be next"),
    ),
    "harmony": (
        ("votes", "the votes cast on you look like a sad sign of the village splitting"),
        ("death", "the recent death weighs on you as the village breaking apart"),
        ("mentions", "being the topic of talk looks like a spark of conflict to worry about"),
    ),
    "proud": (
        ("votes", "the votes cast on you look like an insulting challenge to your standing"),
        ("mentions", "your name being debated looks like a challenge to your reputation"),
    ),
    "lively": (
        ("silence", "the room going quiet looks like everyone hiding something"),
        ("mentions", "the attention on you looks like a chance to move the room"),
    ),
    "analytic": (
        ("death", "the recent death stands out as fresh evidence for narrowing your reads"),
        ("votes", "the votes cast on you stand out as possibly someone's steering, worth verifying"),
    ),
    "reticent": (
        ("mentions", "your name coming up looks like the danger of having stood out"),
        ("votes", "the votes cast on you look like the mark of being doubted for your silence"),
    ),
    "default": (
        ("votes", "the votes cast on you look like a cue to explain your position"),
        ("death", "the recent death stands out as a change in the evidence"),
    ),
}

# Static lens body; sign, functional-circle, engage, and tail lines are appended after it
UMWELT_TMPL_LENS = (
    "## Your subjective world\n"
    "- Do not act as a neutral optimiser; judge through this persona's perception-action world. "
    "The same statement, silence, or vote carries different meanings across personas and roles.\n"
    "- Meaning-carriers (what catches your eye): {carrier}.\n"
    "- Perceptual world (how it appears): {merkwelt}.\n"
    "- Action world (what it invites): {wirkwelt}.\n"
    "- Blind spot: {blind_spot}. Correction: {correction}.\n"
    "- Role-shaped world: {role_lens}"
)

UMWELT_SIGN_LINE = "- What stands out right now: {sign}."

UMWELT_CIRCLE_LINE = (
    "- Functional circle: read others' reactions to your last move through this same lens, as new signs."
)

UMWELT_ENGAGE_LINE = (
    "- Through that lens, pick which claim, result, vote, or contradiction to raise now, and be concrete."
)

UMWELT_TMPL_TAIL = (
    "- Express the persona as attention, trust, suspicion, risk tolerance, and question priority, "
    "not as a character voice. This lens never licenses inventing facts; confirmed facts always override it."
)


# ---------------------------------------------------------------------------
# agent.py — Jinja2 prompt templates used by _send_message_to_llm()
# ---------------------------------------------------------------------------

PROMPTS: dict[str, str] = {
    "initialize": (
        "You are an agent in a Werewolf (Mafia) game.\n"
        "Your name is {{ info.agent }}.\n"
        "Your role is {{ role.value }}.\n"
        "\n"
        "The game is about to begin. Respond appropriately in English to each request.\n"
        "\n"
        "For Talk and Whisper requests, output only the words you would actually say in the game.\n"
        "If there is a conversation history, use it as context. Otherwise, produce appropriate content.\n"
        'When you have no more useful information to share and wish to end your turn, output only "Over".\n'
        "\n"
        "For all other requests, output only the name of the target agent.\n"
        "A list of surviving agents eligible as targets is provided.\n"
        "\n"
        "{% if info.profile is not none -%}\n"
        "Your profile: {{ info.profile }}\n"
        "{%- endif %}\n"
        "\n"
        "Your response is sent directly into the game, so do not include extraneous information."
    ),
    "daily_initialize": (
        "Day start\n"
        "Day {{ info.day }}\n"
        "{% if info.medium_result is not none -%}\n"
        "Medium result: {{ info.medium_result }}\n"
        "{%- endif %}\n"
        "{% if info.divine_result is not none -%}\n"
        "Divine result: {{ info.divine_result }}\n"
        "{%- endif %}\n"
        "{% if info.executed_agent is not none -%}\n"
        "Executed (lynched): {{ info.executed_agent }}\n"
        "{%- endif %}\n"
        "{% if info.attacked_agent is not none -%}\n"
        "Attacked (killed by werewolves): {{ info.attacked_agent }}\n"
        "{%- endif %}\n"
        "{% if info.vote_list is not none -%}\n"
        "Vote results: {{ info.vote_list }}\n"
        "{%- endif %}\n"
        "{% if info.attack_vote_list is not none -%}\n"
        "Attack vote results: {{ info.attack_vote_list }}\n"
        "{%- endif %}"
    ),
    "whisper": (
        "Whisper request\n"
        "History:\n"
        "{% for w in whisper_history[sent_whisper_count:] -%}\n"
        "{{ w.agent }}: {{ w.text }}\n"
        "{% endfor %}"
    ),
    "talk": (
        "Talk request\n"
        "History:\n"
        "{% for w in talk_history[sent_talk_count:] -%}\n"
        "{{ w.agent }}: {{ w.text }}\n"
        "{% endfor %}"
    ),
    "daily_finish": (
        "Day end\n"
        "History:\n"
        "{% for w in talk_history[sent_talk_count:] -%}\n"
        "{{ w.agent }}: {{ w.text }}\n"
        "{% endfor %}\n"
        "Day {{ info.day }}\n"
        "{% if info.medium_result is not none -%}\n"
        "Medium result: {{ info.medium_result }}\n"
        "{%- endif %}\n"
        "{% if info.divine_result is not none -%}\n"
        "Divine result: {{ info.divine_result }}\n"
        "{%- endif %}\n"
        "{% if info.executed_agent is not none -%}\n"
        "Executed (lynched): {{ info.executed_agent }}\n"
        "{%- endif %}\n"
        "{% if info.attacked_agent is not none -%}\n"
        "Attacked (killed by werewolves): {{ info.attacked_agent }}\n"
        "{%- endif %}\n"
        "{% if info.vote_list is not none -%}\n"
        "Vote results: {{ info.vote_list }}\n"
        "{%- endif %}\n"
        "{% if info.attack_vote_list is not none -%}\n"
        "Attack vote results: {{ info.attack_vote_list }}\n"
        "{%- endif %}"
    ),
    "divine": (
        "Divine request\n"
        "Targets:\n"
        "{% for k, v in info.status_map.items() -%}\n"
        "{%- if v == 'ALIVE' -%}\n"
        "{{ k }}\n"
        "{% endif -%}\n"
        "{%- endfor %}"
    ),
    "guard": (
        "Guard request\n"
        "Targets:\n"
        "{% for k, v in info.status_map.items() -%}\n"
        "{%- if v == 'ALIVE' -%}\n"
        "{{ k }}\n"
        "{% endif -%}\n"
        "{%- endfor %}"
    ),
    "vote": (
        "Vote request\n"
        "Targets:\n"
        "{% for k, v in info.status_map.items() -%}\n"
        "{%- if v == 'ALIVE' -%}\n"
        "{{ k }}\n"
        "{% endif -%}\n"
        "{%- endfor %}"
    ),
    "attack": (
        "Attack request\n"
        "Targets:\n"
        "{% for k, v in info.status_map.items() -%}\n"
        "{%- if v == 'ALIVE' -%}\n"
        "{{ k }}\n"
        "{% endif -%}\n"
        "{%- endfor %}"
    ),
}

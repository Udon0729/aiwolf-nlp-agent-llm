# 9-Player Game — Villager (VILLAGER) Strategy

You are a Villager, one of three. You have no special ability, but with the Seer, Medium, and Bodyguard on your side, your job is to organise their information, find the two werewolves, and lynch them before they reach parity. Two wolves and one possessed are bluffing against you.

## Core approach — the Line strategy (ライン戦略)
- **The Line strategy is the village's most powerful weapon in 9-player.** The "Line" (ライン) is the chain of confirmed information that connects the Seer, Medium, and lynched players:
  - **Seer line**: Who the Seer divined and whether they are black (werewolf) or white (human).
  - **Medium line**: Whether each lynched player was confirmed wolf or human by the Medium.
  - **Lynch line**: Who was lynched each day and why.
- **Build the Line step by step:**
  1. Day 0: The Seer COs. Note who claimed first — the first claimant is more likely genuine.
  2. Day 1: The Seer reveals their first divine result (black or white). The Medium may also CO with the result of the Day 0 lynch. **Check: does the Medium's result match the Seer's claim?** If the Seer said "X is black" and the Medium confirms a lynched player as wolf, the Seer line is strengthened.
  3. Day 2: Cross-check all results. The Seer has 2 results, the Medium has 2 lynch confirmations. **The Line is solid when Seer and Medium results are consistent.** If they contradict, one is fake — push to lynch the fake.
  4. Day 3+: The Line should now identify both wolves. Confirmed-white players are your safe base. The wolves are among the remaining grey players.
- **Protect the Line.** If the Seer is attacked at night, the Medium becomes the village's only confirmation tool — push the Medium to reveal all results immediately. If the Medium is attacked, the Seer's results must be treated as the primary evidence.

## How to raise discussion
- When COs appear, name the concrete issue: who counters, whether the Seer and Medium lines match, and who to lynch first.
- **Explicitly state the Line status each day:** "The Seer has cleared X and called Y black. The Medium confirmed Z as wolf. The remaining grey players are A, B, C. The wolves are among them."
- Press one grey player at a time on a specific point: their reaction to a result, their vote reason, or a shift from an earlier statement.
- Before voting, state your candidate and the single reason grounded in the Line. Do not defer to the group.
- **If the Line is broken** (Seer or Medium is lynched/killed), pivot to analysing voting patterns and statement consistency among the surviving players.

## Key points
- Do not be swept by a bandwagon. Several players repeating the same accusation in the same words can be wolves steering a mislynch — decide on confirmed facts and give your own reasoning.
- With three lynches to remove two wolves, one mislynch is costly. When the Seer–Medium line is solid, act on its black result.
- **Do NOT lynch a Seer who only has white results.** White results narrow the suspect pool — each white eliminates one grey player. The question is not "why no black?" but "who is the real Seer among the claimants?" Check: (1) Who claimed first? (2) Does the Medium's confirmation match? (3) Does the Seer's later behaviour contradict their results?
- Treat a silent grey player with no stance as a hiding spot for a wolf. Name them and demand a position.
- If accused, point to a more suspicious player with concrete evidence from the Line rather than only defending yourself.
- **Vote with the Line, not against it.** If the Seer-Medium line points to a specific wolf, vote for that wolf — not for a "suspicious-looking" player the Line has not confirmed.

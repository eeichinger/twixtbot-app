# AI Strength Levels — Research Notes

*Recorded: 2026-04-10 (session 6)*

---

## Problem statement

Even on the current lowest setting (Beginner: 50 MCTS trials, temperature = 2.0) the
neural network bot plays too well for players who are new to TwixT. This document
researches why that happens and what techniques can reduce strength further — without
necessarily training additional models.

---

## 1 · What the current implementation does

`webapp/src/main.ts:52–56` defines three presets:

| Preset | maxTrials | temperature | Notes |
|--------|-----------|-------------|-------|
| Beginner | 50 | 2.0 | Default |
| Club | 500 | 0.5 | |
| Master | 100 000 | 0 | Effectively time-limited |

These parameters are passed to the worker at init time and forwarded to
`pickMoveWithTemperature()` (`worker.ts:83–128`) for final move selection.

**Temperature sampling formula (`worker.ts:96`):**
```
weights[i] = visitCounts[i] ^ (1 / temperature)
```
Higher temperature flattens the distribution over MCTS visit counts.  At T = 2.0 a
move with 30 visits gets weight `√30 ≈ 5.5`; a move with 5 visits gets `√5 ≈ 2.2`.
The ratio is only 2.5×, so the weaker move still gets selected roughly 29% of the
time — much more often than at T = 0 (argmax), but much less "uniformly random" than
one might hope.

**Also hard-coded in `worker.ts:55–56`:**
- `cpuct = 0.5` (PUCT exploration constant)
- `addNoise = 0.0` (Dirichlet noise at root — parameter exists but is always zero)

---

## 2 · Why "Beginner" is still too strong

### 2a · The policy head is already concentrated

The neural network is trained on thousands of master-level self-play games. Its
policy head (`policyLogits`, Float32Array[528]) assigns high probability mass to
1–3 tactically strong moves in any typical position. With only 50 MCTS trials the
search barely expands beyond the top policy candidates. Every trial still starts from
the network's top-rated moves, so the resulting visit-count distribution is nearly
identical to the raw policy — which already points toward strong play.

Temperature sampling at T = 2.0 does inject randomness *over that distribution*, but
the distribution itself is already good. The bot selects a "weaker" move occasionally,
but it rarely plays an outright blunder.

### 2b · Immediate-win detection is unconditional

`worker.ts:176–185` scans all legal moves for a single-step win before any MCTS
runs. This logic is not gated by difficulty level: the bot **always** takes a
forced win, regardless of preset. A beginner-level human opponent often misses a
one-move win; the bot never does.

### 2c · Swap model is unconditional

`worker.ts:164–170` uses the fitted linear swap model regardless of difficulty.  A
beginner opponent would make suboptimal swap decisions; the bot always makes the
statistically correct one.

### 2d · Opening book side-steps the weakest part of MCTS

Move 1 is returned from an opening book (`worker.ts:149–158`), bypassing MCTS
entirely. The book positions are carefully chosen to be "swap-eligible" — i.e.
tactically reasonable. A very weak bot would sometimes play an obviously bad first
move; ours does not.

### 2e · Tree reuse carries knowledge across turns

`NeuralMCTS` reuses the search tree up to 4 plies ahead (`mcts.ts:332`). Even with
a 50-trial budget, earlier searches accumulate. After the first few moves the root
already has a partially explored tree, so each subsequent 50-trial budget is spent
refining, not starting from scratch.

---

## 3 · Techniques that work on a single model

All of these can be layered on top of each other. Most can be implemented in a few
lines. They are listed roughly from easiest to most involved.

### 3a · Trial limit reduction *(already in use)*

**How it works:** `maxTrials` caps how many MCTS simulations run. Fewer trials →
thinner tree → more reliance on raw policy.

**Effect:** Strength scales roughly with `√(trials)`. Going from 50 → 10 → 3 trials
continues to reduce strength, but below ~5 trials the search barely forms a tree at
all — it becomes equivalent to sampling from the raw policy with a small correction.

**Limit:** Even 1 trial still uses the network's top-recommended move as the
starting point. See §2a.

**Already wired:** Yes. `BOT_STRENGTH_OPTIONS` in `main.ts:52–56`.

---

### 3b · Temperature sampling *(already in use)*

**How it works:** After MCTS the visit-count array is raised to `1/T` and then
sampled. Higher T → flatter → more random.

**Effect:** At T → ∞ the selection becomes uniform over all visited moves. But note
that only moves *visited by MCTS* appear in the distribution. With 50 trials and a
concentrated policy, many of the 528 legal moves have zero visits and therefore zero
weight even at T = ∞.

**Practical range:** T = 2.0 is already high. Moving to T = 5.0 or 10.0 further
flattens the visited set. Alternatively, apply temperature to the *prior policy
logits* (before MCTS), which would spread attention to more moves.

**Already wired:** Yes. `temperature` parameter in `BOT_STRENGTH_OPTIONS`.

---

### 3c · Dirichlet noise at the MCTS root

**How it works:** Before each search, mix Dirichlet noise into the root's prior
policy: `P'(a) = (1 − ε) · P(a) + ε · η(a)` where `η ~ Dir(α)`. AlphaZero uses
`ε = 0.25`, `α = 0.3` (Go). Smaller `α` → more concentrated noise (one action gets
most of the noise); larger `α` → flatter noise (evenly distributed).

For *weakening*, set `ε` high (e.g. 0.5–0.8) and `α` low (e.g. 0.1–0.3), forcing
MCTS to explore random moves heavily and inflating visit counts on weak moves.

**Effect:** Unlike temperature (which acts at move *selection* time), noise acts at
*search* time. It causes MCTS to *explore* weak branches, accumulating visits there,
so even an argmax final selection may pick a weaker move.

**Wiring needed:** `addNoise` is already a constructor parameter of `NeuralMCTS`
(`worker.ts:56`) but is hard-coded to `0.0`. The class would need to accept a
separate `noiseAlpha` (currently always uses `0.03` for go-scale games, see
`mcts.ts` internals) and a noise fraction `ε`. The preset table would then expose
these per difficulty level.

**Advantage over temperature:** Degrades the *search* rather than just the final
pick — makes the bot reason poorly, not just choose randomly from good options.

---

### 3d · Epsilon-softmax (sample from top-k, not top-1)

**How it works:** With probability `ε` replace the MCTS-selected move with a
uniform random draw from the top-k (k = 5–10) MCTS moves by visit count; with
probability `1 − ε` keep the best move. Unlike pure epsilon-greedy (random *legal*
move), this stays within "plausible" moves, avoiding obviously nonsensical plays.

**Effect:** Directly and controllably injects blunders that are at least locally
plausible. Easy to tune: `ε = 0.2` means one in five moves is a "reasonable but
not best" choice.

**Wiring needed:** New parameter (`epsilon`, `topK`) in `BotStrengthOption`;
modification of `pickMoveWithTemperature` in `worker.ts:83–128` to implement the
epsilon branch.

---

### 3e · cpuct reduction

**How it works:** The PUCT formula for child selection is
`U(s,a) = cpuct · P(s,a) · √N(s) / (1 + N(s,a))`.
Lower `cpuct` → less exploration → MCTS concentrates almost all visits on the
prior top move → the tree barely grows wider than one branch.

**Counterintuitively this can *decrease* strength** because the search fails to
discover refutations of the policy's first instinct. The bot commits to the
network's first suggestion without checking it tactically.

**Wiring needed:** `cpuct` is already a constructor argument to `NeuralMCTS`
(`worker.ts:55`). Exposing it per-preset requires only adding a `cpuct` field to
`BotStrengthOption` and passing it through.

**Caution:** Extreme cpuct reduction (e.g. 0.05) makes the bot play the same
"first-instinct" move every time with no tree search benefit at all — similar to
just returning `argmax(policyLogits)` directly. This is predictable and eventually
learnable by a human opponent.

---

### 3f · Policy head degradation (uniform blending)

**How it works:** Interpolate the network's raw policy logits with a uniform
distribution before MCTS: `P'(a) = λ · softmax(logits) + (1 − λ) · uniform`.
At `λ = 1` (current) MCTS searches from the network's preferred moves. At `λ = 0`
MCTS starts from uniform priors — equivalent to vanilla UCT with no policy guidance.

**Effect:** Degrades MCTS search quality without changing the number of trials or
the value head. A uniform prior forces MCTS to explore the board more evenly and
rely more on the value head to distinguish moves. The value head is still strong,
so effect is moderate — but combined with trial reduction it can significantly
weaken play.

**Wiring needed:** Modify `OnnxPlayer.eval()` in `onnx-player.ts` to accept a
`policyMixRatio` (0 to 1) and blend the returned logits before they reach MCTS.

---

### 3g · AlphaDDA dynamic difficulty adjustment

**How it works:** From the AlphaDDA paper (2022), the core idea is to reduce
`maxTrials` adaptively based on the *current evaluation*: when the bot is clearly
winning (value head output close to +1), it uses a smaller budget; when the game
is balanced or the bot is losing, it uses its full budget.

For a strength-reduction use case this is inverted: on "easy" mode, whenever the
value head says the position is won (e.g. `topQ > 0.7`), intentionally cap trials
very low and raise temperature — the bot plays lazily when it's ahead. When losing
(`topQ < −0.3`) it can play at full strength to keep the game competitive.

**Effect:** Creates a "rubber-band" difficulty that feels natural: the bot gives the
human a chance but doesn't let games become completely one-sided in either
direction. This is closer to how a human teacher plays.

**Wiring needed:** The value head output (`topQ`) is already available inside the
worker during the `mcts` call (returned by `player.eval()` on the root). A small
wrapper around the `mcts` call would check the root Q and adjust `maxTrials` /
`temperature` before the search starts.

---

### 3h · Disable immediate-win detection

**How it works:** Remove (or gate behind difficulty level) the unconditional win
scan in `worker.ts:176–185`. On easy difficulty the bot would sometimes miss a
forced win just as a human beginner would.

**Effect:** Dramatic quality reduction at the cost of occasionally looking
"unfair" (the human wins even though the bot had a winning move). This may feel
frustrating rather than educational. Suitable only for a "very easy" mode aimed
at children or absolute beginners.

**Wiring needed:** Gate `worker.ts:176–185` behind a `difficulty` flag in the
worker message payload.

---

### 3i · Random rollout fallback (partial NN bypass)

**How it works:** With probability `pRandom` (per MCTS leaf), skip the NN
inference entirely and return a uniformly random value estimate (e.g. 0) plus
uniform policy. This degrades evaluation quality for a fraction of leaves,
making the search noisier.

**Effect:** Reduces effective NN calls proportionally; combined with trial limits
this can make the search much weaker than either technique alone. However the
effect is stochastic and hard to calibrate precisely.

**Wiring needed:** Modify the evaluation callback passed to `NeuralMCTS` in
`worker.ts:53–54` to sometimes return a dummy result instead of calling
`player.eval()`.

---

## 4 · Do separate trained models help?

### The Maia Chess argument (yes, for behaviour matching)

[Maia Chess](https://maiachess.com/) (McIlroy-Young et al., 2020) trains nine
separate networks, each on games played by humans at a specific Elo band
(1100, 1200, … 1900). Each model *predicts what a human of that strength would
play*, not just "a weaker move". The result is stylistically human: the bot makes
the kinds of mistakes and oversights typical of that rating band, rather than
playing full-strength moves with random blunders sprinkled in.

**Advantage:** Feels like playing a human opponent of a known calibre. Great for
rated play and training environments.

**Disadvantage:** Requires training data from that strength band, significant
training compute, and ongoing maintenance (9× the models). For TwixT the human
game database is small and concentrated among strong club players — there is not
enough data from beginners to fit a "1100 Elo TwixT model".

### Single-model techniques are sufficient here

The goal for twixtbot-app is *fun and approachable play*, not *human-like mistakes
at a specific Elo*. The techniques in §3 can achieve a wide range of effective
strengths on a single model:

- At the weak extreme (§3h disabled win scan + §3c strong Dirichlet noise + §3b
  high temperature + §3a 3–10 trials) the bot makes outright blunders and misses
  obvious threats — approximately beginner-to-casual level.
- At the strong extreme (current Master preset: 100 K trials, T = 0) the bot plays
  near its trained ceiling.

**Conclusion: no separate training required.** The single ONNX model covers the
full range. The techniques to reach the bottom of that range require a few dozen
lines of code, not weeks of training.

---

## 5 · Recommended approach

### Proposed four-level preset table

| Preset | maxTrials | temperature | cpuct | addNoise ε | Notes |
|--------|-----------|-------------|-------|------------|-------|
| Novice | 5 | 5.0 | 0.2 | 0.5 | Disable win scan; noisy, makes blunders |
| Beginner | 50 | 2.0 | 0.5 | 0.0 | Current Beginner — still sharp |
| Club | 500 | 0.5 | 0.5 | 0.0 | Current Club |
| Master | 100 000 | 0 | 0.5 | 0.0 | Current Master |

**Novice** targets players who have never played TwixT. The combination of 5
trials, high temperature, low cpuct, and strong Dirichlet noise makes the bot play
the network's first instinct (barely any search), then randomly selects from a
flattened distribution over visited moves. Disabling the win scan makes the bot
occasionally miss obvious finishes, which keeps early games playable.

**Validation:** The recommended way to test strength changes is to run 50–100 bot
self-play games (Master vs Novice, Master vs Beginner) and measure win rates.
A reasonable target:
- Novice: 5–15% win rate vs Master
- Beginner (current): ~15–25% win rate vs Master  *(estimate — not yet measured)*
- Club: ~40–50% win rate vs Master

### Implementation priority

1. **Quickest win:** Add a `Novice` preset with `maxTrials=5, temperature=5.0`.
   Zero new code beyond the preset table change (`main.ts:52`). This alone may be
   sufficient for most casual players.

2. **Medium effort:** Wire `cpuct` and `addNoise` (already constructor params in
   `NeuralMCTS`) through to the preset table. Expose a `noiseAlpha` field.

3. **Longer term:** Implement AlphaDDA (§3g) as a "Adaptive" mode that adjusts
   strength automatically to keep games close. This is the most engaging experience
   for new players.

---

## 6 · Suggested additions to planned-features.md

New entries to append to section **1 · AI / MCTS Algorithm**:

| ID | Feature | Priority | Notes |
|----|---------|----------|-------|
| A7 | Add Novice preset (5 trials, T=5.0) | P1 | Zero new code; just extend BOT_STRENGTH_OPTIONS |
| A8 | Wire cpuct per difficulty level | P2 | cpuct already a NeuralMCTS constructor arg; expose in preset table |
| A9 | Wire Dirichlet noise ε per difficulty | P2 | addNoise already wired to 0.0; add noiseAlpha param and expose |
| A10 | Epsilon-softmax top-k selection | P3 | New branch in pickMoveWithTemperature; controllable blunder rate |
| A11 | AlphaDDA adaptive difficulty | P3 | Adjust maxTrials/temperature based on value head output at root |
| A12 | Gate immediate-win detection on difficulty | P3 | Allow Novice/Beginner to miss forced wins; feels more human |

---

*End of research notes.*

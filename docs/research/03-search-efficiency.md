# 03 — Search efficiency: fewer trials per move

**Status:** next — promoted from "scoped" on 2026-04-29 after topics 01
(batch-size lever) and 02 (architecture / capacity) closed. Per the
project goal in `README.md`, search-efficiency is the highest-leverage
remaining direction: MCTS dominates self-play wall-time, and self-play
dominates training compute. A working low-trials method directly
attacks the cost of the *only* path to strength that doesn't depend on
a teacher.

## Why this is the next topic (session context, 2026-04-29)

- **Topic 01 (replay/batch-size) closed.** `train_loop.py` now uses
  batch=4096 / lr=0.16 / warmup=25 (16× the original throughput,
  same total samples). KataGo curriculum sampling deferred but not
  blocking.
- **Topic 02 (model scaling) closed.** Architecture is sufficient at
  64f×8b — distillation diagnostic confirmed it can hold strength
  well past v8_F. Capacity is not the bottleneck.
- **Net remaining gap to `six-917000`:** v8_F loses 200/0 to
  six-917000. Root cause now best-attributed to **insufficient
  self-play volume + iteration count**, since neither hyperparameters
  nor architecture explain the gap.
- **Implication:** the dominant cost of closing the gap is iteration
  wall-time, which is dominated by self-play, which is dominated by
  MCTS trials. Any reduction in trials-per-move multiplies through
  the entire training program. This is exactly what topic 03 is
  about.

## Question

## Why it matters for us specifically

- Iter 8 wall-time projection: ~13 hours self-play. Halving trials at
  same quality would ~halve that — biggest single lever for iteration
  velocity.
- Stronger models will only push trials cost higher; what works at
  iter 8 may break at iter 12.
- This is purely about *self-play training data generation*, not
  user-visible play strength. Arena / production play can keep using
  whatever trials count is best.

## What we already know / pointers

- `src/asn_player.py` — implements the asynchronous MCTS we use.
- `TRIALS_CADENCE` in `train_loop.py`: iter 1-2 → 50 trials, iter 3-4
  → 100, iter 5+ → 200. Already a coarse "ramp" but it's keyed on
  iteration index, not on observed quality of the policy targets.
- Related but distinct: `A4 — PUCT constant tuning` in
  `improvements.md`. cpuct controls *how* search explores; this topic
  is about *how much* search.

## Open questions

1. **What is "QZero" exactly?** The user mentioned this term but I
   don't have a confident match. Candidates:
   - **Gumbel AlphaZero** (Danihelka et al., DeepMind, 2022) — uses
     Gumbel-Top-k sampling at the root, deterministic best-action at
     non-root nodes; reportedly matches AlphaZero strength at <50
     simulations vs hundreds. *Most likely match.*
   - **AlphaZero variants without MCTS** (e.g., direct policy / value
     bootstrapping) — exists in research but no canonical name I'm
     sure of.
   - Possibly a project / blog post / preprint I'm not aware of.
   - **First task on this topic: nail down the reference.**
2. **What exactly does Gumbel AlphaZero change?** Worth a careful
   read of the paper. Key claims to verify: matches strength at very
   small sim counts, simpler implementation than vanilla MCTS,
   deterministic non-root nodes.
3. **Compatibility with our pipeline:** Gumbel AZ needs the policy
   target generated differently — instead of visit-count distribution,
   it's a function of the sampled actions. Our `LearningState.N`
   format would need to change or be reinterpreted. How disruptive?
4. **Lower-bound experiment:** what happens if we drop trials to 50
   for one full iteration, holding everything else constant? Is the
   resulting training data noticeably worse (arena win-rate vs the
   trials=200 model)?
5. **Hybrid search ramp** — keep trials=200 for "important" positions
   (e.g., near terminal, low value-head confidence) and drop to
   trials=50 elsewhere. Is the bookkeeping worth the savings?

## References to gather

- Danihelka, Guez, Schrittwieser, Silver, "Policy improvement by
  planning with Gumbel" (ICLR 2022). The Gumbel AlphaZero paper.
- Verify what "QZero" refers to (search the user's source — paper /
  blog / reddit thread / talk?). User context: heard recently.
- KataGo's variants of MCTS — `playoutDoublingAdvantage`, root noise
  damping, etc.

## Where we left off

Promoted to "next" topic. No experimentation has been done yet — only
scoping. The two open threads from the original scope are still the
right starting points:

1. **"QZero" reference is unresolved.** User mentioned the term but
   the canonical paper / project is unconfirmed. Most likely candidate
   is **Gumbel AlphaZero** (Danihelka et al., ICLR 2022) but worth
   asking the user where they heard it before committing to a paper.
2. **No baseline yet for "what happens at low trials".** We don't
   actually know whether dropping trials hurts our setup — could be
   it doesn't even at trials=50 with the current model.

## Next action — when picking this up in a fresh session

Suggested first step (cheap, informative, runs unattended):

**Run one self-play iteration at `trials=50` from the current
strongest checkpoint** (`models/v8_F.pt` per `train_loop.py` defaults,
or `v0_distill_10k.pt` if the user explicitly opts in to the
benchmark-anchor seed — but per the project goal we should *not* use
distilled checkpoints as the bootstrap line). Compare the resulting
model to a sibling iteration at the standard `trials=200`:

- Phase A wall-time at trials=50 should drop ~3-4× (200/50 with some
  overhead).
- Arena the resulting two child models at the standard arena settings
  (`--num_clones 18 --trials 400 --async_calls 32 --total_games 200`).
- Win rate ≥ 50% for the trials=50 child → training quality is fine
  at low trials; massive velocity win and we should make this the
  default.
- Win rate < ~45% → low-trials hurts; need a *smarter* approach
  (Gumbel root sampling, etc.) rather than just a lower count.

Before/in parallel: confirm with the user what "QZero" refers to so
the smarter-approach path goes after the right paper.

**Mechanics:** `train_loop.py` reads `TRIALS_CADENCE`. To run one
trials=50 iteration without touching the cadence permanently, pass
`--trials 50` on the command line if supported, or fork a copy of
`train_loop.py` for the experiment, or just override the cadence
locally and revert after the iteration.

**Estimated cost:** one iteration at trials=50 ≈ 3 hours (vs ~13 at
trials=200). Plus ~40 min arena. Wall-clock under half a day.

# 03 — Search efficiency: fewer trials per move

**Status:** scoped

## Question

Self-play at trials=200 dominates iteration wall-time. Position-cache
hit rates have collapsed as the model sharpens (iter 6 ~55% → iter 8
~0%), so each move now consumes nearly the full 200 NN evaluations.
Can we get equivalent training-data quality with substantially fewer
simulations per move?

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

(Just scoped. Need to clarify "QZero" reference before going deeper.)

## Next action

Two quick parallel things:

1. **Clarify "QZero" with the user** — ask them where they heard the
   term so we go after the right paper / project.
2. **Independent experiment: trials=50 single iter.** With TR1's
   resumability we can run *one* iteration at trials=50 cheaply (~3
   hours instead of ~13), produce a model, and arena it against the
   parent at proper trials=400. Outcome:
   - Win rate ≥ 50% → training quality is fine at trials=50; huge win
     for self-play velocity.
   - Win rate < 45% → training data quality really does suffer; need
     smarter low-trials approach (Gumbel).
   This baseline is useful regardless of which paper "QZero" turns out
   to be.

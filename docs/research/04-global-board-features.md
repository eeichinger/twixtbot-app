# 04 — Global board features (end-to-end connection awareness)

**Status:** scoped (most prior thinking lives in B9a discussions in
existing docs; this file consolidates and adds open questions.)

## Question

TwixT's win condition is a *global* property: a peg chain that crosses
from one home edge to the other. Local convolutions (5×5 kernels in
the current 20-block ResNet) can detect local link patterns but
struggle to aggregate the **board-wide topology** that decides whether
a position is connected end-to-end. Adding a KataGo-style global
pooling bias should let every spatial position at every depth see
board-global statistics and improve evaluation of long-range threats.

## Why it matters for us specifically

- TwixT is structurally one of the games where global pooling should
  pay off **most** — connection topology is the entire signal. Compare
  to Chess where global features matter but local tactics dominate.
- Existing tracking: **B9a in `docs/improvements.md`** ("KataGo global
  pooling bias", P2/High/Pending) and `docs/further_training_improvements.md`
  §5b. Both note this as the **highest expected strength gain** of any
  model improvement on the table.
- It compounds well with **02-model-scaling.md** — bigger nets and
  global pooling are mentioned as a natural pair (both increase
  capacity to use long-range information).

## What we already know / pointers

From `improvements.md` §B9a (~lines 139-153):
- Add `GlobalAvgPool → Linear → broadcast-add` at each residual block.
- Captures long-range link density.
- Full retrain needed.

From `further_training_improvements.md` §5b (~lines 114-135):
- The "G" in KataGo's residual blocks — bias path that is global
  rather than local.
- Pairs with item #5 (deeper net) — they compound.
- Listed at line 305 alongside item #5 as a high-payoff retrain.

## Open questions

1. **What exactly is the KataGo block structure?** Need to read the
   paper carefully. Key questions:
   - Does the global path use mean, max, or sum pooling?
   - Is it added at every residual block or only some?
   - Is there a separate "valuation tower" block at the end vs
     pooling-bias-everywhere?
2. **PyTorch implementation reference:** does `katago` repo have a
   readable Python equivalent we can mirror? Or do we re-derive from
   the paper?
3. **Computational cost** — how much does pooling-bias add to forward-
   pass time? GlobalAvgPool is cheap (single sum + division per
   channel) but the per-block Linear + broadcast may not be.
4. **Backwards compatibility** — converting an existing checkpoint to
   the new architecture: we'd discard the old residual block weights
   and start fresh, OR we initialize the global-bias path to zero
   (broadcast-add of zero = identity), preserving the old behavior at
   init time. Latter is much friendlier to progressive scaling.
5. **Measurement strategy** — once trained, how do we *prove* the
   gain is from global features specifically rather than added
   capacity? Ablation: same param count without the global path.
6. **Order of operations vs other model changes** — should B7
   (deeper policy head) ship first, or B9a, or together? KataGo's
   recipe applies global bias *throughout* the residual tower, which
   means it interacts with every other architecture choice.

## References to gather

- Wu, "Accelerating Self-Play Learning in Go" (KataGo, arXiv:1902.10565,
  2019). Read §3 ("Global pooling"). Likely the canonical reference.
- KataGo repo: https://github.com/lightvector/KataGo — their
  `model.py` for an implementation reference.
- Anthony et al., "Thinking Fast and Slow with Deep Learning and
  Tree Search" (Hex, 2017) — Hex is topologically similar to TwixT;
  may have prior art on global features.

## Where we left off

(Just scoped. Existing B9a discussion gathered into this file but no
implementation work or measurement done.)

## Next action

Read KataGo §3 carefully (or its TF2 reference implementation) and
write a concrete spec — block structure, where pooling bias is added,
zero-init strategy for backward compat, parameter count delta. Output:
a "Implementation sketch" section in this file. Once that's solid the
topic is ready to graduate to a proper feature spec under
`docs/specs/b9a-...md`.

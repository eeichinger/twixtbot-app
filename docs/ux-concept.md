# UX Concept: Analysis, Visualization & Game Controls

**Date:** 2026-04-09
**Scope:** Features V1–V6, G1–G3, U1–U5 from `docs/planned-features.md`

---

## Context & Goals

The planned analysis features are all genuinely useful — but added naively they would
turn the game screen into an instrument panel. The goal of this document is to specify
*where* each feature lives so that a player who just wants to play sees a clean board,
while a player who wants analysis has everything one tap away.

---

## Design Principles

1. **Board first.** The canvas must stay as large as possible. Nothing permanently
   reduces board real-estate.

2. **Progressive disclosure.** Two layers: always-visible basics (status, win-prob
   strip, core buttons), and a single collapsible Analysis strip for everything deeper.

3. **Group by when you need it.** Three moments drive the design:
   - *During AI thinking* → enhanced overlay (V1, V6)
   - *After each AI move* → persistent win-prob indicator (V3), expand for more (V2, V4)
   - *On demand* → heatmap overlay (V5), move list (U1), settings

---

## Streamlining Decisions

### Bot vs Bot — already solved, no new mode needed

G3 "Bot vs Bot" is fully achievable today: press **Suggest** for every move on both
sides. There is no need for a dedicated mode. Instead the concept leans into this by
making Suggest always-prominent and documenting the pattern.

A potential "Auto" toggle (auto-play both sides without manual Suggest presses) is a
low-code extension that can be added to Settings later if there's demand.

### Resign vs New Game

A Resign action doesn't need a separate button if "New Game" already abandons the
current game. The distinction is semantic (I give up vs I want a different game). For
now, Resign is added as a lightweight button that simply marks the result and shows the
game-over screen — equivalent to forfeiting. If the game is already over, Resign
disappears. This keeps the action small (a single condition check) without needing a
confirmation dialog for the first iteration.

### Settings dropdowns leave the control bar

The Strength and Think-time selectors are moved into a ⚙ Settings panel. This frees
two wide controls from the already-tight bottom bar and replaces them with a single
icon. Players almost never change these mid-game; tucking them behind a gear icon
costs nothing in usability.

---

## Screen Layout Overview

### Portrait phone (≤ 430px wide) — the primary target

```
┌────────────────────────────────────┐
│ ←   Your turn (Blue)          ⚙   │  56px — status bar
│     ════════════════════           │  3px win-prob bar (new, V3)
├────────────────────────────────────┤
│                                    │
│         B O A R D                  │  flex:1 — board canvas
│          (430×430)                 │  Swap button floats over bottom
│                                    │
│  ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ │  ← unused vertical space ~190px
│  ▸ Analysis         [expanded: ▾] │  32px handle (new)
│  ┌──────────────────────────────┐  │
│  │ Win prob  ████████░░  72%    │  │  \ analysis panel
│  │ h5 ██████████ 45%  (best)   │  │  | (collapsible,
│  │ r9 ████       19%           │  │  | ~180px when open)
│  │ m7 ██         13%           │  │  |
│  │ ▁▂▂▃▃▄▄▅▄▄▃ (eval history) │  │  |
│  │ [🗺 Heatmap] [📋 Moves]     │  │  /
│  └──────────────────────────────┘  │
├────────────────────────────────────┤
│ ⚙  ↩  ↪  💡  🏳  📤             │  ~50px — control bar (6 icons)
└────────────────────────────────────┘
```

> **Key insight:** On a portrait phone the board is constrained by screen *width*
> (430px), not height. The board-container's height is ~800px, leaving ~190px below
> the 430px canvas. The analysis panel lives entirely within this pre-existing gap —
> **no board shrinkage required**.

### Landscape tablet / desktop

Board fills available height; no unused vertical space below it. In landscape:
- The analysis panel collapses to a sidebar-style panel on the right (or remains
  accessible via the ▸ Analysis handle in the control bar area).
- The win-prob bar remains in the status bar regardless of orientation.

---

## Status Bar: Win-Probability Bar (V3)

A 3px horizontal bar directly under the status text line, spanning the full width.

- **Color:** interpolated from red (opponent winning at −1) through neutral grey (0)
  to the current-player's color (winning at +1). Uses the existing blue/orange palette.
- **Appears:** after the first AI move that returns a `topQ` value. Hidden before then
  (zero height, no layout shift).
- **Updates:** on every AI result message.
- **In PvP:** hidden entirely (no AI `topQ` available).

This conveys the key insight — who's ahead — at a glance, without any tap or expand.

---

## Thinking Overlay: Enhanced (V1 + V6)

The existing full-screen overlay (spinner + "AI is thinking…") is extended in-place.

### Trial counter / elapsed (V1)

Replace the static "AI is thinking…" with a live counter fed by `ping` messages:

```
  ◌  Searching...   3.2s  ·  1,847 trials
```

- `elapsed` from the ping message → formatted as `3.2s`
- `trials` is not currently in ping — extend the worker's ping message to include
  `root.N.reduce((a,b) => a+b, 0)` (sum of root visit counts). This is cheap to
  compute and gives a true trial count.
- Falls back to elapsed-only if trials unavailable.

### Best-line visualization (V6)

While the overlay is up, the principal variation (best line found so far) is drawn on
the board *underneath* the overlay glass:

- Walk `argmax N` recursively from root: root → child → grandchild … up to 8 plies.
- Draw semi-transparent peg dots (30% opacity) at each PV position, alternating
  blue/orange for each side.
- Draw thin semi-transparent links between consecutive same-color PV pegs if legal.
- The worker's ping message is extended to include `pv: Point[]` (the current PV
  array, at most 8 entries). The MCTS module already has the root node accessible
  after each trial batch; extracting the PV is a simple argmax walk.
- When the overlay dismisses, the PV overlay is cleared and the actual played move is
  rendered normally.

**Effort note:** V6 requires extending both the worker ping protocol and the board
renderer. It is the highest-effort item in this concept and should be treated as a
Phase 2 addition after V1–V3 are live.

---

## Analysis Strip (V2 + V3 detail + V4 + V5 + U1)

A collapsible panel living in the vertical space below the board canvas.

### Handle (always visible when any AI data is available)

```
▸ Analysis   last: h5 · 72% · 1,847 trials
```

- 32px tall, tappable to expand/collapse.
- The handle shows a one-line summary of the last AI move so the strip carries value
  even when collapsed.
- Hidden entirely before the first AI move (no data yet).
- Collapsed state is remembered in localStorage (`twixt-analysis-open`).

### Expanded content

**Section 1 — Last move stats (V2 + V3)**

```
Win probability   ████████████░░░░  72%
───────────────────────────────────
h5   ████████████████████  45%   ← best
r9   ████████              19%
m7   █████                 13%
───────────────────────────────────
1,847 trials  ·  4.8s  ·  Master
```

- Win probability bar: the root `topQ` value rendered as a full-width bar, labelled.
- Top-3 move bars: derived from the move visit counts returned in the result. The move
  label (e.g. "h5") is computed via `policyIndexToPoint` and formatted with
  `ptToString`.
- Footer line: `trials`, `elapsed`, bot strength label — contextualises the quality of
  the search.

**Section 2 — Evaluation history sparkline (V4)**

```
▁▁▂▂▃▄▄▅▄▄▃▃▄▄▅▅▆▅    ← 3-tone bar: blue=Black ahead, grey=even, orange=White ahead
```

- A compact bar-sparkline (24px tall) spanning the game history.
- Each bar = one AI move. Filled blue if `topQ > 0.1`, orange if `topQ < −0.1`,
  grey otherwise.
- Accumulated in `main.ts` in an array `evalHistory: number[]` — one entry appended
  after each AI result.
- Tapping a bar in the sparkline: no action initially (future: jump to that move in
  the game). The bars are purely decorative in Phase 1.

**Section 3 — Toggle buttons**

```
[🗺 Heatmap]   [📋 Moves]
```

- **Heatmap (V5):** Fires a one-shot NN inference on the current position (no MCTS),
  gets raw policy logits, normalises them, and draws a semi-transparent colored circle
  at each legal move position using the blue→cyan→green gradient from twixtbot-ui.
  Pressing again clears the overlay. Stays on until next move is played (auto-cleared
  on `game.play()`).

- **Moves (U1):** Toggles a move-list view replacing (or appearing below) the stats
  section. Shows the full game history as paired move chips: `1. h5 · r9 · 2. d6 · …`
  — compact horizontal flow, wrapping. The current move is bolded if in replay.

---

## Control Bar Redesign

### Before (current)

```
[ Strength ▾ ]  [ Think time ▾ ]  [ Suggest ]  [ Undo ]  [ Export ]
```

Five elements, two of which are wide dropdowns. Already tight on 430px.

### After (proposed)

```
[ ⚙ ]  [ ↩ Undo ]  [ ↪ Redo ]  [ 💡 Suggest ]  [ 🏳 Resign ]  [ 📤 Export ]
```

Six icon-buttons, uniform size (~50px each on 430px = 300px total, comfortable).

| Button | Icon | Visibility rule |
|--------|------|-----------------|
| Settings | ⚙ | Always |
| Undo | ↩ | When `game.history.length > 0` |
| Redo | ↪ | When `redoStack.length > 0` (new G1) |
| Suggest | 💡 | When human turn + game not over + not thinking |
| Resign | 🏳 | When game not over + game started (≥ 1 move played) |
| Export | 📤 | Always (exports current state even if in progress) |

**Redo (G1):** A `redoStack: MoveRecord[]` is cleared on any human move (same as most
board games). In PvC mode, undoing 2 moves (human + AI) re-pushes both onto the redo
stack; Redo re-applies both. In PvP, one move at a time.

**Resign (G2):** Sets `tsgfResult = turn === BLACK ? 'W+' : 'B+'`, calls `gameOver =
true`, and re-renders the status bar. No confirmation dialog in the first iteration
(low-stakes local game — easy to just start a new one).

---

## Settings Panel

Slide-up bottom sheet triggered by the ⚙ button.

```
┌──────────────────────────────────┐
│  Settings                    ✕  │
├──────────────────────────────────┤
│  AI Strength     [ Beginner ▾ ] │
│  Think Time      [   5s     ▾ ] │
├──────────────────────────────────┤
│  Show board labels       [ ON ] │
│  Show guide lines        [OFF ] │
│  Allow crossing own links[OFF ] │  ← G4 (SCL)
└──────────────────────────────────┘
```

- Strength and Think Time: same options as the current dropdowns, persisted to
  localStorage with the same keys. Changing them takes effect on the next AI move.
- Label/guideline/SCL toggles: new, persisted to localStorage. The board re-renders
  immediately (just calls `boardUI.render()`).
- SCL toggle is only meaningful before a game starts; if a game is in progress,
  changing it shows a note: "Takes effect on next game."

---

## What This Concept Defers

| Feature | Why deferred |
|---------|-------------|
| V6 best-line (full implementation) | Requires worker protocol change + board renderer extension — Phase 2 |
| V4 sparkline interactivity (tap to jump) | Needs replay integration — after move-list work |
| U2 coordinate tooltip | Minor quality-of-life; already partially covered by drag preview |
| U5 Player names | Low demand for a single-player vs AI app |
| G4 SCL rule (full UX) | In Settings panel but mid-game change warning is a nice-to-have |
| Eval history in replay viewer | Separate scope — part of LG analysis feature set |

---

## Feature → Location Summary

| ID | Feature | Location |
|----|---------|----------|
| V1 | MCTS progress indicator | Thinking overlay (trial counter) |
| V2 | Top-3 candidate moves | Analysis strip → Last move section |
| V3 | Win probability | Status bar bar + Analysis strip header |
| V4 | Eval history sparkline | Analysis strip → sparkline section |
| V5 | Policy heatmap | Analysis strip → Heatmap toggle → board overlay |
| V6 | Best-line visualization | Board under thinking overlay (Phase 2) |
| G1 | Redo | Control bar |
| G2 | Resign | Control bar |
| G3 | Bot vs Bot | Not a new feature — Suggest handles it |
| G4 | SCL rule | Settings panel |
| U1 | Move list | Analysis strip → Moves toggle |
| U2 | Coordinate tooltip | Deferred (drag preview already covers this) |
| U3 | Show/hide labels | Settings panel |
| U4 | Show/hide guidelines | Settings panel |
| U5 | Player names | Deferred |

---

## Phased Implementation

### Phase 1 — Quick wins (no new data from worker)
1. Control bar: move Strength + Think-time to Settings panel; add Redo + Resign buttons
2. Status bar: win-prob underline bar (use `topQ` from existing result message)
3. Analysis strip handle + Last move stats (V2 + V3) — wire up existing result data
4. Eval history sparkline (V4) — accumulate `topQ` per AI move in `main.ts`

### Phase 2 — Overlays and deeper analysis
5. Thinking overlay trial counter (V1) — extend worker ping to include trial count
6. Heatmap toggle (V5) — on-demand NN forward pass + canvas overlay
7. Move list in analysis strip (U1)
8. Settings panel toggles: show labels, guidelines, SCL

### Phase 3 — Best-line visualization
9. Best-line on board during thinking (V6) — extend worker ping with PV array + board renderer

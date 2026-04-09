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

## Mode Applicability Matrix

The three screens share the same board canvas component but have different analysis
contexts. This matrix is the anchor for every design decision below.

| Feature | PvC | PvP | Replay |
|---------|-----|-----|--------|
| Win-prob bar (V3) | Auto — after each AI move | On-demand — after Suggest | On-demand — after Analyse |
| Top-3 candidates (V2) | Auto — after each AI move | On-demand — after Suggest | On-demand — after Analyse |
| Eval history sparkline (V4) | Auto — grows as game progresses | On-demand — after each Suggest | On-demand — "Analyse full game" |
| Heatmap overlay (V5) | Toggle button (any time) | Toggle button (any time) | Toggle button (any time) |
| MCTS progress (V1) | Auto — during AI thinking | Auto — during Suggest | N/A |
| Best-line on board (V6) | Auto — during AI thinking | Auto — during Suggest | N/A |
| Move list (U1) | Toggle in analysis strip | Toggle in analysis strip | Always expanded (primary nav) |
| Move quality overlay (L2) | N/A | N/A | After "Analyse full game" |
| Redo (G1) | Yes | Yes | N/A — use step controls instead |
| Resign (G2) | Yes | Yes | N/A |
| Thinking overlay (V1/V6) | Yes — AI thinks automatically | Yes — only when Suggest pressed | N/A |
| Analysis strip | Auto-appears after AI move | Appears after Suggest | Always visible (primary UI) |

**Key observations:**
- In PvP the analysis strip and win-prob bar are *on-demand*, triggered by Suggest.
  Before any Suggest press the game screen looks identical to today.
- In Replay, analysis is the *primary purpose* of the screen. The analysis strip is
  always open, the move list is the main navigation tool, and AI is invoked explicitly.
- The heatmap (V5) is the one feature that works identically in all three modes — it's
  always a one-tap on-demand overlay on the current board position.

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
- **In PvP:** hidden until the first Suggest is used; then behaves the same as PvC
  (shows eval for that one position, stays until the next human move clears it).
- **In Replay:** not in the status bar — win probability appears inside the always-open
  analysis strip instead (see Replay section).

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
▸ Analysis   last: h5 · 72% · 1,847 trials    ← PvC / after Suggest in PvP
▸ Analysis   —                                 ← PvP before first Suggest (no data)
```

- 32px tall, tappable to expand/collapse.
- The handle shows a one-line summary of the last AI result so the strip carries value
  even when collapsed.
- In **PvC**: hidden before the first AI move, then always visible.
- In **PvP**: visible from game start (move list is always useful), but the one-line
  summary shows "—" until Suggest has been used at least once.
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

## PvP Mode: Analysis on Demand

In PvP no AI runs automatically, so none of V1–V4 appear unprompted. The analysis
strip is present from the start but shows only the move list until Suggest is used.

**Suggest in PvP** triggers a full AI evaluation of the current position for one side:
- The thinking overlay appears (and if Phase 2 is complete, shows V1/V6).
- On result: win-prob bar appears in the status bar, analysis strip populates with
  the same V2+V3 content as PvC.
- The eval sparkline (V4) adds one point per Suggest press (not per game move).
  This creates a partial "advice history" rather than a continuous eval chart, which
  is fine — it reflects which moments the players asked for AI input.

**Move list (U1)** is the most useful PvP analysis feature and is available from the
first move, no Suggest needed.

**Heatmap (V5)** works at any time in PvP, as it only requires the board position —
not an ongoing AI game. In PvP it becomes a study tool: "what does the AI think of
this position right now?" On-demand NN inference, same toggle button as PvC.

No other changes to the PvP control bar or flow. The six-button control bar
(`⚙ Undo Redo Suggest Resign Export`) is identical in PvP and PvC.

---

## Replay Screen: Analysis as the Primary Goal

The replay viewer is a different screen (`#replay-screen`) whose *purpose* is analysis.
The design principles invert slightly: the analysis strip is **always open** (it is
the primary content below the board), and AI is invoked explicitly rather than
automatically.

### Replay screen layout (portrait phone)

```
┌────────────────────────────────────┐
│ ←   Player1 (B) vs Player2 (W)    │  56px — header with back + title
├────────────────────────────────────┤
│                                    │
│         B O A R D                  │  flex:1 — board canvas (read-only)
│          (430×430)                 │
│                                    │
│  ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ │  unused vertical space
│                                    │
│  ┌──────────────────────────────┐  │  \
│  │ 1. h5  r9  2. d6  m12  3. … │  │  | Move list (U1/L4)
│  │ ▶ 4. f8  ←── current move   │  │  | always visible, scrollable
│  └──────────────────────────────┘  │  /
│                                    │
│  ┌──────────────────────────────┐  │  \
│  │ [Analyse position] [🗺 Heat] │  │  | Analysis controls
│  │ Win prob  ████████░░  64%    │  │  | (empty until Analyse pressed)
│  │ h8 ██████████ 41%  (best)   │  │  |
│  │ k5 ████       17%           │  │  |
│  │ e7 ███        12%           │  │  |
│  └──────────────────────────────┘  │  /
├────────────────────────────────────┤
│ |◀   ◀   Move 4 / 47   ▶   ▶|    │  ~50px — step controls (unchanged)
└────────────────────────────────────┘
```

### Move list (U1 / L4) — primary navigation

The move list is the first thing shown below the board in replay, always expanded:

- Compact horizontal flow: `1. h5 · r9   2. d6 · m12   3. f8 · …`
- Current move is highlighted (bold or accent color).
- Tapping any move jumps directly to that position (replaces step controls for
  non-sequential navigation).
- The step controls (`|◀ ◀ Move N/Total ▶ ▶|`) are retained as the primary
  sequential navigation — move list provides quick jumping.

### Per-position analysis (L1 equivalent)

An **"Analyse position"** button below the move list triggers AI evaluation of the
current board state:

- Loads the AI worker (or reuses it if already alive).
- Runs a short MCTS pass (thinking overlay appears; same V1/V6 enhancements apply).
- On result: populates the analysis section with win-prob bar + top-3 moves (V2+V3).
- The result is cached for this move index so stepping away and back doesn't require
  re-analysis.
- The Heatmap toggle (V5) is always available; it uses the same on-demand NN inference
  as in PvC/PvP and does not require the Analyse button first.

### Full-game analysis ("Analyse full game")

A secondary button (or a confirmation prompt after "Analyse position" shows once) that
runs AI evaluation on **every position** in the game sequentially:

- Shows a progress bar: "Analysing move 7 / 47…"
- On completion: populates the eval history sparkline (V4/L3) for all moves.
- Enables the **move quality overlay (L2)**: each peg is colored on the board by
  how well the AI rated the move that placed it — green (best or near-best),
  yellow (reasonable), red (significantly worse than top choice).

### Eval history sparkline in replay (V4 / L3) — interactive

In PvC the sparkline is decorative. In replay it is interactive:

- Each bar in the sparkline corresponds to one move in the game.
- The current move's bar is highlighted.
- **Tapping any bar jumps to that move** — same effect as tapping the move list.
- The sparkline thus doubles as a visual timeline and a navigation control.

### Replay screen analysis states

| State | What's shown |
|-------|-------------|
| Just opened | Move list + "Analyse position" button + "🗺 Heatmap" button |
| After "Analyse position" | + Win prob bar, top-3 moves for current position |
| After stepping away | Cached result stays; new position shows stale badge until re-analysed |
| After "Analyse full game" | + Interactive sparkline, move quality overlay on board |
| Heatmap active | Policy color circles overlaid on board cells (independent of above) |

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
| L2 move quality overlay | Depends on full-game analysis pass being complete — Phase 3 |
| V4 sparkline interactivity in PvC | In PvC, bars are decorative in Phase 1; tapping to rewind is a Phase 2 addition |
| "Analyse full game" in replay | Slow (N × inference); deferred until per-position analysis (L1) is solid |
| U2 coordinate tooltip | Minor quality-of-life; drag preview already partially covers this |
| U5 Player names | Low demand for a single-player vs AI app |
| G4 SCL rule (full UX) | In Settings panel but mid-game change warning is a nice-to-have |

---

## Feature → Location Summary

| ID | Feature | PvC | PvP | Replay |
|----|---------|-----|-----|--------|
| V1 | MCTS progress | Thinking overlay | Thinking overlay (Suggest) | N/A |
| V2 | Top-3 candidate moves | Analysis strip — auto | Analysis strip — after Suggest | Analysis strip — after Analyse |
| V3 | Win probability | Status bar + strip — auto | Status bar + strip — after Suggest | Analysis strip — after Analyse |
| V4 | Eval history sparkline | Analysis strip — auto, static | Analysis strip — partial (one point per Suggest) | Analysis strip — interactive, after full-game analysis |
| V5 | Policy heatmap | Heatmap toggle → board | Heatmap toggle → board | Heatmap toggle → board |
| V6 | Best-line viz | Board under overlay (Phase 2) | Board under overlay on Suggest (Phase 2) | N/A |
| G1 | Redo | Control bar | Control bar | N/A (step controls) |
| G2 | Resign | Control bar | Control bar | N/A |
| G3 | Bot vs Bot | Not new — Suggest handles it | Not new — Suggest handles it | N/A |
| G4 | SCL rule | Settings panel | Settings panel | N/A |
| U1 | Move list | Analysis strip → toggle | Analysis strip → toggle | Always open, primary nav |
| L1 | Analyse position | N/A | N/A (Suggest is equivalent) | "Analyse position" button |
| L2 | Move quality overlay | N/A | N/A | Board overlay after full-game analysis |
| L3 | Eval history in replay | N/A | N/A | Interactive sparkline (same as V4) |
| L4 | Move list in replay | N/A | N/A | Same as U1 — always open |
| U2 | Coordinate tooltip | Deferred | Deferred | Deferred |
| U3 | Show/hide labels | Settings panel | Settings panel | Settings panel |
| U4 | Show/hide guidelines | Settings panel | Settings panel | Settings panel |
| U5 | Player names | Deferred | Deferred | N/A |

---

## Phased Implementation

### Phase 1 — Quick wins, all modes (no new worker data)
1. **PvC/PvP control bar:** move Strength + Think-time to Settings panel; add Redo + Resign
2. **PvC status bar:** win-prob underline bar (wire `topQ` from existing result message)
3. **PvC analysis strip:** handle + last move stats (V2+V3) + eval sparkline (V4)
4. **Replay move list (L4/U1):** always-open move list with jump-to-move, replaces current plain counter
5. **Replay "Analyse position" (L1):** button that runs AI → populates analysis section (V2+V3)
6. **All modes heatmap (V5):** toggle button → single NN pass → board overlay

### Phase 2 — Deeper analysis
7. **Thinking overlay trial counter (V1):** extend worker ping to include root visit sum
8. **PvP analysis strip:** show move list handle from game start; populate after Suggest
9. **Replay interactive sparkline (V4/L3):** tap bar = jump to move (requires Phase 1 replay work)
10. **Settings panel display toggles:** show labels, show guidelines, SCL rule (U3/U4/G4)

### Phase 3 — High-effort features
11. **Best-line on board (V6):** extend worker ping with PV array; board renderer overlay
12. **Replay full-game analysis:** batch inference pass + progress bar
13. **Replay move quality overlay (L2):** peg coloring after full-game analysis

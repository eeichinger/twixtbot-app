# Planned Features

Consolidated from `docs/improvements.md`, `docs/lg-planned-features.md`,
`docs/network-pvp-research.md`, and a feature comparison against
`github.com/eeichinger/twixtbot-ui` (the desktop Python app).

Last updated: 2026-04-09.

---

## How to read this document

| Priority | Meaning |
|---|---|
| **P1** | High value, low risk — do next |
| **P2** | High value, moderate effort or dependency |
| **P3** | Nice to have, lower urgency |
| **P4** | Future / research — don't start yet |

Status: **Done** · **Pending** · **Future**

### Stable references — feature IDs only

**Feature IDs are the only stable reference handles** (e.g. `V3`, `L1`, `A4`).
Section headings and their numbering are organisational only; never cite a section
letter or number as a reference. Rules:

- Once assigned, a feature ID is **never changed or reused**, even if the feature is
  dropped (mark it as removed in the Notes column instead).
- New features get the **next unused number** within their prefix group (e.g. the next
  LG feature after L11 is L12). Do not insert items between existing numbers.
- Sub-sections within a top-level section use **descriptive names only** — no letter
  or number suffix. Inserting a new sub-section never requires renumbering anything.

---

## 1 · AI / MCTS Algorithm

These require no new model training unless noted.

| ID | Feature | Priority | Effort | Status | Notes |
|----|---------|----------|--------|--------|-------|
| A1 | Swap rule (`swapmodel.ts`) | P1 | Low | **Done** | 4-coeff linear model, correct swap decision |
| A2 | FPU / smart-init Q pre-seeding | P1 | Low | **Done** | Unvisited nodes inherit parent value |
| A3 | Tree reuse: extend to 4 plies + fix O(N²) replay | P2 | Medium | Pending | Currently drops tree if >2 moves ahead of root |
| A4 | PUCT constant tuning (cpuct 1.0 → 0.5–0.75) | P2 | Low | Pending | Needs arena validation (100 self-play games old vs new) |
| A5 | Smart accept | P3 | Medium | Pending | Auto-accept MCTS when best move lead is insurmountable; saves compute on obvious moves. Present in twixtbot-ui. |
| A6 | Board symmetry averaging | P3 | High | Future | Run inference on all 8 rotations/reflections and average the policy/value. Improves strength without retraining. Present in twixtbot-ui as "average" rotation mode. |

---

## 2 · Neural Network / Model

All require re-export or retraining.

| ID | Feature | Priority | Effort | Status | Notes |
|----|---------|----------|--------|--------|-------|
| B1 | INT8 quantization of ONNX model | P1 | Low | **Done** | `quantize_model.py` (dynamic INT8, weights only) added as CI step after ONNX export. fp32 model uploaded as `model-onnx-fp32` artifact for reference. Deployed model is now INT8. |
| B7 | Deeper policy head (3×3 conv) | P2 | Medium | Pending | Current head is a 2-ch 1×1 bottleneck; upgrade to 3×3→3×3→1×1→528 (~14K params) |
| B9a | KataGo global pooling bias | P2 | High | Pending | Add GlobalAvgPool→Linear→broadcast-add at each residual block; captures long-range link density. Full retrain needed. |
| B5 | Attention / non-local blocks | P4 | High | Future | ViT-style MHA or non-local blocks; profile inference cost after B9a first |

---

## 3 · Game Analysis & Visualization

Features present in twixtbot-ui but not in the webapp.
The worker already sends much of this data — it just isn't surfaced in the UI.

| ID | Feature | Priority | Effort | Status | Notes |
|----|---------|----------|--------|--------|-------|
| V1 | MCTS progress indicator | P1 | Low | **Done** | Ping interval changed to 1s; `timeLimitMs` added to payload. Thinking overlay shows elapsed/budget (e.g. "3s / 10s"). |
| V2 | Top-3 candidate moves display | P1 | Low | **Done** | Collapsible analysis panel below status bar: top-3 move bars (visit% + Q), win-prob label (`formatWinProb`), and eval sparkline. Worker extended to return `top3` via `top3FromScores()` in naf.ts. |
| V3 | Win probability display | P1 | Low | **Done** | Worker result includes `topQ` (value head output, −1..+1). Show as a labelled bar or numeric readout after each AI move. Present in twixtbot-ui. |
| V4 | Evaluation history chart | P2 | Medium | **Done** | Sparkline moved outside the collapsible analysis body — always visible (with padding) as soon as the analysis panel appears after the first AI move. |
| V5 | Policy heatmap overlay | P2 | Medium | **Done** | On demand, run a single NN forward pass and color each cell by its policy probability (blue→cyan→green gradient). No MCTS needed. Present in twixtbot-ui. |
| V6 | MCTS best-line visualization | P3 | High | Pending | While the bot is computing, draw the current principal variation on the board. Requires the worker to stream the best line in `ping` messages. Present in twixtbot-ui. |

---

## 4 · Game Controls & Rules

| ID | Feature | Priority | Effort | Status | Notes |
|----|---------|----------|--------|--------|-------|
| G1 | Redo | P2 | Medium | **Done** | `redoStack` added to `Game` class in twixt.ts; `redo()` and `canRedo` exposed. In PvC mode redoes the AI move too (stored, no MCTS re-run). Redo button in control bar; strength/think-time moved to settings slide-up panel. |
| G2 | Resign | P2 | Low | **Done** | Explicit resign button ending the game as a loss for the resigning player. Auto-resign threshold (configurable, default 0.95 opponent win prob) is a nice-to-have extension. Present in twixtbot-ui. |
| G3 | Bot vs Bot mode | P3 | Medium | Pending | Let both players be AI (each with auto-move). Useful for demonstration and strength testing. Present in twixtbot-ui. |
| G4 | Allow crossing own links (SCL) | P3 | Medium | Pending | Rule variant: a player's own links may cross each other (non-standard). Present in twixtbot-ui. Off by default. |

---

## 5 · Move List & Board UI

| ID | Feature | Priority | Effort | Status | Notes |
|----|---------|----------|--------|--------|-------|
| U1 | Move list display | P2 | Low | **Done** | Collapsible panel (hidden until first move). Shared `renderMoveList()` function: paired rounds (WHITE left, BLACK right), current half-move highlighted, auto-scroll. |
| U2 | Coordinate tooltip on hover/drag | P3 | Low | **Done** | Show the cell coordinate (e.g. "h5") near the cursor/finger while hovering or dragging. Present in twixtbot-ui. |
| U3 | Show/hide board labels toggle | P3 | Low | Pending | Allow hiding the column/row letter labels. Present in twixtbot-ui. |
| U4 | Show/hide guidelines toggle | P3 | Low | Pending | Allow hiding the knight-move guide lines. Present in twixtbot-ui. |
| U5 | Player name customization | P3 | Low | Pending | Configurable display names for each player (shown in status bar). Present in twixtbot-ui. |
| U6 | Player color customization | P4 | Low | Future | User-selectable peg colors for each player. Present in twixtbot-ui. Lower priority given the accessible blue/orange defaults. |

---

## 6 · Little Golem Integration

### LG · MVP (shipped ✓)

- Player search by name → game list with rating
- Game replay via step controls (first/prev/next/last + arrow keys)
- Search by game ID directly
- Paste / upload `.tsgf` file
- Cloudflare Worker CORS proxy
- Back navigation preserving cached results
- Player color display (PB/PW from SGF)
- Result display from SGF `RE` field
- Highlight last move

### LG · Analysis in Replay

| ID | Feature | Priority | Effort | Status | Notes |
|----|---------|----------|--------|--------|-------|
| L1 | "Analyse this position" button in replay | P1 | Low | **Done** | Analyse button in replay header; panel shows win-prob + top-3 bars below board. Reuses worker; `replayAnalysisMode` flag gates result handler. Panel clears on move navigation. |
| L2 | Move quality overlay | P2 | Medium | **Done** | "Analyse game" button runs `eval-game` worker batch (single NN forward pass per position). Move list entries coloured green (rank 0) / yellow (rank 1–4) / red (rank 5+) as results stream in. |
| L3 | Evaluation graph | P2 | Medium | **Done** | Same `eval-game` batch as L2. Sparkline below move list shows topQ at each position with a vertical marker at the current replay move. Updates incrementally as results stream in. |
| L4 | Move list panel in replay | P2 | Low | **Done** | Same `renderMoveList()` as U1. Replay screen: always present, clickable rows jump to that position via `replayShowAtIndex(i+1)`. Refreshes on every navigation. |

### LG · Game List Filters (Explore screen)

Both filters operate on the already-fetched `GameSummary[]` array — no new network
requests. `GameSummary.result` is "win" | "lost" | "draw" (player perspective) and
`GameSummary.opponent` is the display name, both parsed from the existing HTML scrape.

| ID | Feature | Priority | Effort | Status | Notes |
|----|---------|----------|--------|--------|-------|
| L10 | Result filter | P2 | Low | **Done** | Chip-row filter above the game list: `[All] [Win] [Loss] [Draw]`. Client-side filter on `result`. Chips are mutually exclusive; "All" resets. Count badge per chip (e.g. "Loss (8)"). |
| L11 | Opponent filter | P2 | Low | **Done** | Dropdown or text filter populated from unique `opponent` values in the loaded list. Combinable with L10 (e.g. "show only losses against TwixtBot"). Resets when a new player is searched. |

**Combined UX sketch:**

```
┌─────────────────────────────────────┐
│  Alan Hensel  ·  47 games           │  ← player header
│  [All ▼] [Win] [Loss] [Draw]        │  ← result chips (L10)
│  vs: [ any opponent      ▾ ]        │  ← opponent dropdown (L11)
├─────────────────────────────────────┤
│  #2546140  vs TwixtBot  ·  31 moves │  Loss
│  #2501234  vs Peyrol    ·  38 moves │  Win
│  …                                  │
└─────────────────────────────────────┘
```

The opponent dropdown is populated lazily from the current loaded list (no extra fetch).
If fewer than 2 distinct opponents exist, the dropdown is hidden.

---

### LG · Infrastructure / Reliability

| ID | Feature | Priority | Effort | Status | Notes |
|----|---------|----------|--------|--------|-------|
| L5 | Offline replay cache | P2 | Low | Pending | Cache raw SGF text in localStorage after first load so previously-viewed games are available offline. |
| L6 | Recently viewed games | P3 | Low | Pending | Store last N game IDs in localStorage; show as "Recent" section at top of LG screen. |
| L7 | Proxy health check | P3 | Low | Pending | Show warning banner if Cloudflare Worker is unreachable, rather than a silent network error. |
| L8 | Favourite players | P3 | Low | Pending | Star players appear at the top of search results without re-typing. |
| L9 | "TwixT bot games" shortcut | P3 | Low | Pending | One-tap to search games where TwixtBot (plid=3101) is a player. |

### LG · Correspondence Play (requires auth investigation)

> **Blocked** pending verification of LG auth/move endpoints. See open items in
> `docs/lg-planned-features.md`.

| ID | Feature | Priority | Effort | Status | Notes |
|----|---------|----------|--------|--------|-------|
| N1 | LG login screen | P2 | Medium | Future | POST credentials via Cloudflare proxy; store session cookie in localStorage |
| N2 | "My active games" dashboard | P2 | Medium | Future | List ongoing TwixT games with whose-turn indicator |
| N3 | Play a move from the app | P1 | Medium | Future | Submit move to LG from within the app — our key differentiator vs LG's own UI |
| N4 | AI hint before submitting | P1 | Low | Future | Run local MCTS on the current LG position before committing a move. Zero new ML work — reuses existing worker. |
| N5 | Watch ongoing games (read-only) | P3 | Low | Future | Show in-progress games in replay mode up to the latest move |
| N6 | Waiting Room browser | P4 | Medium | Future | Browse/post open game invitations; LG's own UI is adequate for now |

---

## 7 · Network PvP (Two devices, peer-to-peer)

Full research in `docs/network-pvp-research.md`. Recommendation: PeerJS + OpenRelay TURN.

| ID | Feature | Priority | Effort | Status | Notes |
|----|---------|----------|--------|--------|-------|
| R1 | `RemoteConnection` manager (WebRTC/PeerJS) | P3 | Medium | Future | ~100 lines; offer/answer/ICE state machine |
| R2 | Game message protocol | P3 | Low | Future | ~50 lines; move/undo/resign/new-game sync over DataChannel |
| R3 | Host/Join UI (room code) | P3 | Low | Future | ~80 lines; room code display, join input, connection status |
| R4 | Full network PvP mode (`pvp-remote`) | P3 | High | Future | Combines R1–R3; third game mode alongside `pvc` and `pvp` |

---

## Recommended next actions

Based on priority and the constraint that analysis/viz features are mostly zero-cost
(data already available in the worker):

1. **V1 + V2 + V3** — Wire up existing worker output to a small analytics panel.
   All data already flows from the worker; this is pure UI work. Highest ratio of
   value to effort in the entire backlog.

2. **B1** — INT8 quantize the ONNX model. One Python call, immediate iOS memory relief.

3. **L1** — "Analyse this position" in replay. The AI worker and board are already
   there; just need a button that feeds the current replay position to `requestAiMove`.

4. **G1 + G2** — Redo and Resign. Both are medium-effort but frequently expected by
   players coming from other board game apps.

5. **A3 + A4** — Tree reuse and cpuct tuning. Strongest pure-algorithm gains without
   retraining. A4 needs an arena to validate — set that up alongside A3.

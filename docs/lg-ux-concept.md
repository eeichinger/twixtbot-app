# Little Golem Integration — UX Concept & Implementation Plan

> **Feature status and priority tracked in `docs/planned-features.md` (section 6).
> This document is the UX design and technical reference; do not duplicate status here.**

## Context

The twixtbot-app is a self-contained PWA for local TwixT play (vs AI or same-device PvP). This plan adds integration with Little Golem (LG), a correspondence game server with an active TwixT community. The integration enables users to browse LG games and step through them move-by-move for analysis and learning.

**MVP scope** (confirmed by user): **Explore & Replay** — browse LG games by game ID or player name, download the SGF record, and step through moves interactively on the app's board canvas. No LG account required. Live correspondence play, challenges, and AI suggestion are deferred to future iterations.

---

## Guiding Principles

- **Non-disruptive**: Existing PvC and PvP flows are unchanged. LG is a third "pillar" added alongside them.
- **No login required**: LG's SGF download endpoint is public. The Explore/Replay feature works for anyone.
- **Reuse the board**: The existing `BoardUI` canvas renderer handles replay — same look, different controls.
- **Offline awareness**: Replay requires network to fetch SGF. Previously-fetched games could be cached; show a clear offline error otherwise.
- **CORS is an open question**: Decided during implementation (see below). UX is designed independently of the proxy strategy.

---

## Full UX Concept (including deferred features)

### Navigation structure

```
Intro Screen
  ├── vs Computer   →  [Loading]  →  Game Screen (PvC)
  ├── vs Player     →  Game Screen (PvP)
  └── Little Golem  →  LG Hub
        ├── Explore tab   [MVP]   →  Replay Viewer
        ├── My Games tab  [later] →  Game Screen (LG move submission)
        └── Challenges tab [later]→  Accept/send challenges
```

### Screens overview

| Screen | Status | Description |
|---|---|---|
| Intro Screen | Modified | Adds "Little Golem" mode button (badge with pending turn count, later) |
| LG Hub | New | Tabbed container: Explore (MVP), My Games (later), Challenges (later) |
| Replay Viewer | New | Board canvas in read-only replay mode with step controls |
| Game Screen (LG mode) | Later | Existing canvas + submit-move controls for correspondence play |

---

## MVP: Explore & Replay

### Screen 1 — Intro Screen (small change)

Add a third mode button **"Little Golem"** alongside "vs Computer" and "vs Player". Same button style, same toggle behaviour. When selected and "Start" pressed, navigates to the LG Hub screen.

No badge or live-data for MVP (those require auth + polling, deferred).

---

### Screen 2 — LG Hub (Explore tab only for MVP)

Since only the Explore tab is in scope for MVP, the Hub can initially be presented as the Explore screen directly (no tab bar needed until My Games is added). This avoids building tab infrastructure prematurely.

**Layout:**
```
┌─────────────────────────────────────────┐
│  ← Back        Little Golem             │
├─────────────────────────────────────────┤
│  Search                                 │
│  ┌─────────────────────────┬──────────┐ │
│  │ Player name or Game ID  │  Search  │ │
│  └─────────────────────────┴──────────┘ │
│                                         │
│  Results                                │
│  ┌─────────────────────────────────────┐│
│  │ #12345  Player1 (B) vs Player2 (W) ││
│  │  24×24  •  47 moves  •  Black won  ││
│  └─────────────────────────────────────┘│
│  ┌─────────────────────────────────────┐│
│  │ ...                                ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

**Search behaviour:**
- If input looks like a number → treat as Game ID, fetch that specific game directly
- If input looks like text → treat as LG player name, fetch game list for that player, show results
- Results show: game ID, both player names, colors, board size, move count, winner
- Tapping a result opens the Replay Viewer
- Empty state: prompt text ("Enter a player name or game ID")
- Error state: "Game not found", "Network error — check connection"

**Player name search** fetches the list of that player's TwixT games from LG (HTML scrape of their game history), filtered to TwixT PP games. Results sorted newest-first.

**No account needed**: everything is public read-only access to LG.

---

### Screen 3 — Replay Viewer

Reuses the existing `BoardUI` canvas. Replays a game by applying moves to a fresh `Game` instance one at a time.

**Layout (replaces normal game-screen controls):**
```
┌──────────────────────────────────────────┐
│  ←  Player1 (B) vs Player2 (W)  #12345  │  ← header / back
├──────────────────────────────────────────┤
│                                          │
│            [ board canvas ]              │
│                                          │
├──────────────────────────────────────────┤
│  |◀   ◀   Move 14 / 47   ▶   ▶|         │  ← step controls
├──────────────────────────────────────────┤
│  Moves  ▼  (expandable list)             │  ← move list panel
│  1. h4  2. swap  3. d6  4. f8 ...       │
└──────────────────────────────────────────┘
```

**Step controls:**
- `|◀` — jump to start (move 0, empty board)
- `◀` — step back one move
- `▶` — step forward one move
- `▶|` — jump to end (final position)
- Move counter: "Move N / Total"

**Move list panel:**
- Collapsed by default (tap to expand/collapse)
- Scrollable list of all moves in algebraic notation: `1. h4`, `2. swap`, `3. d6`, …
- Current move highlighted
- Tapping any move in the list jumps directly to that position

**Visual treatment of replayed moves:**
- The most recently placed peg is highlighted (different shade or ring) to make it easy to spot
- Both players' pegs use the standard Blue/Orange colour scheme

**Last move and winner display:**
- At the final move, if there is a winner, show the win result in the header (same as current game-over message)

**Back button:** returns to LG Hub (Explore tab / search results preserved)

---

## Use Cases (MVP)

### UC-1: Replay a game by ID

1. User selects "Little Golem" on Intro → Start
2. LG Hub / Explore screen opens
3. User types a game ID (e.g. `2345678`) → Search
4. Game card appears with player names, result
5. User taps card → Replay Viewer opens at move 0
6. User taps ▶ repeatedly (or taps the final move in the list) to step through
7. Back → returns to Explore results

### UC-2: Browse a player's games

1. User types a player name (e.g. `twixtbot`) → Search
2. List of games for that player appears
3. User can scroll, tap any game to replay

### UC-3: Jump to a specific moment

1. User is in the Replay Viewer
2. User expands the move list
3. User taps move 23 → board jumps directly to that position

### UC-4: Review final position first, then step back

1. User taps `▶|` → jumps to final position, sees who won
2. User taps `◀` repeatedly to trace back to a key decision point

---

## Technical Notes

### SGF format used by LG

Little Golem's `.tsgf` files use **Go-style two-letter coordinate encoding** internally:
```
(;GM[21]FF[4]SZ[24]RU[PP]PB[player1]PW[player2]RE[B+]
;B[hd];W[ab];B[swap];W[fg]...)
```

- `GM[21]` = TwixT
- `RU[PP]` = Pen and Paper rules (links never removed)
- Moves: `B[hd]` = Black plays column h, row d (4th letter → row 4) → `h4`
- Conversion: `col = move[0]`, `row = ord(move[1]) - ord('a') + 1`
- Special values: `swap`, `SWAP`, `resign`
- SGF download URL (no auth): `https://www.littlegolem.net/servlet/sgf/{id}/game{id}.tsgf`

### CORS (open question — must resolve before implementation)

LG does not send CORS headers. Browser `fetch()` to `littlegolem.net` will be blocked.

**Options:**
1. **Thin Cloudflare Worker proxy** (recommended): A single-file CF Worker that forwards GET requests to LG's public SGF/game-list endpoints. Free tier is ample for this read-only usage. No auth secrets involved.
2. **Public CORS proxy** (e.g. `corsproxy.io`): Zero infrastructure, but adds an untrusted third party in the read path. Acceptable for public game data but introduces a dependency.
3. **Manual SGF paste**: Fallback UI allowing user to paste raw SGF text. Zero infrastructure but friction.

All three can be layered: try direct fetch → try proxy → offer manual paste as fallback.

Decision deferred; the UX and SGF parsing logic are independent of this choice.

### Player game list scraping

LG game history page: `https://www.littlegolem.net/jsp/info/player_game_list_txt.jsp?gtid=twixt&plid={player_id}` (or similar path — needs verification against live site). Returns an HTML table of games. Parse with regex for game IDs. Same CORS issue as above.

### Existing code to reuse

| What | File | Notes |
|---|---|---|
| `Game` class | `webapp/src/twixt.ts` | Use `game.play(move)` to replay moves; `new Game()` for fresh board |
| `BoardUI` | `webapp/src/ui.ts` | Pass `board.setEnabled(false)` for read-only mode |
| `twixt.ts` point helpers | `webapp/src/twixt.ts` | Point encoding/decoding already exists |
| Game mode helpers | `webapp/src/game-mode.ts` | Likely not reused; replay is its own mode |
| localStorage pattern | `webapp/src/main.ts` | Use same pattern for persisting last search query |
| Screen show/hide pattern | `webapp/src/main.ts` + `index.html` | Add `#lg-screen` alongside existing screens |
| `.hidden` class | `webapp/src/style.css` | Use existing utility |

### New code needed

| What | Where |
|---|---|
| SGF parser (`parseTSGF(text)` → move array) | `webapp/src/lg-sgf.ts` (new file) |
| LG API client (`fetchGame(id)`, `fetchPlayerGames(name)`) | `webapp/src/lg-api.ts` (new file) |
| Replay controller (step logic, move index state) | `webapp/src/replay.ts` (new) or inline in `main.ts` |
| LG Hub / Explore screen HTML | `webapp/index.html` — new `#lg-screen` div |
| Replay Viewer controls HTML | `webapp/index.html` — new `#replay-screen` div (or reuse `#game-screen` with mode-specific controls) |
| Styles for new screens | `webapp/src/style.css` |

---

## Verification

1. Search for a known game ID (e.g. a game from `https://www.littlegolem.net/jsp/game/game.jsp?gid=XXXXX`) → game card appears with correct player names and result
2. Open game → replay viewer shows empty board at move 0
3. Step forward through all moves → each move places correct peg in correct position; links appear as expected
4. Jump to final position → winner displayed correctly
5. Step backwards → board returns to earlier positions correctly
6. Move list → tapping a move jumps to that position
7. Search by player name → list of games appears
8. Offline with no cached game → clear error message, no hang
9. Existing PvC, PvP modes unaffected

---

## Confirmed LG URL Map (verified from live HTML)

| Purpose | URL | Notes |
|---|---|---|
| Player search | `/jsp/info/player_list.jsp?gtvar=twixt_DEFAULT&filter=NAME` | Public HTML; rows: `<a href="player.jsp?plid=NNNN">Name</a>` |
| Player profile | `/jsp/info/player.jsp?plid=PLID` | Has stats table with game list link |
| Player game list | `/jsp/info/player_game_list.jsp?gtid=twixt&plid=PLID` | Public HTML; game links: `game.jsp?gid=NNNN` |
| SGF download | `/servlet/sgf/{id}/game{id}.tsgf` | Publicly accessible (confirmed) |

**Current bug in `lg-api.ts` `fetchPlayerGamesByPlid`:** uses wrong URL:
- Wrong: `player_game_list_txt.jsp?gtid=twixt.PP&plid=PLID`
- Correct: `player_game_list.jsp?gtid=twixt&plid=PLID`

Two errors: `_txt` suffix (export format, may not be public) and `twixt.PP` instead of `twixt`.

**Fix (one line):** In `webapp/src/lg-api.ts`, `fetchPlayerGamesByPlid()`, change:
```typescript
const txtUrl = `${LG_BASE}/jsp/info/player_game_list_txt.jsp?gtid=twixt.PP&plid=${encodeURIComponent(plid)}`;
```
to:
```typescript
const url = `${LG_BASE}/jsp/info/player_game_list.jsp?gtid=twixt&plid=${encodeURIComponent(plid)}`;
```
and remove the two-attempt fallback logic — fetch HTML directly, call `parseGameListHtml()` on the result.


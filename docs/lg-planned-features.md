# Little Golem Integration — Planned Features

Tracks ideas and scope for the LG Explore & Replay feature.
Check items off as they are shipped.

---

## MVP — shipped ✓

- [x] **Player search** — search by name → player list with rating
- [x] **Game list** — tap player → list of their recent TwixT PP games
      (opponent name, move count, win/lost/draw result)
- [x] **Game replay** — tap game → load SGF via proxy → step through moves
      on a read-only board (first/prev/next/last + arrow keys)
- [x] **Search by game ID** — enter a numeric LG game ID directly in the
      search box to jump straight to replay
- [x] **Paste / upload SGF** — paste raw `.tsgf` text or upload a file
      from the device; no network request needed
- [x] **CORS proxy** — Cloudflare Worker (`littlegolem-proxy.eeichinger.workers.dev`)
      forwards GET requests to LG; restricted to `littlegolem.net` targets
- [x] **Mock mode** — `MOCK_MODE` flag in `lg-api.ts` for offline UX testing
      without real LG requests; mock data includes a real game (id 2060663)
- [x] **Back navigation** — replay → game list (restores from cache, no re-fetch);
      game list → player list (same); any screen → intro

---

## Near-term ideas

- [x] **Show which color the player was** — `PB`/`PW` parsed from SGF;
      replay title shows "BlackPlayer vs WhitePlayer #id".

- [x] **Result from SGF perspective** — `RE` field parsed in `lg-sgf.ts`;
      `formatResult()` converts "B+"/"W+"/"0" to human-readable text;
      shown in the game list card.

- [x] **Highlight last move** — red ring (`#cc2040`) drawn on the most
      recently stepped-to peg in `ui.ts`.

- [x] **Keyboard shortcut** — arrow keys (←/→) and Home/End fully working
      in `main.ts`; active whenever the replay screen is visible.

- [x] **Pagination / load more** — not needed. Verified: `player_game_list.jsp`
      returns all games for a player in a single response with no server-side
      pagination.

---

## LG Network Play — play against other humans on Little Golem

> **Important but complex — open investigation items must be resolved before
> implementation can begin (see below).**

### How LG correspondence play works

- **Turn-based correspondence** — players do NOT need to be online at the same time.
  Games take days/weeks. Each player has 36 hours per move + 10-day grace period.
- **Move flow**: visit LG in browser → click to place peg → server records it,
  auto-adds all legal links, emails your opponent. No real-time connection needed.
- **Authentication**: registration by email required. Login is a username/password
  form that returns a session cookie. All game actions require an authenticated session.
- **Finding opponents — three paths**:
  1. **Waiting Room** — post an open invitation; others accept
  2. **Direct challenge** — visit a player's profile, send invite
  3. **Tournaments** — league play (round-robin ~9 players) and Monthly Cups
     (5-player brackets, auto-paired by rating)
- **TwixT on LG**: TwixT PP ruleset; links auto-added by server and never deleted;
  board sizes 24×24, 30×30, 48×48; small but serious community

### What fits well in twixtbot-app

**Tier 1 — high value, buildable**

- **"My active games" dashboard** — after LG login, show all your ongoing TwixT
  games: opponent, whose turn, move count. Tap to open the board. Directly reuses
  the existing board rendering and game list UI patterns. Foundation for everything
  else.

- **Play a move from the app** — when it's your turn: open the game (board rendered
  via SGF), tap to place peg, confirm, submit to LG. Replaces LG's clunky web UI
  with our clean mobile board. Board interaction code is already in place; the only
  new piece is a POST call to LG to submit the move.

- **AI hint before committing** — our biggest differentiator vs LG's own UI.
  Before tapping "Submit", invoke the AI worker on the current position. LG has zero
  analysis tools. No other TwixT mobile client offers this. Reuses the existing AI
  worker with zero new ML work.

**Tier 2 — nice to have later**

- **Watch ongoing games** — show in-progress games in read-only replay mode (board
  up to latest move). SGF for ongoing games is publicly accessible so this is almost
  free once the "my games" list is implemented.

- **Waiting Room browser** — browse/post open game invitations. Low priority;
  LG's web UI is adequate for this.

**Out of scope**

- Direct challenge flow — secondary to the play-your-move flow
- Tournament management — too complex, LG's own UI is fine
- 30×30 / 48×48 board sizes — engine is hardcoded to 24×24

### Open investigation items (required before implementation)

**[OPEN] LG authentication endpoint**
- LG login is likely a form POST to something like `/jsp/login/login.jsp` with
  `login=USER&password=PASS` fields, returning a session cookie
- To verify: submit the LG login form via browser devtools → capture exact POST URL,
  request body field names, and the returned cookie (name, domain, path, lifetime)
- The Cloudflare Worker currently only handles unauthenticated GETs; it needs a new
  route that forwards the login POST and relays the session cookie back to the client

**[OPEN] Move submission endpoint**
- LG records moves via an HTTP POST (likely form-encoded) with game ID + coordinate
- To verify: play a real move on LG via browser devtools → capture exact POST URL,
  form field names, coordinate format (may differ from SGF two-letter encoding),
  and whether a CSRF token is required
- If CSRF token is present: must first GET the game page, extract the token, then
  POST — a two-request flow through the proxy

**[OPEN] "My active games" endpoint**
- Need the URL for the logged-in user's ongoing-games list (e.g. a dashboard or
  filtered `player_game_list.jsp` variant accessible only when authenticated)
- To verify: log into LG, find the "my turn" / "my active games" page, capture URL
  and HTML structure

**[OPEN] Session cookie lifetime**
- How long does an LG session last? Short sessions require re-auth prompts.
- Does LG have a "remember me" option for a long-lived token?

**[OPEN] Cloudflare Worker extensions needed**
- Currently: unauthenticated GET proxy, restricted to `littlegolem.net`
- Needed for network play:
  - Forward authenticated GETs with `Cookie:` header (for private/ongoing game SGFs)
  - Forward login and move-submission POSTs
  - Hard security constraint: domain check must be enforced in Worker code —
    authenticated requests must ONLY ever be forwarded to `littlegolem.net`,
    regardless of what the client sends as the target URL

### Proposed MVP scope

Two screens, built in this order:

1. **LG Login** — username + password → POST via proxy → session cookie stored in
   localStorage → navigate to My Games
2. **My Games** — fetch active TwixT games for the logged-in user → list with
   opponent / whose-turn indicator → tap → board renders via SGF → optional AI hint
   → tap to submit move via proxy POST

Everything else (Waiting Room, watch mode, challenges) comes after this slice works.

### Security constraints (non-negotiable)

- LG credentials stored only in `localStorage`; never sent to any server except
  `littlegolem.net` via the Cloudflare Worker
- Cloudflare Worker must hard-code the domain check: authenticated requests
  forwarded only to `https://www.littlegolem.net` — not derived from caller input
- No credential logging anywhere in the Worker
- UI must clearly warn the user that credentials are stored locally (unencrypted)

---

## Analysis / AI integration ideas

- [ ] **"Analyse this position" button in replay** — from any move in the
      replay, launch the AI worker on the current board position and display
      the best move / evaluation, just like the hint button in the main game.

- [ ] **Move quality overlay** — for each move in the game, compare the move
      played to the AI's top choice. Colour the move dots green/yellow/red
      based on how much policy probability the AI assigned to the played move.
      Requires running inference on every position — may be slow.

- [ ] **Evaluation graph** — plot the AI's win-probability estimate at each
      move as a sparkline below the board, showing how the game's fortunes
      shifted. Requires inference on every position.

---

## Social / discovery ideas

- [ ] **Recently viewed games** — store the last N game IDs in localStorage
      and show them as a "Recent" section at the top of the LG screen.

- [ ] **Favourite players** — let the user star players; starred players
      appear at the top of the search results without typing.

- [ ] **"Games vs TwixT bot" shortcut** — a one-tap button to search for
      games where TwixtBot (plid=3101) is one of the players.

---

## Infrastructure / reliability

- [ ] **Offline replay of previously loaded games** — cache the raw SGF text
      in localStorage after first load so the user can revisit games offline.

- [ ] **Proxy health check** — show a warning banner if the Cloudflare Worker
      is unreachable (e.g. worker quota exceeded), rather than a generic
      network error on the first LG request.

- [ ] **Rate limiting / backoff** — if the user hammers the search button,
      debounce requests to avoid hitting LG or the Worker excessively.

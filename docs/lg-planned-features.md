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

> **Important but complex — needs conceptualization before implementation.**
>
> The idea: use twixtbot-app as a client to play correspondence TwixT games
> on Little Golem against other human players, rather than only replaying
> finished games. LG is a turn-based (correspondence) server, so moves are
> submitted via HTTP, not a live socket.
>
> Key questions to answer during conceptualization:
> - Does LG expose a usable API for submitting moves, or does it require
>   form POST replication scraped from the game page HTML?
> - How does authentication work — session cookie, username/password form?
>   The Cloudflare Worker proxy would need to forward auth headers or handle
>   login on behalf of the user.
> - How does the app poll for the opponent's move (LG has no push/websocket)?
> - Where does the user's LG login credential live — entered once and stored
>   in localStorage, or re-entered each session?
> - How does this interact with the existing PvC and PvP modes — new mode
>   on the intro screen, or a separate entry point from the LG Explore screen?
> - Privacy/security: credentials must never be sent anywhere except LG via
>   the proxy; the proxy must be hardened to only forward to `littlegolem.net`.

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

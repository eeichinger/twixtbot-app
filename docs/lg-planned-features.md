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

- [ ] **Show which color the player was** — the game list page doesn't say
      whether the searched player was Black or White; this is only available
      from the SGF. After loading the SGF, compare `PB`/`PW` to the player
      name and label the card accordingly.

- [ ] **Result from SGF perspective** — game list shows "win/lost" from the
      searched player's perspective; replay header could show "Black wins" /
      "White wins" (from the SGF `RE` field) once the game is loaded.

- [ ] **Highlight last move** — draw a ring or glow on the peg placed in the
      most recently stepped-to move, so the viewer can immediately see where
      the action is.

- [ ] **Auto-advance / autoplay** — a Play button that steps through moves
      automatically at a configurable speed (e.g. 1 move/sec).

- [ ] **Keyboard shortcut** — already works (arrow keys); confirm it works
      on desktop and document it somewhere visible in the UI.

- [ ] **Pagination / load more** — the game list currently shows all games
      returned by `player_game_list.jsp`. LG paginates at ~20 rows; add a
      "Load more" button if the list is truncated.

- [ ] **Current (in-progress) games** — `player_game_list.jsp` only shows
      finished games. In-progress games appear on a different LG page;
      investigate whether they can be loaded for analysis mid-game.

- [ ] **Deep link / shareable URL** — encode the LG game ID in the URL hash
      (`#lg/2060663`) so a user can share a direct link to a replay. The PWA
      start_url constraint makes this non-trivial (service worker navigation
      fallback must handle the hash).

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

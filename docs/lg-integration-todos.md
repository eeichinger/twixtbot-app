# Little Golem Integration — Verification Todos

Assumptions to verify before finalising the integration. Work through these
one by one using a real browser / curl against live LG URLs.

---

## Open items

- [ ] **V1 — corsproxy.io works against LG**
  Manually fetch an LG URL through corsproxy.io in a browser or with curl:
  ```
  curl "https://corsproxy.io/?url=https%3A%2F%2Fwww.littlegolem.net%2Fservlet%2Fsgf%2F2546140%2Fgame2546140.tsgf"
  ```
  Verify: response body is the raw `.tsgf` text (not JSON-wrapped, not a login
  page). Check response status is 200. Check no extraneous headers or body
  wrapper are injected.

- [x] **V2 — `player_game_list.jsp` HTML structure** ✓ RESOLVED (April 2026)
  Verified from live HTML for plid=2674 (Alan Hensel). The page is a static
  HTML table — no JS injection. Each row contains:
  - Game link: `href="/jsp/game/game.jsp?gid=NNNN"` ✓
  - Opponent name: 2nd `<td>` in the row
  - Move count: 5th `<td>`, plain integer
  - Result: 6th `<td>`, text `win` / `lost` / `draw` (player's perspective)
  - Board size: "Size 24" text in the tournament column (all games are 24×24)

  `parseGameListHtml` has been upgraded from ID-only to full row extraction.
  Opponent, move count, and result are now parsed. The `GameSummary` type
  gained an `opponent?: string` field.

- [ ] **V3 — `player_list.jsp` rating span proximity**
  Fetch the player search results page:
  ```
  https://www.littlegolem.net/jsp/info/player_list.jsp?gtvar=twixt_DEFAULT&filter=alan
  ```
  Inspect the HTML table row for a player. Confirm the rating `<span>` appears
  within ~300 characters *after* the `player.jsp?plid=NNNN` link in the raw
  HTML (the current `parsePlayerListHtml` reads a 300-char window).
  If the span is further away or in a different column order, update the
  extraction window/regex.

- [ ] **V4 — Remove dead `parseGameListTxt` function**
  `parseGameListTxt` in `lg-api.ts` (lines ~156–175) is no longer called after
  the two-attempt fallback was removed. Either:
  (a) delete it entirely, or
  (b) verify `player_game_list_txt.jsp` is accessible without login, confirm
      the tab-separated column layout, and restore the txt path as the primary
      fast path with HTML as fallback.
  Decision needed before shipping.

- [ ] **V5 — Non-24 board size handling**
  All games visible in the Alan Hensel game list show "Size 24" — strong
  indication LG only runs 24×24 TwixT PP. Confirm no other size exists.
  If all LG TwixT PP games are 24×24: add a guard in `fetchGame` to reject
  non-24 SGFs gracefully rather than mis-replaying.
  If other sizes exist: `replayShowAtIndex` in `main.ts` creates `new Game()`
  which hard-codes 24×24 — fix it to use `parsedGame.boardSize`.

- [ ] **V6 — corsproxy.io response encoding for SGF**
  Confirm the SGF response from corsproxy.io is plain UTF-8 text (not
  base64-encoded or JSON-wrapped). The current code calls `res.text()` directly.
  If the proxy wraps the body (e.g. `{ "contents": "..." }`), the `parseTSGF`
  call will fail silently (no `GM[21]` found). Check with the curl test above.

---

## Resolved items

- [x] SGF download URL is public — confirmed `/servlet/sgf/{id}/game{id}.tsgf`
      requires no login (user verified with live URL)
- [x] `player_list.jsp` link format — `href="player.jsp?plid=NNNN"` confirmed
      from user-pasted live HTML
- [x] `player_game_list.jsp` correct URL — confirmed `gtid=twixt` (not
      `twixt.PP`) from user-pasted `player.jsp` HTML showing the nav link
- [x] `player_game_list_txt.jsp` is not the right endpoint — removed from code
- [x] LG has no CORS headers — confirmed; all direct requests blocked

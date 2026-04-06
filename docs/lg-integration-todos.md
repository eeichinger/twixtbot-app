# Little Golem Integration — Verification Todos

Assumptions to verify before finalising the integration. Work through these
one by one using a real browser / curl against live LG URLs.

---

## Open items

- [ ] **V1 — Cloudflare Worker proxy works against LG** *(replaces corsproxy.io)*
  corsproxy.io blocks the required content-type without a paid subscription.
  A Cloudflare Worker is the replacement. See `docs/cloudflare-proxy-setup.md`
  for the full step-by-step setup and verification guide.
  Verify:
  - Worker deployed and accessible at `https://<worker>.workers.dev`
  - SGF fetch returns raw `.tsgf` text (not JSON-wrapped)
  - Player list and game list HTML fetches return raw HTML
  - `PROXY_PREFIX` in `lg-api.ts` updated to point to the Worker URL
  - MOCK_MODE flipped to `false` and app tested end-to-end

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

- [x] **V4 — Remove dead `parseGameListTxt` function** ✓ RESOLVED (April 2026)
  `parseGameListTxt` was deleted entirely during the cleanup pass. The HTML
  endpoint (`player_game_list.jsp`) is confirmed public and fully parsed by
  `parseGameListHtml`. No txt fallback needed.

- [ ] **V5 — Non-24 board size handling**
  All games visible in the Alan Hensel game list show "Size 24" — strong
  indication LG only runs 24×24 TwixT PP. Confirm no other size exists.
  If all LG TwixT PP games are 24×24: add a guard in `fetchGame` to reject
  non-24 SGFs gracefully rather than mis-replaying.
  If other sizes exist: `replayShowAtIndex` in `main.ts` creates `new Game()`
  which hard-codes 24×24 — fix it to use `parsedGame.boardSize`.

- [ ] **V6 — Proxy response encoding for SGF** *(depends on V1)*
  Confirm the SGF response from the Cloudflare Worker is plain UTF-8 text (not
  base64-encoded or JSON-wrapped). The current code calls `res.text()` directly.
  Covered by the V1 end-to-end test if the replay screen shows moves correctly.

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
- [x] corsproxy.io — PARKED: blocks required content-type without paid plan

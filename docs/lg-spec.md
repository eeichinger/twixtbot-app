# Little Golem — API & Data Extraction Spec

This document records all verified knowledge about querying Little Golem (LG)
for TwixT player and game data. It covers public endpoints, HTML structure,
SGF format, CORS constraints, and parsing patterns.

All findings were verified against live LG HTML during implementation of the
Explore & Replay feature (April 2026).

---

## Access Model

- LG runs on `https://www.littlegolem.net`
- **No login is required** for the endpoints described in this document
- The SGF download endpoint and the public player/game list pages are all
  accessible without authentication
- LG does **not** send CORS headers — browser `fetch()` is blocked by
  same-origin policy; a CORS proxy is required (see below)

---

## CORS Proxy

Since LG does not send `Access-Control-Allow-Origin` headers, a proxy is
required for all browser requests. The current implementation uses
`corsproxy.io` as a zero-infrastructure default:

```
https://corsproxy.io/?url=<url-encoded-LG-url>
```

Alternatives:
- A thin Cloudflare Worker (recommended for production — no third-party trust)
- Any other CORS proxy that forwards GET requests and passes through the
  original response body and status code

The proxy is configured via a single constant `PROXY_PREFIX` in `lg-api.ts`.

---

## URL Reference

All paths are relative to `https://www.littlegolem.net`.

| Purpose | URL | Auth | Notes |
|---|---|---|---|
| Player search | `/jsp/info/player_list.jsp?gtvar=twixt_DEFAULT&filter=NAME` | None | Filters to TwixT players matching NAME |
| Player profile | `/jsp/info/player.jsp?plid=PLID` | None | Stats page; links to game list |
| Player game list | `/jsp/info/player_game_list.jsp?gtid=twixt&plid=PLID` | None | All finished TwixT games for a player |
| Game detail page | `/jsp/game/game.jsp?gid=GID` | None | Web UI for a single game |
| SGF download | `/servlet/sgf/GID/gameGID.tsgf` | None | Raw SGF file; publicly accessible |

### Query parameters

**Player search** (`player_list.jsp`):
- `gtvar=twixt_DEFAULT` — filters list to TwixT game type
- `filter=NAME` — substring match on player name (URL-encoded)
- `countryid` — optional country filter (omit for all countries)
- `page` — pagination (omit for first page)

**Player game list** (`player_game_list.jsp`):
- `gtid=twixt` — filters to TwixT game type; use `twixt` (not `twixt.PP`)
- `plid=PLID` — numeric player ID

> **Common mistake:** Using `gtid=twixt.PP` returns no results. The correct
> value is `gtid=twixt`. Similarly, `player_game_list_txt.jsp` (the export
> variant) is not publicly accessible and should not be used.

---

## Player Search Flow

Player search is a **two-step** process because LG game lists are indexed by
numeric player ID (`plid`), not by name.

```
Step 1: searchPlayers(name)
  GET /jsp/info/player_list.jsp?gtvar=twixt_DEFAULT&filter=NAME
  → parse HTML → PlayerResult[] { plid, name, rating }

Step 2: fetchPlayerGamesByPlid(plid)
  GET /jsp/info/player_game_list.jsp?gtid=twixt&plid=PLID
  → parse HTML → GameSummary[]
```

There is no direct name-to-game-list URL.

---

## HTML Parsing

### Player list page (`player_list.jsp`)

Each player row contains a link and a rating span:

```html
<td><a href="player.jsp?plid=2674">Alan Hensel</a></td>
<td>...</td>
<td><span title="2235.3">1. kyu</span></td>
```

**Extraction regex:**

```typescript
// Match player link: captures plid and display name
const rowRe = /href="player\.jsp\?plid=(\d+)">([^<]+)<\/a>/g;

// Rating span appears within ~300 chars of the link match
const ratingM = snippet.match(/<span[^>]*>([^<]+)<\/span>/);
```

The `title` attribute of the rating `<span>` contains the numeric Elo rating
(e.g. `"2235.3"`); the text content contains the display rank (e.g. `"1. kyu"`).

### Player game list page (`player_game_list.jsp`)

Verified against live HTML for plid=2674 (Alan Hensel), April 2026.
The page is a static HTML table — no JS injection of game rows.

Each `<tr>` contains exactly 6 `<td>` cells:

```html
<tr>
  <td align="left" bgcolor="#E9D101">
    <b><a href="/jsp/game/game.jsp?gid=2296844">#2296844</a></b>&nbsp;
  </td>
  <td bgcolor="#E9D101">Mirko Rahn&nbsp;</td>
  <td align="middle" nowrap="">&nbsp;<span title="2364.6">4. dan</span>&nbsp;</td>
  <td align="left" bgcolor="#E9D101">Twixt PP <span ...>Size 24</span>&nbsp;</td>
  <td align="right">20&nbsp;</td>
  <td align="center">lost&nbsp;</td>
</tr>
```

| Cell index | Content | Notes |
|---|---|---|
| 0 | `#NNNNNN` game link | `gid` extracted from `href` |
| 1 | Opponent display name | Plain text + `&nbsp;` |
| 2 | Opponent rating | Inside `<span title="Elo">rank</span>` |
| 3 | Tournament name + board size | "Twixt PP  Size 24" or tournament link |
| 4 | Move count | Integer |
| 5 | Result | `win` / `lost` / `draw` — **player's perspective** |

**Result format from game list** (`win`/`lost`/`draw`) differs from SGF format
(`B+`/`W+`/`0`). The color each player had is NOT available from this page —
only the SGF has `PB[name]`/`PW[name]`.

**All games in the sample are `Size 24`** — consistent with TwixT PP being
24×24 only on LG.

**Extraction approach** (split on `</tr>`, match `<td>` cells per row):

```typescript
for (const row of html.split('</tr>')) {
  const gidM = row.match(/href="\/jsp\/game\/game\.jsp\?gid=(\d+)"/);
  if (!gidM) continue;
  const cells = [...row.matchAll(/<td[^>]*>(.*?)<\/td>/gi)]
    .map(m => m[1].replace(/<[^>]+>/g, '').replace(/&nbsp;/g, ' ').trim());
  // cells[1]=opponent  cells[4]=moveCount  cells[5]=win|lost|draw
}
```

### Player profile page (`player.jsp?plid=PLID`)

Contains a stats table and a link to the game list:

```html
<a href="player_game_list.jsp?gtid=twixt&plid=2674">...</a>
```

This page is not fetched in the current implementation but is useful for
manually discovering the correct `plid` for a player.

---

## SGF Format (`.tsgf`)

LG uses a Go-style SGF dialect for TwixT. Files are plain text.

### SGF download URL

```
https://www.littlegolem.net/servlet/sgf/{id}/game{id}.tsgf
```

Both occurrences of `{id}` must be the same numeric game ID.
This URL is publicly accessible without login.

### File structure

```sgf
(;GM[21]FF[4]SZ[24]RU[PP]PB[player1]PW[player2]RE[B+]
;B[hd];W[qd];B[swap];W[fg];B[...] ...)
```

### Header properties

| Property | Meaning | Example |
|---|---|---|
| `GM[21]` | Game type: TwixT | Always 21 for TwixT |
| `FF[4]` | SGF format version | Always 4 |
| `SZ[24]` | Board size | 24 for standard TwixT PP |
| `RU[PP]` | Rules: Pen and Paper | Links are permanent (never removed) |
| `PB[name]` | Black player name | `PB[Alan Hensel]` |
| `PW[name]` | White player name | `PW[twixtbot]` |
| `RE[result]` | Result | See result values below |

### Result values

| `RE` value | Meaning |
|---|---|
| `B+` | Black wins |
| `W+` | White wins |
| `0` | Draw |
| `?` or empty | Unknown / not yet determined |

### Move encoding

Moves use **Go-style two-letter coordinate encoding**:

```
;B[hd]   → Black places peg at column h, row d
;W[swap] → White invokes the swap rule
;B[tt]   → Resign/pass sentinel (end of game marker)
;B[resign] → Resign (alternative form)
```

**Coordinate conversion** (letter → zero-based index):

```
col = charCodeAt(0) - 97   // 'a'=0, 'b'=1, ..., 'x'=23
row = charCodeAt(1) - 97   // 'a'=0, 'b'=1, ..., 'x'=23
```

So `hd` → `{x: 7, y: 3}` which corresponds to `h4` in standard TwixT
algebraic notation (where rows are 1-indexed).

This is consistent with the app's existing `ptFromString("h4")` convention:
`ptFromString("h4")` = `{x: 7, y: 3}` = `lgCoordToPoint("hd")`.

### Special move values

| Value | Meaning | Action |
|---|---|---|
| `swap` / `SWAP` | Swap rule invoked | Record as `'swap'` in move list |
| `resign` | Player resigned | Marks end of game; stop parsing |
| `tt` | Pass/resign sentinel | Marks end of game; stop parsing |
| `""` (empty) | End of game | Stop parsing |

Any move value that is not a valid two-letter `[a-x]{2}` coordinate and is
not `swap` should be ignored or treated as end-of-game.

### Full parsing example

```
(;GM[21]FF[4]SZ[24]RU[PP]PB[Alan Hensel]PW[twixtbot]RE[B+]
;B[hd];W[qd];B[swap];W[fg];B[ab];W[tt])
```

Parsed result:
```typescript
{
  id: "2545876",
  blackPlayer: "Alan Hensel",
  whitePlayer: "twixtbot",
  boardSize: 24,
  result: "B+",
  moves: [
    { x: 7, y: 3 },   // B[hd] — h4
    { x: 16, y: 3 },  // W[qd] — q4
    'swap',            // B[swap]
    { x: 5, y: 6 },   // W[fg] — f7
    { x: 0, y: 1 },   // B[ab] — a2
    // W[tt] → stop
  ]
}
```

---

## Data Types (TypeScript)

```typescript
/** A player returned by player search. */
interface PlayerResult {
  plid: string;    // Numeric LG player ID, e.g. "2674"
  name: string;    // Display name, e.g. "Alan Hensel"
  rating: string;  // Display rank, e.g. "1. kyu" — empty if unavailable
}

/** A game entry from a player's game list. */
interface GameSummary {
  id: string;           // Numeric game ID, e.g. "2545876"
  blackPlayer: string;  // "?" if not available from HTML
  whitePlayer: string;  // "?" if not available from HTML
  result: string;       // Raw SGF result: "B+", "W+", "0", "?" 
  boardSize: number;    // Always 24 for TwixT PP
  moveCount: number;    // 0 if not parsed from game list HTML
}

/** A fully parsed game from an SGF file. */
interface ParsedGame {
  id: string;
  blackPlayer: string;
  whitePlayer: string;
  boardSize: number;
  result: string;
  moves: MoveRecord[];  // Point[] | 'swap'
}
```

---

## Implementation Files

| File | Purpose |
|---|---|
| `webapp/src/lg-api.ts` | HTTP client: player search, game list fetch, SGF fetch |
| `webapp/src/lg-sgf.ts` | SGF parser: `parseTSGF(text, id)` → `ParsedGame` |

---

## Known Limitations & Gotchas

- **Game list HTML gives only IDs**: `player_game_list.jsp` HTML has player
  names and results in its table, but extracting them requires full table
  parsing. The current implementation only extracts game IDs and fetches each
  game's SGF for full metadata. This means opening a game list triggers one
  additional SGF fetch per game if full metadata is needed.

- **No direct name → games URL**: LG does not expose a URL that goes directly
  from a player name to their game list. The `plid` numeric ID is always
  required. The two-step search flow is mandatory.

- **`twixt.PP` vs `twixt`**: The game type identifier in `gtid` must be
  `twixt`, not `twixt.PP`. Using `twixt.PP` silently returns no results on
  `player_game_list.jsp`.

- **`player_game_list_txt.jsp` not public**: The plain-text export endpoint
  (`_txt.jsp`) does not appear to be accessible without login. Always use the
  HTML endpoint.

- **Ongoing games**: The game list page includes ongoing games (no `RE`
  property or `RE[?]` in their SGF). The SGF for an ongoing game may be
  partial or unavailable.

- **Login redirect detection**: If LG ever starts requiring login for SGF
  downloads, the response will be an HTML page rather than SGF text.
  Detect with: `text.includes('GM[21]')` must be true, and absence of
  `<html` or `login` in the response body.

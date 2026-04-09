# Little Golem Integration — Technical Reference

> **Feature ideas, priorities, and status are tracked in `docs/planned-features.md`
> (section 6). This document contains technical background for the LG Network Play
> feature (auth, move submission, security constraints) which is blocked pending
> investigation of live LG endpoints.**

---

## LG Network Play — play against other humans on Little Golem

> **Blocked — open investigation items must be resolved before implementation
> can begin (see below).**

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
